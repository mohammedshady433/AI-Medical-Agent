"""Content-addressed ingestion and persistent multilingual hybrid retrieval."""

import asyncio
import json
import logging
import re
import time
import uuid
import numpy as np
import httpx
from app.documents.extract import extract, chunks
from app.core.settings import local_url


def normalize(text):
    text = re.sub("[\u064b-\u065f\u0670\u0640]", "", text.lower())
    return text.translate(str.maketrans("أإآى", "اااي"))


class Library:
    def __init__(self, database, settings):
        self.db, self.settings = database, settings
        self.folder = database.data / "documents"
        self.folder.mkdir(exist_ok=True)
        self.queue_lock = asyncio.Lock()
        self.query_cache = {}

    async def embed(self, texts):
        cfg = self.settings.snapshot()["documents"]["embedding"]
        if not local_url(cfg["endpoint"]):
            raise ValueError("Embeddings must stay local.")
        async with httpx.AsyncClient(timeout=120, trust_env=False, follow_redirects=False) as client:
            show = await client.post(cfg["endpoint"].rstrip("/") + "/api/show", json={"model": cfg["model"]})
            show.raise_for_status()
            if show.json().get("remote_host") or show.json().get("remote_model"):
                raise ValueError("Embedding model must reside on this computer.")
            response = await client.post(
                cfg["endpoint"].rstrip("/") + "/api/embed",
                json={
                    "model": cfg["model"],
                    "input": texts,
                    "truncate": False,
                    "options": {"num_gpu": 0},
                    "keep_alive": "10m",
                },
            )
            response.raise_for_status()
            array = np.asarray(response.json()["embeddings"], dtype=np.float32)
            if array.ndim != 2 or len(array) != len(texts) or not np.all(np.isfinite(array)):
                raise ValueError("Invalid embedding dimensions.")
            return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-9)

    def register(self, path, name, digest):
        with self.db.connect() as db:
            existing = db.execute("SELECT id FROM documents WHERE digest=?", (digest,)).fetchone()
            if existing:
                return existing["id"], False
            doc_id = uuid.uuid4().hex
            dest = self.folder / (doc_id + path.suffix.lower())
            path.replace(dest)
            db.execute(
                "INSERT INTO documents(id,name,digest,status,created) VALUES(?,?,?,?,?)",
                (doc_id, name, digest, "queued", time.time()),
            )
        return doc_id, True

    def path(self, doc_id):
        if not re.fullmatch("[a-f0-9]{32}", doc_id):
            raise ValueError("Invalid document ID.")
        found = list(self.folder.glob(doc_id + ".*"))
        if not found:
            raise ValueError("Document file is missing.")
        return found[0]

    async def ingest(self, doc_id):
        async with self.queue_lock:
            cfg = self.settings.snapshot()["documents"]
            try:
                self.status(doc_id, "extracting", 5)
                parts, warnings = await asyncio.to_thread(extract, self.path(doc_id), cfg)
                if sum(len(p[1]) for p in parts) > cfg["max_extracted_chars"]:
                    raise ValueError("Extracted text exceeds the configured limit.")
                rows = list(chunks(parts, cfg))
                if not rows:
                    raise ValueError(
                        "No readable text was found. Configure OCR for scanned documents or images."
                    )
                with self.db.connect() as db:
                    db.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
                    db.executemany(
                        "INSERT INTO chunks(document_id,location,text,normalized) VALUES(?,?,?,?)",
                        [(doc_id, loc, text, normalize(text)) for loc, text in rows],
                    )
                    db.execute(
                        "UPDATE documents SET warnings=? WHERE id=?",
                        (json.dumps(warnings, ensure_ascii=False), doc_id),
                    )
                await self.index(doc_id)
            except Exception as exc:
                logging.getLogger(__name__).exception("Document extraction failed: %s", doc_id)
                self.status(
                    doc_id,
                    "error",
                    0,
                    str(exc)[:400]
                    if isinstance(exc, ValueError)
                    else "Extraction failed. Check the file and configured document tools.",
                )

    def status(self, doc_id, status, progress, error=""):
        with self.db.connect() as db:
            db.execute(
                "UPDATE documents SET status=?,progress=?,error=? WHERE id=?",
                (status, progress, error, doc_id),
            )

    async def index(self, doc_id):
        cfg = self.settings.snapshot()["documents"]["embedding"]
        with self.db.connect() as db:
            rows = db.execute(
                "SELECT id,text FROM chunks WHERE document_id=? AND (vector IS NULL OR embedding_model!=?)",
                (doc_id, cfg["model"]),
            ).fetchall()
        self.status(doc_id, "indexing", 35)
        try:
            for offset in range(0, len(rows), cfg["batch_size"]):
                batch = rows[offset : offset + cfg["batch_size"]]
                vectors = await self.embed([r["text"] for r in batch])
                with self.db.connect() as db:
                    db.executemany(
                        "UPDATE chunks SET vector=?,embedding_model=? WHERE id=?",
                        [(vec.tobytes(), cfg["model"], row["id"]) for row, vec in zip(batch, vectors)],
                    )
                self.status(doc_id, "indexing", 35 + int(60 * (offset + len(batch)) / max(1, len(rows))))
            self.status(doc_id, "ready", 100)
        except (httpx.HTTPError, ValueError, KeyError):
            self.status(
                doc_id,
                "keyword_only",
                100,
                "Text is ready. Install/start the local embedding model and retry indexing for cross-language semantic search.",
            )

    def source(self, row):
        source = {
            "reference": f"[D{row['id']}]",
            "document_id": row["document_id"],
            "name": row["name"],
            "location": row["location"],
            "text": row["text"],
        }
        if row.get("instructions_enabled", True) and row.get("instructions"):
            source["owner_instructions"] = row["instructions"]
        return source

    def guidance(self):
        """Explicit owner memory and a bounded file catalog, not extracted instructions."""
        profile = self.db.knowledge()["instructions"]
        with self.db.connect() as db:
            rows = db.execute("SELECT name FROM documents WHERE enabled=1 ORDER BY created DESC LIMIT 20").fetchall()
        if not profile and not rows:
            return ""
        return json.dumps({"saved_owner_instructions": profile, "available_files": [r["name"][:80] for r in rows],
            "catalog_note": "At most 20 recent names. Search the library for facts; a name is not evidence."}, ensure_ascii=False)

    def update(self, doc_id, instructions, enabled, instructions_enabled):
        with self.db.connect() as db:
            changed = db.execute("UPDATE documents SET instructions=?,enabled=?,instructions_enabled=? WHERE id=?",
                (instructions.strip(), int(enabled), int(instructions_enabled), doc_id))
            if not changed.rowcount:
                raise ValueError("This file is no longer in the library. Refresh and try again.")

    def preview(self, doc_id, offset=0):
        with self.db.connect() as db:
            doc = db.execute("SELECT id,name,instructions,enabled,instructions_enabled FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not doc:
                raise ValueError("This file is no longer in the library. Refresh and try again.")
            rows = db.execute("SELECT id,location,text FROM chunks WHERE document_id=? ORDER BY id LIMIT 20 OFFSET ?", (doc_id, offset)).fetchall()
            count = db.execute("SELECT count(*) FROM chunks WHERE document_id=?", (doc_id,)).fetchone()[0]
        return {"document": dict(doc), "passages": [dict(r) for r in rows], "total": count, "offset": offset}

    async def search(self, query):
        cfg = self.settings.snapshot()["documents"]
        words = re.findall(r"\w+", normalize(query))[:32]
        expression = " OR ".join('"' + w + '"' for w in words)
        scores, candidates, warnings = {}, {}, []
        with self.db.connect() as db:
            has_vectors = db.execute("SELECT 1 FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.enabled=1 AND c.vector IS NOT NULL AND c.embedding_model=? LIMIT 1", (cfg["embedding"]["model"],)).fetchone()
            if expression:
                rows = db.execute(
                    "SELECT c.*,d.name,d.instructions,d.instructions_enabled FROM chunks_fts f JOIN chunks c ON c.id=f.rowid JOIN documents d ON d.id=c.document_id WHERE chunks_fts MATCH ? AND d.enabled=1 ORDER BY bm25(chunks_fts) LIMIT 50",
                    (expression,),
                ).fetchall()
                for rank, row in enumerate(rows):
                    candidates[row["id"]] = dict(row)
                    scores[row["id"]] = 1 / (60 + rank)
        try:
            if not has_vectors:
                raise ValueError("No semantic index is available yet.")
            cache_key = (cfg["embedding"]["model"], cfg["embedding"]["endpoint"], query)
            vector = self.query_cache.get(cache_key)
            if vector is None:
                vector = (await self.embed([query]))[0]
                if len(self.query_cache) > 128:
                    self.query_cache.clear()
                self.query_cache[cache_key] = vector
            best = []
            with self.db.connect() as db:
                cursor = db.execute(
                    "SELECT c.*,d.name,d.instructions,d.instructions_enabled FROM chunks c JOIN documents d ON d.id=c.document_id WHERE vector IS NOT NULL AND embedding_model=? AND d.enabled=1",
                    (cfg["embedding"]["model"],),
                )
                while batch := cursor.fetchmany(256):
                    for row in batch:
                        v = np.frombuffer(row["vector"], dtype=np.float32)
                        if v.shape != vector.shape:
                            continue
                        similarity = float(v @ vector)
                        if similarity > 0.28:
                            best.append((similarity, dict(row)))
                    best = sorted(best, key=lambda r: r[0], reverse=True)[:50]
            for rank, (_, row) in enumerate(best):
                candidates[row["id"]] = row
                scores[row["id"]] = scores.get(row["id"], 0) + 1 / (60 + rank)
        except (httpx.HTTPError, ValueError, KeyError):
            warnings.append("Semantic search is unavailable; these results use keyword search only.")
        ordered = sorted(candidates.values(), key=lambda r: scores[r["id"]], reverse=True)
        sources, length = [], 0
        for row in ordered:
            if len(sources) >= cfg["retrieval_limit"] or length + len(row["text"]) > cfg["context_chars"]:
                break
            sources.append(self.source(row))
            length += len(row["text"])
        return {"sources": sources, "warnings": warnings}

    def overview(self):
        # Representative excerpts, not a fabricated/precomputed whole-library summary.
        sources, names, total = [], [], 0
        with self.db.connect() as db:
            docs = db.execute(
                "SELECT id,name,status,instructions,instructions_enabled FROM documents WHERE enabled=1 ORDER BY created DESC LIMIT 50"
            ).fetchall()
            for doc in docs:
                names.append(dict(doc))
                rows = db.execute(
                    "SELECT c.*,? AS name FROM chunks c WHERE document_id=? ORDER BY id",
                    (doc["name"], doc["id"]),
                ).fetchall()
                for idx in sorted(set([0, len(rows) // 2, len(rows) - 1])) if rows else []:
                    row = {**dict(rows[idx]), "instructions": doc["instructions"], "instructions_enabled": doc["instructions_enabled"]}
                    if total + len(row["text"]) > 10000:
                        break
                    sources.append(self.source(row))
                    total += len(row["text"])
        return {
            "documents": names,
            "sources": sources,
            "coverage": "Representative excerpts only. Search individual topics for more detail; do not claim a complete audit.",
        }

    async def delete(self, doc_id):
        async with self.queue_lock:
            path = self.path(doc_id)
            with self.db.connect() as db:
                db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            path.unlink(missing_ok=True)
