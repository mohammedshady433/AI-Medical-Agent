import hashlib
import numpy as np
from app.rag.library import normalize


async def test_dedup_and_index_reuse(settings, database, library, tmp_path, monkeypatch):
    cfg = settings.snapshot()
    cfg["documents"]["embedding"]["batch_size"] = 2
    settings.update({"documents": cfg["documents"]})
    calls = []

    async def embedding(texts):
        calls.append(texts)
        return np.asarray([[1, 0, 0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr(library, "embed", embedding)
    content = "Training budget is 8000 EGP. The operations budget is 12000 EGP."
    path = tmp_path / "upload.txt"
    path.write_text(content)
    digest = hashlib.sha256(content.encode()).hexdigest()
    doc_id, created = library.register(path, "budget.txt", digest)
    assert created
    await library.ingest(doc_id)
    count = len(calls)
    await library.index(doc_id)
    assert len(calls) == count
    duplicate = tmp_path / "copy.txt"
    duplicate.write_text(content)
    assert library.register(duplicate, "copy.txt", digest) == (doc_id, False)
    assert len(database.documents()) == 1
    await library.delete(doc_id)
    assert database.documents() == []


async def test_cross_language_hybrid_pipeline(settings, database, library, monkeypatch):
    # Controlled vectors exercise retrieval plumbing. Live BGE-M3 semantic quality is tested separately.
    async def embedding(texts):
        return np.asarray([[1, 0, 0]], dtype=np.float32)

    monkeypatch.setattr(library, "embed", embedding)
    with database.connect() as db:
        db.execute(
            "INSERT INTO documents(id,name,digest,status,created) VALUES('d','English budget.txt','hash','ready',1)"
        )
        db.execute(
            "INSERT INTO chunks(document_id,location,text,normalized,vector,embedding_model) VALUES(?,?,?,?,?,?)",
            (
                "d",
                "Page 1",
                "Training budget is 8000 EGP",
                normalize("Training budget is 8000 EGP"),
                np.asarray([1, 0, 0], dtype=np.float32).tobytes(),
                "bge-m3:567m",
            ),
        )
    result = await library.search("ميزانية التدريب")
    assert result["sources"][0]["name"] == "English budget.txt"


async def test_missing_embeddings_report_keyword_only(settings, database, library, tmp_path, monkeypatch):
    async def unavailable(texts):
        raise ValueError("No model")

    monkeypatch.setattr(library, "embed", unavailable)
    path = tmp_path / "upload.txt"
    path.write_text("Training budget 8000 EGP")
    doc_id, _ = library.register(path, "budget.txt", "hash")
    await library.ingest(doc_id)
    assert database.documents()[0]["status"] == "keyword_only"
    result = await library.search("budget")
    assert result["sources"] and result["warnings"]
