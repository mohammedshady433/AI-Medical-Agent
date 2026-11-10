"""One shared SQLite library across modes; WAL permits reads during ingestion."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager


class Database:
    SCHEMA_VERSION = 2

    def __init__(self, data):
        self.data = data
        self.path = data / "ems.sqlite3"
        data.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > self.SCHEMA_VERSION:
                raise ValueError("This library was opened by a newer EMS. Install that version; do not reset your data.")
            if version and version < self.SCHEMA_VERSION:
                folder = data / "backups"
                folder.mkdir(exist_ok=True)
                backup = folder / (f"before-schema-{version}-" + uuid.uuid4().hex + ".sqlite3")
                with sqlite3.connect(backup) as target:
                    db.backup(target)
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY,title TEXT NOT NULL,updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY,chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,role TEXT NOT NULL,content TEXT NOT NULL,meta TEXT NOT NULL DEFAULT '{}',created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,name TEXT NOT NULL,digest TEXT UNIQUE NOT NULL,status TEXT NOT NULL,progress INTEGER DEFAULT 0,error TEXT DEFAULT '',warnings TEXT DEFAULT '[]',created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY,document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,location TEXT NOT NULL,text TEXT NOT NULL,normalized TEXT NOT NULL,vector BLOB,embedding_model TEXT);
            CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(document_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(normalized,content='chunks',content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN INSERT INTO chunks_fts(rowid,normalized) VALUES(new.id,new.normalized); END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN INSERT INTO chunks_fts(chunks_fts,rowid,normalized) VALUES('delete',old.id,old.normalized); END;
            """)
            # SQLite DDL and its version marker commit together. A failed migration
            # leaves the previous schema usable; the snapshot also includes WAL data.
            if version < 2:
                db.execute("BEGIN IMMEDIATE")
                # Older releases reset user_version on every launch. Preserve an
                # existing v2 table even if one of those releases was opened.
                columns = {r["name"] for r in db.execute("PRAGMA table_info(documents)")}
                for name, declaration in [("instructions", "TEXT NOT NULL DEFAULT ''"), ("enabled", "INTEGER NOT NULL DEFAULT 1"), ("instructions_enabled", "INTEGER NOT NULL DEFAULT 1")]:
                    if name not in columns:
                        db.execute(f"ALTER TABLE documents ADD COLUMN {name} {declaration}")
                db.execute("CREATE TABLE IF NOT EXISTS knowledge_profile(id INTEGER PRIMARY KEY CHECK(id=1), instructions TEXT NOT NULL, updated REAL NOT NULL)")
                db.execute("INSERT OR IGNORE INTO knowledge_profile VALUES(1,'',?)", (time.time(),))
                db.execute("PRAGMA user_version=2")

    def knowledge(self):
        with self.connect() as db:
            return dict(db.execute("SELECT instructions,updated FROM knowledge_profile WHERE id=1").fetchone())

    def save_knowledge(self, instructions):
        with self.connect() as db:
            db.execute("UPDATE knowledge_profile SET instructions=?,updated=? WHERE id=1", (instructions.strip(), time.time()))
        return self.knowledge()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def chats(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM chats ORDER BY updated DESC")]

    def new_chat(self):
        cid = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO chats VALUES(?,?,?)", (cid, "New conversation", time.time()))
        return cid

    def messages(self, chat_id):
        with self.connect() as db:
            return [
                {**dict(r), "meta": json.loads(r["meta"])}
                for r in db.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY id", (chat_id,))
            ]

    def add_message(self, chat_id, role, content, meta=None):
        with self.connect() as db:
            db.execute(
                "INSERT INTO messages(chat_id,role,content,meta,created) VALUES(?,?,?,?,?)",
                (chat_id, role, content, json.dumps(meta or {}, ensure_ascii=False), time.time()),
            )
            db.execute("UPDATE chats SET updated=? WHERE id=?", (time.time(), chat_id))
            if role == "user":
                db.execute(
                    "UPDATE chats SET title=? WHERE id=? AND title='New conversation'",
                    (content[:70], chat_id),
                )

    def documents(self):
        with self.connect() as db:
            return [
                {**dict(r), "warnings": json.loads(r["warnings"])}
                for r in db.execute(
                    "SELECT d.*, (SELECT count(*) FROM chunks c WHERE c.document_id=d.id) AS chunks FROM documents d ORDER BY created DESC"
                )
            ]
