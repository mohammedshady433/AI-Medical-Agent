import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from app.agents.assistant import Assistant
from app.core.settings import Settings
from app.main import create_app
from app.rag.library import Library
from app.rag.transfer import export_library, import_library
from app.storage.database import Database


def add_file(library, name="products.csv", content="product,price\nCedar,725\nPalm,430\n"):
    source = library.db.data / name
    source.write_text(content, encoding="utf8")
    return library.register(source, name, hashlib.sha256(content.encode()).hexdigest())[0]


def test_upgrade_backups_include_committed_wal_and_refuses_newer_schema(tmp_path):
    path = tmp_path / "ems.sqlite3"
    with sqlite3.connect(path) as old:
        old.executescript("PRAGMA journal_mode=WAL; CREATE TABLE documents(id TEXT PRIMARY KEY,name TEXT,digest TEXT UNIQUE,status TEXT,progress INTEGER DEFAULT 0,error TEXT DEFAULT '',warnings TEXT DEFAULT '[]',created REAL); PRAGMA user_version=1;")
        old.execute("INSERT INTO documents(id,name,digest,status,created) VALUES('original','Keep me','hash','ready',1)")
        old.commit()
        db = Database(tmp_path)
    assert db.documents()[0]["name"] == "Keep me"
    backups = list((tmp_path / "backups").glob("*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as saved:
        assert saved.execute("PRAGMA user_version").fetchone()[0] == 1
        assert saved.execute("SELECT name FROM documents").fetchone()[0] == "Keep me"
    Database(tmp_path)
    assert len(list((tmp_path / "backups").glob("*.sqlite3"))) == 1
    with db.connect() as connection:
        connection.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="newer EMS"):
        Database(tmp_path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 999


async def test_memory_restart_search_exclusion_and_source_preview(library, settings, monkeypatch):
    monkeypatch.setattr(library, "embed", AsyncMock(side_effect=ValueError("No local model")))
    doc_id = add_file(library)
    await library.ingest(doc_id)
    library.db.save_knowledge("Use the product library for product questions.")
    library.update(doc_id, "Prices include VAT.", True, True)
    restarted = Library(Database(settings.data), settings)
    monkeypatch.setattr(restarted, "embed", library.embed)
    assert "product library" in restarted.guidance()
    result = await restarted.search("Cedar")
    assert "price: 725" in result["sources"][0]["text"]
    assert result["sources"][0]["owner_instructions"] == "Prices include VAT."
    assert restarted.preview(doc_id)["total"] == 2
    restarted.update(doc_id, "Prices include VAT.", False, True)
    assert not (await restarted.search("Cedar"))["sources"]
    assert not restarted.overview()["sources"]
    assert restarted.path(doc_id).is_file()


def test_transfer_merges_duplicates_and_requires_instruction_review(library, tmp_path):
    doc_id = add_file(library)
    library.update(doc_id, "Use EGP.", True, True)
    library.db.save_knowledge("Answer staff policy questions.")
    target = tmp_path / "bundle.ems-library"
    export_library(library, target)
    destination = Library(Database(tmp_path / "another-device"), Settings(tmp_path / "another-device"))
    destination.db.save_knowledge("Keep my existing instructions.")
    result = import_library(destination, target)
    assert len(result["created"]) == 1
    assert result["instructions_to_review"] == "Answer staff policy questions."
    row = destination.db.documents()[0]
    assert row["instructions"] == "Use EGP." and row["instructions_enabled"] == 0
    assert destination.db.knowledge()["instructions"] == "Keep my existing instructions."
    destination.update(row["id"], "Reviewed custom instructions", False, True)
    again = import_library(destination, target)
    assert again["duplicates"] == 1 and again["created"] == []
    assert destination.db.documents()[0]["instructions"] == "Reviewed custom instructions"
    assert destination.db.documents()[0]["enabled"] == 0
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == {"manifest.json", "files/" + doc_id + ".csv"}


@pytest.mark.parametrize("attack", ["hash", "path", "duplicate", "too_large", "unexpected", "instructions"])
def test_bad_bundle_never_partially_imports(library, tmp_path, attack):
    add_file(library)
    good = tmp_path / "good.ems-library"
    export_library(library, good)
    with zipfile.ZipFile(good) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    if attack == "hash": manifest["files"][0]["sha256"] = "0" * 64
    if attack == "path": manifest["files"][0]["name"] = "../outside.txt"
    if attack == "duplicate": manifest["files"] *= 2
    if attack == "instructions": manifest["files"][0]["instructions"] = "x" * 1001
    if attack == "unexpected": entries["extra.txt"] = b"extra"
    if attack == "too_large": manifest["files"] *= 501
    entries["manifest.json"] = json.dumps(manifest).encode()
    bad = tmp_path / "bad.ems-library"
    with zipfile.ZipFile(bad, "w") as archive:
        for name, value in entries.items(): archive.writestr(name, value)
    dest = Library(Database(tmp_path / "destination"), Settings(tmp_path / "destination"))
    with pytest.raises(ValueError): import_library(dest, bad)
    assert not dest.db.documents() and not list(dest.folder.iterdir())
    assert not (tmp_path / "outside.txt").exists()


async def test_saved_memory_and_derived_history_never_leak_when_cloud_sharing_is_off(settings, database, library, secrets):
    database.save_knowledge("PRIVATE-MEMORY-042")
    settings.update({"mode": "offline"})
    captured = []
    class Provider:
        adapter = "ollama"
        def __init__(self, *args): pass
        async def stream(self, messages, tools=()):
            captured.append(json.dumps(messages))
            yield {"type": "token", "text": "PRIVATE-DERIVED-ANSWER" if len(captured) == 1 else "Hello"}
    assistant = Assistant(settings, secrets, database, library, Provider)
    chat = database.new_chat()
    first = [e async for e in assistant.reply(chat, "Recall my working instructions")]
    assert "PRIVATE-MEMORY-042" in captured[0]
    assert first[-1]["meta"]["document_context"]
    settings.update({"mode": "online", "privacy": {"cloud_document_context": False}})
    _ = [e async for e in assistant.reply(chat, "Hello")]
    assert "PRIVATE-" not in captured[-1]
    assert "Recall my working instructions" not in captured[-1]


def test_knowledge_api_validation_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(Library, "embed", AsyncMock(side_effect=ValueError("Missing model")))
    headers = {"Authorization": "Bearer token"}
    app = create_app(tmp_path, "token")
    with TestClient(app) as client:
        assert client.get("/knowledge").status_code == 401
        assert client.patch("/knowledge", headers=headers, json={"instructions": "x" * 2001}).status_code == 422
        assert client.patch("/knowledge", headers=headers, json={"instructions": "Remember units: EGP"}).status_code == 200
        doc = client.post("/documents", headers=headers, files={"file": ("data.txt", b"Product Cedar costs 725 EGP.")}).json()["id"]
        response = client.patch(f"/documents/{doc}", headers=headers, json={"instructions": "Cite exact values", "enabled": False})
        assert response.status_code == 200
        assert client.get(f"/documents/{doc}?offset=-1", headers=headers).status_code == 400
        assert client.get(f"/documents/{doc}", headers=headers).json()["document"]["enabled"] == 0
        export_id = client.post("/knowledge/export", headers=headers).json()["id"]
        bundle = client.get("/knowledge/exports/" + export_id, headers=headers)
        assert bundle.status_code == 200
        assert client.get("/knowledge/exports/" + export_id, headers=headers).status_code == 404
        result = client.post("/knowledge/import", headers=headers, files={"file": ("knowledge.ems-library", bundle.content)})
        assert result.status_code == 200 and result.json()["duplicates"] == 1
    assert Database(tmp_path).knowledge()["instructions"] == "Remember units: EGP"


def test_older_app_version_marker_does_not_erase_new_memory(database):
    database.save_knowledge('Keep the new memory')
    with database.connect() as db:
        db.execute('PRAGMA user_version=1')
    reopened = Database(database.data)
    assert reopened.knowledge()['instructions'] == 'Keep the new memory'


async def test_keyword_search_does_not_wait_on_an_unavailable_semantic_model(library, monkeypatch):
    doc_id = add_file(library)
    monkeypatch.setattr(library, 'embed', AsyncMock(side_effect=ValueError('offline model absent')))
    await library.ingest(doc_id)
    library.embed.reset_mock()
    result = await library.search('Cedar')
    assert result['sources'] and result['warnings']
    library.embed.assert_not_called()
