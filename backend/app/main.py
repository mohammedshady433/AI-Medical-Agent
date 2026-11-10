"""Authenticated loopback API. Electron holds the session token, never renderer storage."""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import httpx
import time
from pathlib import Path
import secrets as random_secrets
from contextlib import asynccontextmanager, aclosing
from collections import deque
from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field, ConfigDict
from app.core.settings import Settings, DATA
from app.core.secrets import SecretStore
from app.storage.database import Database
from app.rag.library import Library
from app.rag.transfer import export_library, import_library, MAX_BYTES
from app.agents.assistant import Assistant, voice_config
from app.providers.client import Provider, ProviderError
from app.speech.service import Speech
from app.speech.reply import prepared_voice_reply
from app.services.diagnostics import hardware, build_info
from app.documents.extract import SUPPORTED
from app.services.local_runtime import LocalRuntime
from app.services.model_manager import ModelManager
from app.services.gpu import GpuCoordinator
from app.core.options import resident_voice
from app.extensions.research import Research
from app.extensions.vision import Vision
from app.extensions.mcp import Mcp
from app.services.activity import Activity
from app.services.setup import Setup, recover_installed, remember_installation


def create_app(data=DATA, token=None, provider_factory=Provider):
    settings = Settings(Path(data))
    recover_installed(settings)
    secrets = SecretStore(settings.data)
    db = Database(settings.data)
    library = Library(db, settings)
    gpu = GpuCoordinator(settings)
    mcp = Mcp(settings, secrets)
    assistant = Assistant(settings, secrets, db, library, provider_factory, gpu, mcp)
    vision = Vision(settings, gpu)
    speech = Speech(settings, secrets, gpu=gpu)
    gpu.offload_speech = speech.offload
    runtime = LocalRuntime(settings)
    model_manager = ModelManager(settings)
    setup = Setup(settings, speech, runtime, model_manager)
    research = Research(settings, secrets, gpu, provider_factory)
    jobs, active, busy = set(), {}, set()
    synthesis_jobs, speech_generation = {}, 0
    preparing = asyncio.Lock()
    preparations, cancelled_preparations = {}, deque(maxlen=32)
    activity = Activity(settings.data)
    exports = {}
    api_token = token or os.environ.get("EMS_SESSION_TOKEN")
    if not api_token:
        raise RuntimeError("EMS_SESSION_TOKEN is required; launch the backend through backend/launcher.py or scripts/test_console.py.")

    @asynccontextmanager
    async def lifespan(app):
        await runtime.start()
        voice = settings.snapshot()["speech"]
        if (
            voice["tts"].get("prewarm", True)
            and voice["tts"]["provider"] in ("kemetone", "chatterbox")
            and voice["tts"]["auto_speak"]
            and speech.status('ar')['ready']
        ):

            async def prepare_voice():
                try:
                    # Warm only installed local assets; never play or send this audio.
                    async with preparing: await speech.local("warmup")
                except Exception:
                    pass  # A real voice request reports setup errors to the user.

            schedule(prepare_voice())
        with db.connect() as connection:
            connection.execute(
                "UPDATE documents SET status='interrupted',error='Processing was interrupted. Retry this document.' WHERE status IN ('queued','extracting','indexing')"
            )
        yield
        pending = set(jobs) | set(synthesis_jobs) | set(preparations.values())
        for job in pending:
            job.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await setup.close()
        await speech.close()
        await research.close()
        await model_manager.close()
        await runtime.close()
        for path in exports.values():
            path.unlink(missing_ok=True)
        activity.close()

    async def authorize(authorization: str = Header(default="")):
        if not hmac.compare_digest(authorization, "Bearer " + api_token):
            raise HTTPException(401, "Unauthorized desktop session.")

    app = FastAPI(
        title="EMS Desktop API",
        lifespan=lifespan,
        dependencies=[Depends(authorize)],
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings, app.state.db, app.state.library = settings, db, library
    app.state.research = research
    app.state.mcp, app.state.vision = mcp, vision
    app.state.speech = speech
    app.state.setup = setup

    def setup_idle():
        if active or synthesis_jobs or preparations or preparing.locked():
            raise ValueError('Finish the current conversation before changing setup.')

    @app.get('/setup/status')
    async def setup_status():
        return await setup.status()

    @app.post('/setup/repair')
    async def setup_repair():
        setup_idle()
        return await setup.repair()

    class SetupRequest(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        component: str = Field(pattern=r'^(voice|assistant)$')
        accepted_download: bool

    @app.post('/setup/install')
    async def setup_install(body: SetupRequest):
        setup_idle()
        return await setup.start(body.component, body.accepted_download)

    @app.post('/setup/cancel')
    async def setup_cancel():
        await setup.close()
        return dict(setup.state)

    @app.middleware("http")
    async def timings(request, call_next):
        reference = random_secrets.token_hex(4)
        request.state.reference = reference
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-EMS-Request-ID"] = reference
            return response
        finally:
            if request.url.path != "/diagnostics/activity":
                activity.add(reference, request.method, request.url.path, status, time.perf_counter() - started)

    @app.exception_handler(ProviderError)
    async def provider_error(request, exc):
        return JSONResponse(status_code=400, content=exc.event())

    @app.exception_handler(ValueError)
    async def input_error(request, exc):
        return JSONResponse(status_code=400, content={"message": str(exc)[:600]})

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        reference = getattr(request.state, "reference", random_secrets.token_hex(4))
        logging.getLogger(__name__).error("Request failed [%s]: %s %s", reference, request.method, request.url.path, exc_info=exc)
        return JSONResponse(status_code=500, content={"message": f"The local service could not complete the request. Restart EMS; error reference {reference} is in the application log."})

    def schedule(coro):
        task = asyncio.create_task(coro)
        jobs.add(task)
        task.add_done_callback(jobs.discard)

    def schedule_ingest(doc_id):
        busy.add(doc_id)
        async def ingest():
            try:
                await library.ingest(doc_id)
            finally:
                busy.discard(doc_id)
        schedule(ingest())

    @app.get("/health")
    async def health():
        return {"ok": True, "version": build_info()["version"], "data_dir": str(settings.data)}

    def research_view(record, detail=False):
        result = {key: record[key] for key in ("id", "query", "deep", "generate", "mode", "model", "language", "status", "stage", "budget", "model_calls", "input_chars", "searches", "page_requests", "warning", "updated")}
        result["source_count"] = len(record["sources"])
        if detail:
            result["report"] = record["report"]
            result["sources"] = [{k: s[k] for k in ("title", "url", "reference")} for s in record["sources"]]
        return result

    class ResearchStart(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        query: str = Field(min_length=3, max_length=500)
        deep: bool = False
        generate: bool = False
        confirm_network: bool = False
        confirm_cloud: bool = False

    @app.get("/research")
    async def research_list():
        return {"sessions": [research_view(r) for r in research.list()]}

    @app.post("/research")
    async def research_start(body: ResearchStart):
        return research_view(research.start(**body.model_dump()), True)

    @app.get("/research/{rid}")
    async def research_get(rid: str):
        return research_view(research.load(rid), True)

    @app.post("/research/{rid}/pause")
    async def research_pause(rid: str):
        return research_view(research.pause(rid), True)

    class ResumeResearch(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        confirm_cloud: bool = False

    @app.post("/research/{rid}/resume")
    async def research_resume(rid: str, body: ResumeResearch):
        return research_view(research.resume(rid, body.confirm_cloud), True)

    @app.delete("/research/{rid}")
    async def research_delete(rid: str):
        research.delete(rid)
        return {"deleted": True}

    class ImageQuestion(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        image: str = Field(max_length=12000000)
        question: str = Field(min_length=1, max_length=1000)

    @app.post("/vision")
    async def image_question(body: ImageQuestion):
        try: return await vision.analyze(body.image, body.question)
        except (TimeoutError, httpx.HTTPError):
            raise ValueError("Local image analysis could not finish. Check the installed model and Ollama in Diagnostics.") from None

    @app.get("/mcp")
    async def mcp_status():
        rows = []
        for server in settings.snapshot()["assistant"]["mcp_servers"]:
            try: catalog = mcp.catalog(server["id"])
            except ValueError: catalog = []
            rows.append({**server, "has_key": bool(secrets.get("mcp:" + server["id"])), "tools": catalog})
        return {"servers": rows}

    @app.post("/mcp/{sid}/connect")
    async def mcp_connect(sid: str):
        return await mcp.connect(sid)

    class McpRequest(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        server: str = Field(max_length=32)
        tool: str = Field(max_length=128)
        arguments: dict

    @app.post("/mcp/prepare")
    async def mcp_prepare(body: McpRequest):
        return mcp.prepare(body.server, body.tool, body.arguments)

    class ConfirmCommand(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        id: str = Field(pattern=r"^[a-f0-9]{32}$")
        approve: bool

    @app.post("/mcp/confirm")
    async def mcp_confirm(body: ConfirmCommand):
        return await mcp.confirm(body.id, body.approve)

    @app.get("/settings")
    async def get_settings():
        cfg = settings.snapshot()
        return {
            "config": cfg,
            "credentials": {
                key: bool(secrets.get(key, p.get("secret_env", ""))) for key, p in cfg["providers"].items()
            },
            "azure_key": bool(secrets.get("azure", "AZURE_SPEECH_KEY")),
            "brave_key": bool(secrets.get("brave", "BRAVE_SEARCH_API_KEY")),
        }

    @app.patch("/settings")
    async def update_settings(patch: dict):
        if active or synthesis_jobs or preparations or setup.state['running']:
            raise ValueError("Stop the current response or voice preparation before changing settings.")
        if "key" in patch or "api_key" in patch:
            raise ValueError("Use the secure credential endpoint.")
        result = settings.update(patch)
        if speech.status('ar')['ready']: remember_installation(settings)
        return result

    class Credential(BaseModel):
        model_config = ConfigDict(extra="forbid")
        provider: str = Field(max_length=64)
        key: str = Field(max_length=4096)

    @app.post("/credentials")
    async def credential(body: Credential):
        mcp_keys = {"mcp:" + s["id"] for s in settings.snapshot()["assistant"]["mcp_servers"]}
        if body.provider not in set(settings.snapshot()["providers"]) | {"azure", "brave"} | mcp_keys:
            raise ValueError("Unknown provider.")
        secrets.save(body.provider, body.key)
        return {"saved": bool(body.key.strip())}

    class Mode(BaseModel):
        mode: str

    def for_mode(mode):
        if mode not in ("online", "offline"):
            raise ValueError("Invalid mode.")
        config = settings.snapshot()
        config["mode"] = mode
        return provider_factory(config, secrets)

    @app.post("/providers/test")
    async def test_provider(body: Mode):
        provider = for_mode(body.mode)
        async with gpu.assistant({"mode": body.mode}):
            return await provider.test()

    @app.post("/providers/models")
    async def list_models(body: Mode):
        return {"models": await for_mode(body.mode).models()}

    @app.get("/chats")
    async def chats():
        return db.chats()

    @app.post("/chats")
    async def new_chat():
        return {"id": db.new_chat()}

    @app.get("/chats/{chat_id}")
    async def messages(chat_id: str):
        return db.messages(chat_id)

    @app.delete("/chats/{chat_id}")
    async def delete_chat(chat_id: str):
        if chat_id in active:
            raise ValueError("Stop this response before deleting the conversation.")
        with db.connect() as connection:
            connection.execute("DELETE FROM chats WHERE id=?", (chat_id,))
        return {"deleted": True}

    class Question(BaseModel):
        model_config = ConfigDict(extra="forbid")
        chat_id: str = Field(pattern="^[a-f0-9]{32}$")
        text: str = Field(min_length=1, max_length=20000)
        voice: bool = False

    @app.post("/chat")
    async def chat(body: Question):
        with db.connect() as connection:
            if not connection.execute("SELECT id FROM chats WHERE id=?", (body.chat_id,)).fetchone():
                raise HTTPException(404, "Conversation not found.")
        if active:
            raise HTTPException(409, "A response is already running. Stop it before sending another message.")
        cancel = asyncio.Event()
        active[body.chat_id] = cancel

        async def events():
            try:
                replies = assistant.reply(body.chat_id, body.text.strip(), cancel, voice=body.voice)
                if body.voice:
                    replies = prepared_voice_reply(replies, speech, gpu, settings.snapshot(), cancel)
                async with aclosing(replies):
                    async for event in replies:
                        yield json.dumps(event, ensure_ascii=False) + "\n"
            except Exception:
                logging.getLogger(__name__).exception("Assistant response failed")
                yield (
                    json.dumps(
                        {
                            "type": "error",
                            "kind": "internal",
                            "message": "The response could not finish. Check Diagnostics and retry.",
                        }
                    )
                    + "\n"
                )
            finally:
                active.pop(body.chat_id, None)

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.post("/chat/cancel")
    async def cancel_chat():
        for event in active.values():
            event.set()
        return {"cancelled": True}

    @app.get("/documents")
    async def documents():
        return {"documents": db.documents(), "extensions": sorted(SUPPORTED)}

    class KnowledgeInstructions(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        instructions: str = Field(max_length=2000)

    class DocumentInstructions(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        instructions: str = Field(max_length=1000)
        enabled: bool
        instructions_enabled: bool = True

    @app.get("/knowledge")
    async def knowledge():
        return db.knowledge()

    @app.patch("/knowledge")
    async def save_knowledge(body: KnowledgeInstructions):
        return db.save_knowledge(body.instructions)

    @app.get("/documents/{doc_id}")
    async def preview_document(doc_id: str, offset: int = 0):
        if offset < 0 or offset > 1000000:
            raise ValueError("Invalid preview page.")
        return library.preview(doc_id, offset)

    @app.patch("/documents/{doc_id}")
    async def document_instructions(doc_id: str, body: DocumentInstructions):
        if active:
            raise ValueError("Wait for the current answer before changing its sources.")
        library.update(doc_id, body.instructions, body.enabled, body.instructions_enabled)
        return {"saved": True}

    @app.post("/knowledge/export")
    async def knowledge_export():
        if len(exports) >= 2:
            raise ValueError("Finish the current export before creating another.")
        export_id = random_secrets.token_hex(16)
        path = settings.data / ("knowledge-export-" + export_id + ".ems-library")
        try:
            async with library.queue_lock:
                await asyncio.to_thread(export_library, library, path)
            exports[export_id] = path
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {"id": export_id}

    def clean_export(export_id):
        path = exports.pop(export_id, None)
        if path:
            path.unlink(missing_ok=True)

    @app.get("/knowledge/exports/{export_id}")
    async def download_knowledge(export_id: str):
        path = exports.get(export_id)
        if not path or not path.is_file():
            raise HTTPException(404, "Export expired. Export the library again.")
        return FileResponse(path, media_type="application/zip", filename="EMS-knowledge.ems-library",
            background=BackgroundTask(clean_export, export_id))

    @app.delete("/knowledge/exports/{export_id}")
    async def discard_export(export_id: str):
        clean_export(export_id)
        return {"discarded": True}

    @app.post("/knowledge/import")
    async def knowledge_import(file: UploadFile = File(...)):
        if active:
            raise ValueError("Wait for the current answer before importing knowledge.")
        path = settings.data / ("knowledge-upload-" + random_secrets.token_hex(16) + ".zip")
        try:
            size = 0
            with path.open("wb") as out:
                while chunk := await file.read(1024**2):
                    size += len(chunk)
                    if size > MAX_BYTES + 2 * 1024**2:
                        raise ValueError("This knowledge bundle exceeds the 258 MB upload limit.")
                    out.write(chunk)
            async with library.queue_lock:
                result = await asyncio.to_thread(import_library, library, path)
            for doc_id in result["created"]:
                schedule_ingest(doc_id)
            return result
        finally:
            path.unlink(missing_ok=True)
            await file.close()

    @app.post("/documents")
    async def upload(file: UploadFile = File(...)):
        name = Path(file.filename.replace("\\", "/")).name if file.filename else "document"
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED:
            raise ValueError("This file type is not supported.")
        limit = settings.snapshot()["documents"]["max_file_mb"] * 1024**2
        digest, size = hashlib.sha256(), 0
        path = settings.data / ("upload-" + random_secrets.token_hex(16) + suffix)
        try:
            with path.open("wb") as out:
                while chunk := await file.read(1024**2):
                    size += len(chunk)
                    if size > limit:
                        raise ValueError("File exceeds the configured upload limit.")
                    digest.update(chunk)
                    out.write(chunk)
            doc_id, created = library.register(path, name, digest.hexdigest())
            if created:
                schedule_ingest(doc_id)
            return {"id": doc_id, "created": created}
        finally:
            path.unlink(missing_ok=True)
            await file.close()

    @app.post("/documents/{doc_id}/retry")
    async def retry(doc_id: str):
        if doc_id in busy:
            raise ValueError("This document is already queued.")
        library.path(doc_id)
        busy.add(doc_id)

        async def run():
            try:
                current = next((d for d in db.documents() if d["id"] == doc_id), None)
                if (
                    current
                    and current["chunks"]
                    and current["status"] in ("keyword_only", "ready", "interrupted")
                ):
                    async with library.queue_lock:
                        await library.index(doc_id)
                else:
                    await library.ingest(doc_id)
            finally:
                busy.discard(doc_id)

        schedule(run())
        return {"queued": True}

    @app.delete("/documents/{doc_id}")
    async def remove(doc_id: str):
        if active:
            raise ValueError("Wait for the response before removing its possible sources.")
        await library.delete(doc_id)
        return {"deleted": True}

    @app.post("/speech/transcribe")
    async def transcribe(file: UploadFile = File(...), language: str = "ar"):
        if setup.state['running']: raise ValueError('Voice setup is running. Please wait for it to finish.')
        if language not in ("ar", "en", "auto"):
            raise ValueError("Invalid recognition language.")
        try:
            data = await file.read(16 * 1024**2 + 1)
            if len(data) > 16 * 1024**2:
                raise ValueError("Audio upload is too large.")
            return await speech.transcribe(data, language)
        finally:
            await file.close()

    @app.get("/speech/status")
    async def speech_status(language: str = "ar", conversation: bool = False):
        if language not in ("ar", "en"):
            raise ValueError("Invalid speech language.")
        return speech.status(language, conversation=conversation)

    class Speak(BaseModel):
        text: str = Field(min_length=1, max_length=6000)
        language: str = Field(pattern="^(ar|en)$")
        generation: int = Field(default=0, ge=0)
        local_only: bool = False
        voice_conversation: bool = False

    @app.post("/speech/synthesize")
    async def synthesize(body: Speak):
        if setup.state['running']: raise ValueError('Voice setup is running. Please wait for it to finish.')
        if body.generation < speech_generation: raise HTTPException(409, "Speech request was cancelled.")
        task = asyncio.create_task(speech.synthesize(body.text, body.language,
            local_only=body.local_only, voice_conversation=body.voice_conversation))
        synthesis_jobs[task] = body.generation
        try:
            audio, warnings = await task
            return {"audio": base64.b64encode(audio).decode("ascii"), "warnings": warnings}
        except asyncio.CancelledError:
            raise HTTPException(409, "Speech request was cancelled.") from None
        finally: synthesis_jobs.pop(task, None)

    class CancelSpeech(BaseModel):
        generation: int = Field(ge=0)

    @app.post("/speech/cancel")
    async def cancel_speech(body: CancelSpeech):
        nonlocal speech_generation
        speech_generation = max(speech_generation, body.generation)
        tasks = [t for t, g in synthesis_jobs.items() if g < speech_generation]
        for task in tasks:
            if not task.cancelling(): task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return {"cancelled": True}

    @app.post("/assistant/sleep")
    async def sleep_assistant():
        if active or synthesis_jobs: raise ValueError("Stop the current response before releasing model memory.")
        if preparing.locked(): raise ValueError("Wait for voice preparation to finish before releasing model memory.")
        if research.tasks and any(not t.done() for t in research.tasks.values()): raise ValueError("Pause research before releasing model memory.")
        await speech.close()
        await gpu.release()
        return {"released": True}

    class VoiceSession(BaseModel):
        model_config = ConfigDict(extra="forbid")
        id: str = Field(pattern=r"^[a-f0-9-]{36}$")

    @app.post("/voice/cancel")
    async def cancel_voice_preparation(body: VoiceSession):
        cancelled_preparations.append(body.id)
        task = preparations.get(body.id)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return {"cancelled": True}

    @app.post("/voice/prepare")
    async def prepare_voice_conversation(body: VoiceSession):
        if setup.state['running']: raise ValueError('Voice setup is still running. Please wait for it to finish.')
        if body.id in cancelled_preparations:
            raise HTTPException(409, "Voice preparation was cancelled.")
        if preparations: raise ValueError("Voice preparation is already running.")
        cfg = voice_config(settings.snapshot())
        cloud_speech = cfg["speech"]["tts"]["provider"] == "azure"
        if cfg["speech"]["tts"]["provider"] == "none":
            raise ValueError("Spoken replies are off. Choose a voice in Settings → Advanced settings → Voice.")
        if cloud_speech and cfg["mode"] != "online":
            raise ValueError("Azure Speech needs Online voice. Choose Chatterbox to keep this conversation offline.")
        if cloud_speech and not speech.status(cfg["language"], conversation=True)["synthesis"]:
            raise ValueError("Add your Azure Speech key in Advanced settings before starting Online voice.")
        if active: raise ValueError("Stop the current response first.")
        async def prepare_model():
            async with gpu.assistant(cfg):
                await assistant.prepare_voice(cfg)
        async def prepare():
            async with preparing:
                # Apply the 20B voice memory allocation before loading speech weights.
                # A preceding typed chat may have used more GPU layers.
                if resident_voice(cfg) or cfg["mode"] == "online":
                    await prepare_model()
                await speech.local("warmup_asr" if cloud_speech else "warmup")
                if not resident_voice(cfg) and cfg["mode"] == "offline":
                    await prepare_model()
        task = asyncio.create_task(prepare())
        preparations[body.id] = task
        try:
            await task
        except asyncio.CancelledError:
            raise HTTPException(409, "Voice preparation was cancelled.") from None
        finally:
            preparations.pop(body.id, None)
        return {"ready": True, "model": cfg["models"][cfg["mode"]]["id"], "local": cfg["mode"] == "offline"}

    @app.get("/diagnostics/activity")
    async def service_activity():
        return {"events": list(activity.rows)}

    @app.get("/diagnostics")
    async def diagnostics():
        return {
            "hardware": await asyncio.to_thread(hardware, settings.data),
            "build": build_info(),
            "data_dir": str(settings.data),
            "database_schema": 1,
            "active_response": bool(active),
            "processing_jobs": len(jobs),
        }

    class InstallModel(BaseModel):
        model: str = Field(max_length=160)
        accepted_download: bool = False

    @app.get("/local/models/status")
    async def model_status():
        return model_manager.state

    @app.post("/local/models/install")
    async def install_model(body: InstallModel):
        if setup.state['running']: raise ValueError('Wait for the current setup to finish.')
        return await model_manager.start(body.model, body.accepted_download)

    @app.post("/local/models/cancel")
    async def cancel_model():
        await model_manager.close()
        return model_manager.state

    return app
