"""Local speech worker is isolated from packaged backend dependencies and cloud clients."""

import asyncio
from collections import OrderedDict
import hashlib
from contextlib import asynccontextmanager, nullcontext
import base64
import html
import json
import os
import re
import uuid
from pathlib import Path
import httpx
from app.core.settings import ROOT
from app.core.options import resident_voice
from app.providers.client import http_error


class Speech:
    def __init__(self, settings, secrets, worker_kind="speech", gpu=None):
        self.settings, self.secrets = settings, secrets
        self.worker_kind = worker_kind
        self.gpu = gpu
        self.voice_worker = None
        self.process = None
        self.signature = ""
        self.lock = asyncio.Lock()
        self.audio_cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_locks = {}
        self.decoder_graph_disabled = False
        self.cloud_client = None

    def status(self, language="ar", conversation=False):
        cfg = self.settings.snapshot()
        if conversation:
            cfg["mode"] = cfg["assistant"].get("voice_mode", "offline")
        speech = cfg["speech"]
        runtime, voice = speech["runtime"], speech["tts"]
        recognition = (
            bool(runtime['python']) and Path(runtime["python"]).is_file()
            and bool(runtime['whisper_path'])
            and all((Path(runtime['whisper_path']) / name).is_file() for name in ('model.bin', 'config.json', 'tokenizer.json'))
        )
        if voice["provider"] == "none":
            synthesis = False
        elif voice["provider"] == "azure":
            synthesis = cfg["mode"] == "online" and bool(self.secrets.get("azure", "AZURE_SPEECH_KEY"))
        elif language == "en":
            synthesis = Path(runtime["python"]).is_file()
        elif voice["provider"] == "chatterbox":
            folder = Path(runtime["chatterbox_path"])
            required = [
                f"t3_mtl23ls_{voice['chatterbox_version']}.safetensors",
                "s3gen.pt",
                "ve.pt",
                "conds.pt",
                "grapheme_mtl_merged_expanded_v1.json",
            ]
            synthesis = Path(runtime["tts_python"]).is_file() and all(
                (folder / name).is_file() for name in required
            )
            if voice.get("voice") == "egyptian":
                synthesis = synthesis and (ROOT / "assets/voices/egyptian-reference.wav").is_file()
        else:
            synthesis = (
                Path(runtime["python"]).is_file()
                and (Path(runtime["kemetone_path"]) / "kemetone.pth").is_file()
            )
        return {
            "ready": bool(recognition and synthesis),
            "recognition": bool(recognition),
            "synthesis": bool(synthesis),
            "provider": voice["provider"],
        }

    async def local(self, action, **payload):
        settings = self.settings.snapshot()
        cfg = settings["speech"]
        cfg["keep_resident"] = resident_voice(settings)
        if self.worker_kind == "speech" and cfg["tts"]["provider"] == "chatterbox":
            if action == "warmup":
                await self.local("warmup_asr")
            if action == "warmup" or (action == "synthesize" and payload.get("language") == "ar"):
                if self.voice_worker is None:
                    self.voice_worker = Speech(self.settings, self.secrets, "chatterbox", self.gpu)
                return await self.voice_worker.local(action, **payload)
        stage = "asr" if action in ("transcribe", "warmup_asr") else "tts"
        uses_cuda = cfg[stage]["device"] == "cuda" and not (action == "synthesize" and payload.get("language") == "en")
        async with self.gpu.speech(uses_cuda) if self.gpu else nullcontext():
            return await self._request(cfg, action, **payload)

    async def _request(self, cfg, action, **payload):
        for attempt in range(2):
            try:
                result = await self._exchange(cfg, action, **payload)
            except (BrokenPipeError, ConnectionResetError, EOFError) as error:
                if attempt == 0:
                    continue
                raise ValueError("The local speech worker closed twice. Check the speech diagnostic log.") from error
            if not result.get('retry_without_decoder_graph'):
                return result
        raise ValueError('Local speech could not recover. Restart EMS and check the speech diagnostic log.')

    async def _exchange(self, cfg, action, **payload):
        python = cfg["runtime"]["tts_python" if self.worker_kind == "chatterbox" else "python"]
        if not python or not Path(python).is_file():
            raise ValueError(
                "The Arabic voice is not installed yet. Open Settings → Voice conversation → Set up. Existing downloads will be reused."
                if self.worker_kind == "chatterbox"
                else "Speech recognition is not installed yet. Open Settings → Voice conversation → Set up. Existing downloads will be reused."
            )
        async with self.lock:
            if self.worker_kind == 'chatterbox' and self.decoder_graph_disabled:
                cfg = {**cfg, '_decoder_graph': False, '_cpu_single_batch': False}
            signature = json.dumps(cfg, sort_keys=True)
            if self.process and (self.signature != signature or self.process.returncode is not None):
                if self.process.returncode is None:
                    self.process.terminate()
                await self.process.wait()
                self.process = None
            if self.process is None:
                env = dict(
                    os.environ,
                    PYTHONUTF8="1",
                    PYTHONDONTWRITEBYTECODE="1",
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    HF_HUB_DISABLE_TELEMETRY="1",
                    DO_NOT_TRACK="1",
                )
                logs = self.settings.data / "logs"
                logs.mkdir(exist_ok=True)
                with (logs / (self.worker_kind + ".log")).open("ab") as log:
                    self.process = await asyncio.create_subprocess_exec(
                        python,
                        "-u",
                        str(
                            ROOT
                            / (
                                "scripts/chatterbox_worker.py"
                                if self.worker_kind == "chatterbox"
                                else "scripts/speech_worker.py"
                            )
                        ),
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=log,
                        env=env,
                        limit=64 * 1024**2,
                        creationflags=0x08000000 if os.name == "nt" else 0,
                    )
                self.signature = signature
            request_id = uuid.uuid4().hex
            cancel_file = self.settings.data / ("speech-cancel-" + request_id + ".flag")
            request = (
                (
                    json.dumps(
                        {
                            "action": action,
                            "request_id": request_id,
                            "cancel_file": str(cancel_file),
                            "config": cfg,
                            "mode": self.settings.snapshot()["mode"],
                            **payload,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode("utf8")
            )
            reader = asyncio.create_task(self.process.stdout.readline())
            try:
                self.process.stdin.write(request)
                await self.process.stdin.drain()
                line = await asyncio.wait_for(asyncio.shield(reader), timeout=240)
                if not line:
                    raise EOFError("The speech worker closed its output.")
                result = json.loads(line)
            except (BrokenPipeError, ConnectionResetError, EOFError):
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
                if self.process.returncode is None:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass
                await self.process.wait()
                self.process = None
                raise
            except asyncio.CancelledError:
                # Keep the expensive Arabic model resident after a spoken interruption.
                # Drain this request's reply before another request can use the pipe.
                retained = False
                if self.worker_kind == "chatterbox" and action == "synthesize":
                    try:
                        cancel_file.touch()
                        line = await asyncio.wait_for(asyncio.shield(reader), 2)
                        reply = json.loads(line)
                        retained = not reply.get('retry_without_decoder_graph')
                        if not retained:
                            self.decoder_graph_disabled = True
                    except (OSError, TimeoutError, ValueError):
                        pass
                if not retained:
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                    if self.process.returncode is None: self.process.kill()
                    await self.process.communicate()
                    self.process = None
                raise
            except (TimeoutError, ValueError):
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
                self.process.kill()
                await self.process.communicate()
                self.process = None
                raise ValueError(
                    "Local speech did not finish. Check Voice Settings and the speech diagnostic log."
                )
            finally:
                cancel_file.unlink(missing_ok=True)
            if result.get('retry_without_decoder_graph'):
                if self.worker_kind != 'chatterbox' or action not in ('warmup', 'synthesize'):
                    raise ValueError('Unexpected local speech worker response.')
                self.decoder_graph_disabled = True
                if self.process.returncode is None:
                    self.process.kill()
                await self.process.communicate()
                self.process = None
            if result.get("error"):
                raise ValueError(result["error"])
            return result

    async def transcribe(self, recording, language):
        if isinstance(recording, bytes):
            return await self.local("transcribe", audio=base64.b64encode(recording).decode("ascii"), language=language)
        return await self.local("transcribe", path=str(recording), language=language)

    @asynccontextmanager
    async def cached_clip(self, key):
        # Two identical preview/play requests must not synthesize the same clip
        # twice. Canceling a waiter does not cancel the request already running.
        entry = self.cache_locks.setdefault(key, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if not entry[1]:
                self.cache_locks.pop(key, None)

    async def synthesize(self, text, language, local_only=False, voice_conversation=False):
        cfg = self.settings.snapshot()
        if voice_conversation:
            cfg["mode"] = cfg["assistant"].get("voice_mode", "offline")
        voice = cfg["speech"]["tts"]
        if local_only and voice["provider"] == "azure": raise ValueError("Offline voice needs a local voice. Choose Chatterbox in Settings → Advanced settings → Voice.")
        text = re.sub(r"\[D\d+\]|[`#*_]", "", text).strip()
        if not text or len(text) > 6000:
            raise ValueError("Speech text must contain 1–6000 characters.")
        if voice["provider"] == "none":
            raise ValueError("Spoken replies are disabled in Voice Settings.")
        if voice["provider"] != "azure":
            spoken_key = text
            if voice["provider"] == "chatterbox" and language == "ar":
                # The existing Chatterbox worker removes these marks before
                # tokenization. Share exactly that spoken form, never synonyms,
                # numbers, punctuation boundaries or the recognition transcript.
                spoken_key = re.sub(r"[\u064b-\u065f\u0670ـ]", "", spoken_key)
                spoken_key = re.sub(r"[ \t]+", " ", spoken_key)
            key = hashlib.sha256((json.dumps(cfg["speech"], sort_keys=True) + language + spoken_key).encode()).hexdigest()
            async with self.cached_clip(key):
                if key in self.audio_cache:
                    self.audio_cache.move_to_end(key)
                    return self.audio_cache[key]
                result = await self.local("synthesize", text=text, language=language)
                audio = base64.b64decode(result["audio"])
                value = (audio, result.get("warnings", []))
                if len(text) <= 240 and len(audio) <= 2_000_000:
                    if key in self.audio_cache: self.cache_bytes -= len(self.audio_cache.pop(key)[0])
                    self.audio_cache[key] = value; self.cache_bytes += len(audio)
                    while self.cache_bytes > 8_000_000 or len(self.audio_cache) > 24:
                        self.cache_bytes -= len(self.audio_cache.popitem(last=False)[1][0])
                return value
        if cfg["mode"] == "offline":
            raise ValueError(
                "Azure Speech is online-only. Select the local voice in Settings to speak while Offline."
            )
        key = self.secrets.get("azure", "AZURE_SPEECH_KEY")
        if not key:
            raise ValueError("Add your Azure Speech key in Voice Settings.")
        region = voice["azure_region"]
        if not re.fullmatch("[a-z0-9-]{2,40}", region):
            raise ValueError("Invalid Azure region.")
        name = voice["azure_voice_en" if language == "en" else "azure_voice_ar"]
        ssml = f'<speak version="1.0" xml:lang="{language}"><voice name="{html.escape(name, quote=True)}"><prosody rate="{int((voice["speed"] - 1) * 100):+d}%">{html.escape(text)}</prosody></voice></speak>'
        # Reuse TLS connections across chunks. Keys remain request headers,
        # so changing a credential cannot reuse an old Authorization value.
        if self.cloud_client is None or self.cloud_client.is_closed:
            self.cloud_client = httpx.AsyncClient(timeout=45, trust_env=False, follow_redirects=False)
        try:
            response = await self.cloud_client.post(
                f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
                content=ssml.encode("utf8"),
                headers={
                    "Ocp-Apim-Subscription-Key": key,
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
                    "User-Agent": "EMS-AI-Assistant",
                },
            )
        except httpx.TimeoutException as error:
            raise ValueError("The online voice took too long. Try again, or choose Chatterbox for local speech.") from error
        except httpx.HTTPError as error:
            raise ValueError("The online voice connection stopped. Check your connection and try again.") from error
        if response.status_code >= 300:
            raise http_error(response.status_code, "online")
        if len(response.content) <= 44 or response.content[:4] != b"RIFF" or response.content[8:12] != b"WAVE":
            raise ValueError("The online voice returned no playable audio. Check the selected Azure voice and region.")
        return response.content, []

    async def close(self):
        self.audio_cache.clear(); self.cache_bytes = 0
        if self.cloud_client is not None:
            await self.cloud_client.aclose()
            self.cloud_client = None
        if self.voice_worker:
            await self.voice_worker.close()
        if self.process and self.process.returncode is None:
            self.process.terminate()
            await self.process.wait()
        self.process = None

    async def offload(self):
        if self.voice_worker: await self.voice_worker.offload()
        if self.process and self.process.returncode is None:
            cfg = self.settings.snapshot()["speech"]
            cfg["keep_resident"] = resident_voice(self.settings.snapshot())
            await self._request(cfg, "offload")
