"""Explicit in-app provisioning; reuse installed components before downloading anything."""
import asyncio
import hashlib
import json
import logging
import shutil
import platform
import os
from pathlib import Path
import subprocess
import sys

import httpx
import psutil

from app.core.platforms import venv_python, ollama_name
from app.core.settings import ROOT, replace_settings_file
from .diagnostics import hardware
from .model_manager import STARTER_MODELS

ASR_FILES = ('model.bin', 'config.json', 'tokenizer.json')
VOICE_FILES = ('t3_mtl23ls_v3.safetensors', 's3gen.pt', 've.pt', 'conds.pt', 'grapheme_mtl_merged_expanded_v1.json')


def complete(folder, names):
    return bool(folder) and all((Path(folder) / name).is_file() and (Path(folder) / name).stat().st_size for name in names)


def recover_installed(settings):
    """Bounded discovery in the app's own install folder/registry, never a disk-wide scan."""
    cfg = settings.snapshot()
    root = settings.data / 'runtime'
    registry = root / 'installation.json'
    saved = {}
    if registry.is_file():
        try:
            saved = json.loads(registry.read_text(encoding='utf8'))
            if not isinstance(saved, dict): saved = {}
            for key in ('speech', 'local_runtime'):
                if not isinstance(saved.get(key, {}), dict): saved[key] = {}
        except (ValueError, OSError): pass
    runtime = cfg['speech']['runtime']
    defaults = {'python': str(venv_python(root / 'speech-env')),
                'tts_python': str(venv_python(root / 'voice-env')),
                'whisper_path': str(root / 'whisper'), 'chatterbox_path': str(root / 'chatterbox-v3')}
    found = {}
    for key, normal in defaults.items():
        def valid(value):
            if not isinstance(value, str): return False
            if key == 'whisper_path': return complete(value, ASR_FILES)
            if key == 'chatterbox_path': return complete(value, VOICE_FILES)
            return bool(value) and Path(value).is_file()
        if valid(runtime.get(key)): continue
        for candidate in [saved.get('speech', {}).get(key, ''), normal]:
            if valid(candidate): found[key] = candidate; break
    patch = {'speech': {'runtime': found}} if found else {}
    if not Path(cfg.get('local_runtime', {}).get('executable', '')).is_file():
        for local in [saved.get('local_runtime', {}), {'executable': str(root / 'ollama' / ollama_name()), 'models_dir': str(root / 'ollama-models')}]:
            if isinstance(local.get('executable'), str) and isinstance(local.get('models_dir'), str) and local['models_dir'] and Path(local['executable']).is_file():
                patch.update(local_runtime=local)
                patch['providers'] = {'ollama_local': {'endpoint': 'http://127.0.0.1:11435'}}
                patch['documents'] = {'embedding': {'endpoint': 'http://127.0.0.1:11435'}}
                break
    if patch: settings.update(patch)
    return bool(patch)


def remember_installation(settings):
    cfg = settings.snapshot(); root = settings.data / 'runtime'; root.mkdir(exist_ok=True)
    target = root / 'installation.json'
    # Paths only: never store keys, conversations or document content in this registry.
    value = {'speech': cfg['speech']['runtime'], 'local_runtime': cfg.get('local_runtime', {})}
    temp = target.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2), encoding='utf8'); replace_settings_file(temp, target)


class Setup:
    def __init__(self, settings, speech, runtime, models):
        self.settings, self.speech, self.runtime, self.models = settings, speech, runtime, models
        self.root = settings.data / 'runtime'
        self.state = {'running': False, 'stage': 'idle', 'component': '', 'error': False}
        self.task = None
        self.process = None

    async def status(self):
        cfg = self.settings.snapshot(); speech = self.speech.status('ar')
        endpoint = cfg['providers']['ollama_local']['endpoint'].rstrip('/')
        installed, available = [], False
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                response = await client.get(endpoint + '/api/tags'); response.raise_for_status()
                installed = [x['name'] for x in response.json()['models']]; available = True
        except (httpx.HTTPError, ValueError, KeyError): pass
        return {'job': dict(self.state), 'speech': speech, 'installed_models': installed, 'engine_ready': available,
                'selected_model': cfg['assistant']['voice_model'] or cfg['models']['offline']['id'],
                'starter_models': list(STARTER_MODELS), 'download': dict(self.models.state)}

    async def repair(self):
        if self.state['running']: raise ValueError('Setup is already running.')
        # Reload disk explicitly so a setup completed outside the app is not lost to a stale draft.
        self.settings.reload()
        recover_installed(self.settings)
        await self.speech.close()
        await self.runtime.start()
        remember_installation(self.settings)
        return await self.status()

    async def start(self, component, accepted):
        if not accepted: raise ValueError('Confirm the component download before starting setup.')
        if component not in {'voice', 'assistant'}: raise ValueError('Unknown setup component.')
        if self.state['running'] or self.models.state['running']: raise ValueError('Another setup is running.')
        self.state = {'running': True, 'stage': 'checking', 'component': component, 'error': False}
        self.task = asyncio.create_task(self._install(component))
        return dict(self.state)

    async def _run(self, args, stage, timeout=2400):
        self.state['stage'] = stage
        env = dict(os.environ, PYTHONUTF8='1', PIP_NO_INPUT='1', PIP_DISABLE_PIP_VERSION_CHECK='1', HF_HUB_DISABLE_TELEMETRY='1')
        for key in ('PYTHONHOME', 'PYTHONPATH', 'ELECTRON_RUN_AS_NODE', 'OPENAI_API_KEY', 'OLLAMA_API_KEY'): env.pop(key, None)
        logs = self.settings.data / 'logs'; logs.mkdir(exist_ok=True)
        try:
            with (logs / 'setup.log').open('ab') as log:
                self.process = await asyncio.create_subprocess_exec(*map(str, args), env=env, stdin=asyncio.subprocess.DEVNULL,
                    stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                code = await asyncio.wait_for(self.process.wait(), timeout)
            if code not in (0, 3010): raise RuntimeError('Setup step failed: ' + stage)
        except (asyncio.CancelledError, TimeoutError):
            await self._stop_process()
            raise
        finally: self.process = None

    async def _stop_process(self):
        if self.process and self.process.returncode is None:
            try:
                for child in psutil.Process(self.process.pid).children(recursive=True):
                    try: child.terminate()
                    except psutil.Error: pass
                self.process.terminate(); await self.process.wait()
            except (ProcessLookupError, psutil.Error): pass

    async def _python(self):
        cfg = self.settings.snapshot(); candidates = [str(self.root / ('python/python.exe' if os.name == 'nt' else 'python/bin/python3.12')), cfg['speech']['runtime']['python'], cfg['speech']['runtime']['tts_python']]
        if not getattr(sys, 'frozen', False): candidates.append(sys.executable)
        if sys.platform == 'darwin':
            candidates += [shutil.which('python3.12') or '', '/opt/homebrew/bin/python3.12', '/usr/local/bin/python3.12', '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12']
        if os.name == 'nt':
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(hive, r'Software\Python\PythonCore\3.12\InstallPath') as key:
                        candidates.append(str(Path(winreg.QueryValueEx(key, '')[0]) / 'python.exe'))
                except OSError: pass
        for candidate in dict.fromkeys(candidates):
            if not candidate or not Path(candidate).is_file(): continue
            try:
                result = await asyncio.to_thread(subprocess.run, [candidate, '-I', '-c', 'import sys,struct,json,platform;print(json.dumps([list(sys.version_info[:2]),struct.calcsize("P"),sys._base_executable,platform.machine()]))'], capture_output=True, timeout=15, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                version, width, executable, machine = json.loads(result.stdout)
                if sys.platform == "darwin" and machine != platform.machine(): continue
                if version == [3, 12] and width == 8 and Path(executable).is_file(): return Path(executable)
            except (OSError, ValueError, subprocess.TimeoutExpired): pass
        if os.name != 'nt': raise ValueError('Install native Python 3.12 from python.org or Homebrew, then retry setup. See the macOS setup guide.')
        spec = json.loads((ROOT / 'config/runtime-lock.json').read_text())['python_installer']
        installer = self.root / 'downloads/python-3.12.10-amd64.exe'; installer.parent.mkdir(parents=True, exist_ok=True)
        self.state['stage'] = 'python'
        def verified():
            if not installer.is_file(): return False
            with installer.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest() == spec['sha256']
        if not verified():
            part = installer.with_suffix('.part')
            async with httpx.AsyncClient(timeout=90, follow_redirects=False, trust_env=False) as client:
                async with client.stream('GET', spec['url']) as response:
                    response.raise_for_status()
                    with part.open('wb') as handle:
                        async for chunk in response.aiter_bytes(1024 * 1024): handle.write(chunk)
            part.replace(installer)
            if not verified(): raise ValueError('Runtime download verification failed.')
        await self._run([installer, '/quiet', 'InstallAllUsers=0', 'TargetDir=' + str(self.root / 'python'), 'Include_launcher=0', 'Include_pip=1', 'Include_test=0', 'Include_doc=0', 'Include_tcltk=0', 'PrependPath=0', 'AssociateFiles=0', 'Shortcuts=0'], 'python', 600)
        executable = self.root / 'python/python.exe'
        if not executable.is_file(): raise ValueError('The runtime installer did not create the expected files.')
        return executable

    async def _download(self, python, component, destination):
        await self._run([python, '-m', 'pip', 'install', 'httpx==0.28.1'], 'dependencies')
        await self._run([python, ROOT / 'scripts/download_runtime.py', component, '--destination', destination], component, 7200)

    async def _install(self, component):
        try:
            recover_installed(self.settings)
            self.root.mkdir(exist_ok=True)
            specs = await asyncio.to_thread(hardware, self.settings.data)
            if component == 'voice' and self.speech.status('ar')['ready']:
                remember_installation(self.settings); self.state['stage'] = 'ready'; return
            if component == 'voice' and specs['free_disk_gb'] < 25: raise ValueError('disk_space')
            if component == 'assistant':
                if sys.platform == 'darwin':
                    cfg = self.settings.snapshot()
                    if not Path(cfg.get('local_runtime', {}).get('executable', '')).is_file():
                        executable = next((p for p in [shutil.which('ollama'), '/Applications/Ollama.app/Contents/Resources/ollama', '/opt/homebrew/bin/ollama', '/usr/local/bin/ollama'] if p and Path(p).is_file()), None)
                        if executable: self.settings.update({'local_runtime': {'executable': executable, 'models_dir': str(self.root / 'ollama-models')}, 'providers': {'ollama_local': {'endpoint': 'http://127.0.0.1:11435'}}, 'documents': {'embedding': {'endpoint': 'http://127.0.0.1:11435'}}})
                existing = await self.status()
                cfg = self.settings.snapshot()
                if not existing['engine_ready'] and not Path(cfg.get('local_runtime', {}).get('executable', '')).is_file():
                    if specs['free_disk_gb'] < 25:
                        raise ValueError('disk_space')
                    python = await self._python()
                    bootstrap = venv_python(self.root / 'setup-env')
                    if not bootstrap.is_file():
                        await self._run([python, '-m', 'venv', self.root / 'setup-env'], 'runtime')
                    await self._download(bootstrap, 'ollama', self.root)
                    recover_installed(self.settings)
                await self.runtime.start()
                current = await self.status()
                if not current['engine_ready']:
                    raise ValueError('Ollama did not start. Run Check & repair, then retry.')
                cfg = self.settings.snapshot()
                self.state['stage'] = 'assistant'
                installed = set(current['installed_models'])
                for model in dict.fromkeys([*STARTER_MODELS, cfg['documents']['embedding']['model']]):
                    if model in installed:
                        continue
                    await self.models.start(model, True)
                    await self.models.task
                    if self.models.state['status'] != 'installed':
                        raise ValueError(f"{model}: {self.models.state['status']}")
            else:
                python = await self._python()
                if sys.platform == 'darwin' and platform.machine() not in ('arm64', 'aarch64'):
                    raise ValueError('Local voice setup currently requires an Apple Silicon Mac. Intel Macs support text and document features; see the macOS guide.')
                await self.speech.close()
                cuda = sys.platform != 'darwin' and any(g['vram_mb'] >= 6000 for g in specs['gpus'])
                torch_index = 'https://download.pytorch.org/whl/cu124' if cuda else 'https://download.pytorch.org/whl/cpu'
                cfg = self.settings.snapshot(); existing = self.speech.status('ar'); patch = {}
                for kind, env_name, requirement, folder, model in [('recognition', 'speech-env', 'requirements-recognition.in', 'whisper', 'whisper'), ('synthesis', 'voice-env', 'requirements-chatterbox.in', 'chatterbox-v3', 'chatterbox')]:
                    if existing[kind]: continue
                    target = self.root / env_name; exe = venv_python(target)
                    if not exe.is_file(): await self._run([python, '-m', 'venv', target], 'runtime')
                    torch_args = [exe, '-m', 'pip', 'install', 'torch==2.6.0', 'torchaudio==2.6.0']
                    if sys.platform != 'darwin': torch_args += ['--index-url', torch_index]
                    await self._run(torch_args, 'dependencies')
                    await self._run([exe, '-m', 'pip', 'install', '-r', ROOT / requirement], 'recognition' if kind == 'recognition' else 'arabic_voice')
                    if kind == 'synthesis':
                        revision = next(line.split(' @ ', 1)[1] for line in (ROOT / requirement).read_text().splitlines() if line.startswith('chatterbox-tts @ '))
                        await self._run([exe, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', revision], 'arabic_voice')
                    await self._download(exe, model, self.root / folder)
                    patch.update({'python': str(exe), 'whisper_path': str(self.root / folder)} if kind == 'recognition' else {'tts_python': str(exe), 'chatterbox_path': str(self.root / folder)})
                device = 'cuda' if cuda else 'cpu'
                self.settings.update({'speech': {'runtime': patch, 'asr': {'device': device, 'compute_type': 'float16' if cuda else 'int8'}, 'tts': {'provider': 'chatterbox', 'voice': 'egyptian', 'chatterbox_version': 'v3', 'device': device}}})
                self.state['stage'] = 'verifying'
                await self.speech.local('warmup')
                if not self.speech.status('ar')['ready']: raise ValueError('verification')
            remember_installation(self.settings)
            self.state['stage'] = 'ready'
        except asyncio.CancelledError:
            self.state['stage'] = 'cancelled'
        except Exception as exc:
            logging.getLogger(__name__).exception('Managed setup failed at %s', self.state['stage'])
            self.state.update(stage='failed', error=True, reason='disk_space' if str(exc) == 'disk_space' else 'setup_failed', message=str(exc)[:300] if isinstance(exc, ValueError) else '')
        finally: self.state['running'] = False

    async def close(self):
        if self.task and not self.task.done():
            self.task.cancel(); await self.task
        if self.state['component'] == 'assistant' and self.models.state['running']:
            await self.models.close()
