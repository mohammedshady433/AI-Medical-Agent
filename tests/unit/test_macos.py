"""Native platform boundary regressions without launching apps or accessing user secrets."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import subprocess
import sys

import pytest
from app.core import platforms
from app.core.secrets import SecretStore
from app.services.setup import Setup
from app.services.local_runtime import LocalRuntime


def test_native_paths(monkeypatch):
    monkeypatch.setattr(platforms, 'sys', SimpleNamespace(platform='darwin'))
    assert platforms.data_directory().parts[-3:] == ('Library', 'Application Support', 'EMS-AI-Assistant')
    if sys.platform == 'darwin': assert platforms.venv_python(Path('/tmp/voice')) == Path('/tmp/voice/bin/python')


def test_keychain_namespaces_and_no_plaintext(tmp_path, monkeypatch):
    monkeypatch.setattr('app.core.secrets.sys', SimpleNamespace(platform='darwin'))
    values = {}
    class Native:
        def set_password(self, service, name, value): values[service, name] = value
        def get_password(self, service, name): return values.get((service, name))
        def delete_password(self, service, name): values.pop((service, name))
    monkeypatch.setattr(SecretStore, '_keychain', staticmethod(Native))
    a, b = SecretStore(tmp_path / 'installed'), SecretStore(tmp_path / 'dev')
    a.save('ollama', 'dummy-provider-key')
    assert a.get('ollama') == 'dummy-provider-key' and b.get('ollama') == ''
    assert not a.path.exists()
    a.save('ollama', '')
    monkeypatch.setenv('EMS_TEST_KEY', 'dummy-env-key')
    assert a.get('ollama', 'EMS_TEST_KEY') == 'dummy-env-key'
    with pytest.raises(ValueError): a.save('../bad', 'value')


async def test_mac_voice_uses_pypi_cpu_and_native_venvs(settings, monkeypatch):
    from app.services.setup import ASR_FILES, VOICE_FILES, complete
    from app.core.platforms import venv_python
    class Speech:
        def status(self, lang):
            cfg=settings.snapshot()['speech']['runtime']
            ready=complete(cfg['whisper_path'], ASR_FILES) and complete(cfg['chatterbox_path'], VOICE_FILES)
            return dict(ready=ready,recognition=ready,synthesis=ready)
        async def close(self): pass
        async def local(self, action): pass
    monkeypatch.setattr('app.services.setup.sys', SimpleNamespace(platform='darwin'))
    monkeypatch.setattr('app.services.setup.platform.machine', lambda: 'arm64')
    monkeypatch.setattr('app.services.setup.hardware', lambda p: dict(free_disk_gb=50,gpus=[]))
    service=Setup(settings, Speech(), None, SimpleNamespace(state={'running':False})); calls=[]
    async def python(): return Path('/native/python3.12')
    async def run(args, stage, timeout=2400):
        calls.append(list(map(str,args)))
        if 'venv' in args:
            target=venv_python(Path(args[-1])); target.parent.mkdir(parents=True); target.touch()
    async def download(python, component, folder):
        folder.mkdir(exist_ok=True)
        for name in ASR_FILES if component=='whisper' else VOICE_FILES: (folder/name).write_bytes(b'fixture')
    monkeypatch.setattr(service,'_python',python); monkeypatch.setattr(service,'_run',run); monkeypatch.setattr(service,'_download',download)
    await service.start('voice',True); await service.task
    assert service.state['stage']=='ready'
    assert settings.snapshot()['speech']['tts']['device']=='cpu'
    assert not any('--index-url' in c for c in calls)
    assert settings.snapshot()['speech']['runtime']['python'] != settings.snapshot()['speech']['runtime']['tts_python']


def test_posix_shutdown_targets_only_owned_session(monkeypatch):
    if sys.platform != 'darwin': pytest.skip('POSIX-specific process groups')
    calls=[]
    monkeypatch.setattr('app.services.local_runtime.os.getpgid', lambda pid: pid)
    monkeypatch.setattr('app.services.local_runtime.os.killpg', lambda pid,sig: calls.append(pid))
    owned=SimpleNamespace(pid=12345,wait=lambda timeout: None)
    LocalRuntime._stop_owned_tree(owned)
    assert calls==[12345]


def test_mac_english_text_stays_on_stdin(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('speech_mac',Path(__file__).resolve().parents[2]/'scripts/speech_worker.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module,'sys',SimpleNamespace(platform='darwin'))
    calls=[]
    def run(args,**kw):
        import wave
        calls.append((args,kw))
        with wave.open(args[-1], 'wb') as wav:
            wav.setparams((1,2,22050,0,'NONE','not compressed')); wav.writeframes(b'\x00\x00' * 200)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',run)
    text='hello; $(arbitrary) --voice'
    assert module.english_voice(text,1.0).startswith(b'RIFF')
    assert calls[0][1]['input']==text and text not in calls[0][0]


async def test_mac_assistant_setup_reuses_native_ollama(settings,tmp_path,monkeypatch):
    import asyncio
    executable = tmp_path / 'ollama'
    executable.touch()
    settings.update({'local_runtime': {'executable': str(executable), 'models_dir': str(tmp_path / 'models')}})

    class Models:
        state = {'running': False, 'status': 'installed'}

        async def start(self, model, accepted):
            assert accepted
            self.task = asyncio.create_task(asyncio.sleep(0))

    async def start():
        pass

    async def status():
        return {'engine_ready': True, 'installed_models': []}

    async def unexpected_bootstrap(*args):
        raise AssertionError('Existing native Ollama needs neither Python nor a runtime download')

    service = Setup(settings, SimpleNamespace(), SimpleNamespace(start=start), Models())
    monkeypatch.setattr('app.services.setup.sys', SimpleNamespace(platform='darwin'))
    monkeypatch.setattr('app.services.setup.hardware', lambda p: dict(free_disk_gb=50, gpus=[]))
    monkeypatch.setattr(service, 'status', status)
    monkeypatch.setattr(service, '_python', unexpected_bootstrap)
    monkeypatch.setattr(service, '_download', unexpected_bootstrap)
    await service.start('assistant', True)
    await service.task
    assert service.state['stage'] == 'ready'


@pytest.mark.parametrize('unsafe',[False,True])
def test_mac_ollama_download_selects_native_verified_archive(tmp_path,monkeypatch,unsafe):
    import io, tarfile, shutil
    script=Path(__file__).resolve().parents[2]/'scripts/download_runtime.py'
    spec=importlib.util.spec_from_file_location('download_mac',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'sys',SimpleNamespace(platform='darwin'))
    source=tmp_path/'fixture.tgz'
    with tarfile.open(source,'w:gz') as tar:
        info=tarfile.TarInfo('../../escaped' if unsafe else 'ollama');info.size=7;info.mode=0o755
        tar.addfile(info,io.BytesIO(b'fixture'))
    calls=[]
    def download(client,url,target,expected):
        calls.append((url,expected));shutil.copyfile(source,target)
    monkeypatch.setattr(module,'download',download)
    target=tmp_path/'runtime';target.mkdir()
    if unsafe:
        with pytest.raises(tarfile.FilterError):module.ollama(target)
    else:
        module.ollama(target)
        assert (target/'ollama/ollama').read_bytes()==b'fixture'
    assert calls[0][0].endswith('/ollama-darwin.tgz')
    assert calls[0][1]=='2608dbb0a0f0136a198db9d48b4f74ece55f452314a39452fca35b7cf20c2589'
