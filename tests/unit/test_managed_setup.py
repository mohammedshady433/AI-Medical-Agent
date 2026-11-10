import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.core.settings import Settings
from app.core.platforms import venv_python, ollama_name
from app.services.setup import Setup, ASR_FILES, VOICE_FILES, recover_installed, remember_installation


def installed(settings):
    root = settings.data / 'runtime'
    for name in (venv_python(Path('speech-env')), venv_python(Path('voice-env')), Path('ollama') / ollama_name()):
        file = root / name; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(b'fixture')
    for folder, names in [('whisper', ASR_FILES), ('chatterbox-v3', VOICE_FILES)]:
        for name in names:
            file = root / folder / name; file.parent.mkdir(exist_ok=True); file.write_bytes(b'fixture')
    return root


def test_reconnect_standard_installation_without_changing_selected_voice(settings):
    installed(settings)
    before = settings.snapshot()
    assert recover_installed(settings)
    after = settings.snapshot()
    assert after['speech']['runtime']['whisper_path'].endswith('runtime\\whisper') or after['speech']['runtime']['whisper_path'].endswith('runtime/whisper')
    assert after['speech']['tts'] == before['speech']['tts']
    assert after['models'] == before['models']
    assert after['local_runtime']['executable'].endswith(ollama_name())
    remember_installation(settings)
    registry = json.loads((settings.data / 'runtime/installation.json').read_text())
    assert set(registry) == {'speech', 'local_runtime'}
    assert not recover_installed(settings)


def test_registry_recovers_paths_after_settings_reset(settings, tmp_path):
    root = installed(settings); recover_installed(settings); remember_installation(settings)
    settings.update({'speech': {'runtime': {'python': '', 'whisper_path': ''}}})
    assert recover_installed(settings)
    assert settings.snapshot()['speech']['runtime']['python'] == str(venv_python(root / 'speech-env'))
    # A working custom path always wins over a standard/registry candidate.
    custom = tmp_path / 'custom.exe'; custom.write_bytes(b'fixture')
    settings.update({'speech': {'runtime': {'python': str(custom)}}})
    recover_installed(settings)
    assert settings.snapshot()['speech']['runtime']['python'] == str(custom)


def test_incomplete_model_is_not_recovered(settings):
    root = installed(settings)
    (root / 'whisper/tokenizer.json').unlink()
    recover_installed(settings)
    assert not settings.snapshot()['speech']['runtime']['whisper_path']


def test_reload_saved_settings_without_restart_or_exists_gate(settings, monkeypatch):
    settings.path.write_text(json.dumps({'language': 'en', 'assistant': {'voice_model': 'qwen2.5:7b-instruct'}}))
    assert settings.reload()
    monkeypatch.setattr(Path, 'exists', lambda *a, **kw: False)
    restarted = Settings(settings.data)
    assert restarted.snapshot()['assistant']['voice_model'] == 'qwen2.5:7b-instruct'
    assert restarted.snapshot()['language'] == 'en'


def test_bad_saved_settings_are_not_silently_discarded(settings):
    settings.path.write_text('{invalid')
    with pytest.raises(ValueError): settings.reload()


def test_settings_survive_transient_windows_reader_lock(settings, monkeypatch):
    import os
    replace = os.replace
    calls = []
    def locked(source, target):
        calls.append(target)
        if len(calls) < 3: raise PermissionError('Windows sharing violation')
        replace(source, target)
    monkeypatch.setattr('app.core.settings.os.replace', locked)
    settings.update({'assistant': {'voice_model': 'gpt-oss:20b'}})
    assert Settings(settings.data).snapshot()['assistant']['voice_model'] == 'gpt-oss:20b'


def test_permanent_write_denial_preserves_previous_settings(settings, monkeypatch):
    settings.update({'language': 'en'})
    before = settings.path.read_bytes()
    def denied(*args): raise PermissionError('Permanent denial')
    monkeypatch.setattr('app.core.settings.os.replace', denied)
    monkeypatch.setattr('app.core.settings.time.sleep', lambda _: None)
    with pytest.raises(PermissionError): settings.update({'language': 'ar'})
    assert settings.path.read_bytes() == before
    assert settings.snapshot()['language'] == 'en'


@pytest.mark.asyncio
async def test_setup_requires_consent_and_reuses_existing_files(settings, monkeypatch):
    installed(settings)
    class Speech:
        def status(self, language): return {'ready': True, 'recognition': True, 'synthesis': True}
    models = SimpleNamespace(state={'running': False})
    service = Setup(settings, Speech(), None, models)
    with pytest.raises(ValueError): await service.start('voice', False)
    async def no_python(): raise AssertionError('A complete installation must not download/install Python')
    monkeypatch.setattr(service, '_python', no_python)
    monkeypatch.setattr('app.services.setup.hardware', lambda p: {'free_disk_gb': 40, 'gpus': []})
    await service.start('voice', True); await service.task
    assert service.state['stage'] == 'ready'
    assert not service.state['running']
    assert (settings.data / 'runtime/installation.json').is_file()


@pytest.mark.asyncio
async def test_setup_reports_disk_space_without_starting_installation(settings, monkeypatch):
    speech = SimpleNamespace(status=lambda language: {'ready': False})
    service = Setup(settings, speech, None, SimpleNamespace(state={'running': False}))
    monkeypatch.setattr('app.services.setup.hardware', lambda p: {'free_disk_gb': 1, 'gpus': []})
    await service.start('voice', True); await service.task
    assert service.state['reason'] == 'disk_space'
    assert service.state['error'] and not service.state['running']


@pytest.mark.asyncio
async def test_managed_voice_install_uses_separate_pinned_environments(settings, monkeypatch):
    class Speech:
        def status(self, language):
            ready = complete(settings.snapshot()['speech']['runtime']['whisper_path'], ASR_FILES) and complete(settings.snapshot()['speech']['runtime']['chatterbox_path'], VOICE_FILES)
            return {'ready': ready, 'recognition': ready, 'synthesis': ready}
        async def close(self): pass
        async def local(self, action): assert action == 'warmup'
    from app.services.setup import complete
    service = Setup(settings, Speech(), None, SimpleNamespace(state={'running': False}))
    monkeypatch.setattr('app.services.setup.sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr('app.services.setup.hardware', lambda p: {'free_disk_gb': 50, 'gpus': [{'vram_mb': 16384}]})
    commands = []
    async def python(): return Path('python.exe')
    async def run(args, stage, timeout=2400):
        commands.append([str(x) for x in args])
        if '-m' in args and 'venv' in args:
            target = venv_python(Path(args[-1])); target.parent.mkdir(parents=True); target.write_bytes(b'fixture')
    async def download(python, component, folder):
        for name in ASR_FILES if component == 'whisper' else VOICE_FILES:
            target = folder / name; target.parent.mkdir(exist_ok=True); target.write_bytes(b'fixture')
    monkeypatch.setattr(service, '_python', python); monkeypatch.setattr(service, '_run', run); monkeypatch.setattr(service, '_download', download)
    await service.start('voice', True); await service.task
    cfg = settings.snapshot()['speech']
    assert service.state['stage'] == 'ready'
    assert cfg['runtime']['python'] != cfg['runtime']['tts_python']
    assert cfg['tts']['provider'] == 'chatterbox' and cfg['tts']['voice'] == 'egyptian'
    assert cfg['asr']['device'] == 'cuda'
    assert any('torch==2.6.0' in x and 'https://download.pytorch.org/whl/cu124' in x for x in commands)
    assert any('--force-reinstall' in x and any('5de7a54a' in arg for arg in x) for x in commands)


@pytest.mark.asyncio
async def test_setup_cancel_rejects_overlapping_jobs(settings, monkeypatch):
    speech = SimpleNamespace(status=lambda language: {'ready': False})
    service = Setup(settings, speech, None, SimpleNamespace(state={'running': False}))
    monkeypatch.setattr('app.services.setup.hardware', lambda p: {'free_disk_gb': 50, 'gpus': []})
    entered = asyncio.Event()
    async def python(): entered.set(); await asyncio.Event().wait()
    monkeypatch.setattr(service, '_python', python)
    await service.start('voice', True); await entered.wait()
    with pytest.raises(ValueError): await service.start('assistant', True)
    await service.close()
    assert not service.state['running'] and service.state['stage'] == 'cancelled'


def test_invalid_install_registry_does_not_prevent_startup(settings):
    root = settings.data / 'runtime'; root.mkdir()
    for value in ('[1,2]', '{"speech":["bad"],"local_runtime":false}', '{"speech":{"python":123}}'):
        (root / 'installation.json').write_text(value)
        assert not recover_installed(settings)


@pytest.mark.asyncio
async def test_assistant_setup_reuses_20b_and_downloads_7b_without_switching(settings, monkeypatch):
    settings.update({'assistant': {'voice_model': 'gpt-oss:20b'}})
    installed_models = ['gpt-oss:20b', settings.snapshot()['documents']['embedding']['model']]
    pulled = []

    class Models:
        state = {'running': False, 'status': 'installed'}

        async def start(self, model, accepted):
            assert accepted
            pulled.append(model)
            self.task = asyncio.create_task(asyncio.sleep(0))

    async def noop():
        pass

    async def status():
        return {'engine_ready': True, 'installed_models': installed_models}

    service = Setup(settings, None, SimpleNamespace(start=noop), Models())
    monkeypatch.setattr(service, 'status', status)
    monkeypatch.setattr('app.services.setup.hardware', lambda _: {'free_disk_gb': 50})
    await service.start('assistant', True)
    await service.task
    assert service.state['stage'] == 'ready'
    assert pulled == ['qwen2.5:7b-instruct']
    assert settings.snapshot()['assistant']['voice_model'] == 'gpt-oss:20b'
