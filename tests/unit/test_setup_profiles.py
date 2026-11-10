"""Exercise the Windows provisioning command plan without network or downloads."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell provisioning")
@pytest.mark.parametrize("options,models", [([], ["gpt-oss:20b", "bge-m3:567m"]), (["-EmbeddingsOnly"], ["bge-m3:567m"]), (["-AssistantModel", "7B"], ["qwen2.5:7b-instruct", "bge-m3:567m"]), (["-AssistantModel", "Both", "-IncludeVision"], ["qwen2.5:7b-instruct", "gpt-oss:20b", "qwen2.5vl:3b", "bge-m3:567m"])])
def test_setup_profile_download_plan(tmp_path, options, models):
    script = Path(__file__).resolve().parents[2] / "scripts/setup-models.ps1"
    shutil.copyfile(script, tmp_path / script.name)
    (tmp_path / "common.ps1").write_text('''
$ErrorActionPreference = 'Stop'
$Python = 'fake-python.exe'
function Assert-Tools {}
function Invoke-RestMethod { return @{version='test'} }
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    ConvertTo-Json -Compress -InputObject @($Executable, $Arguments) | Add-Content -LiteralPath (Join-Path $PSScriptRoot 'calls.jsonl')
}
''', encoding="utf8")
    runtime = tmp_path / "EMS-AI-Assistant/runtime/ollama"
    runtime.mkdir(parents=True)
    (runtime / "ollama.exe").touch()
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(tmp_path / script.name), *options], env={**os.environ, "LOCALAPPDATA": str(tmp_path)}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    pulled = [args[1] for _, args in calls if args[0] == "pull"]
    assert pulled == models
    assert any(args[0] == "scripts/configure_runtime.py" for _, args in calls)
    if "-EmbeddingsOnly" in options:
        assert calls[0][1] == ["scripts/hardware_check.py", "--embeddings-only"]


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell provisioning")
def test_conflicting_setup_profiles_stop_before_download(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/setup-models.ps1"
    shutil.copyfile(script, tmp_path / script.name)
    (tmp_path / "common.ps1").write_text("$ErrorActionPreference = 'Stop'\nfunction Assert-Tools { throw 'Unexpected setup side effect' }\n", encoding="utf8")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(tmp_path / script.name), "-EmbeddingsOnly", "-Experimental120B"], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Choose either" in result.stderr
    assert "Unexpected setup side effect" not in result.stderr
