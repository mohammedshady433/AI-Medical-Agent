param([switch]$DownloadModels,[switch]$Cuda)
. "$PSScriptRoot/common.ps1"
Assert-Tools
$VoiceRoot = Join-Path $env:LOCALAPPDATA 'EMS-AI-Assistant/runtime'
$VoicePython = Join-Path $VoiceRoot 'voice-env/Scripts/python.exe'
New-Item -ItemType Directory -Force $VoiceRoot | Out-Null
Invoke-Checked $Python @('scripts/hardware_check.py')
if (!(Test-Path $VoicePython)) {Invoke-Checked $Python @('-m','venv',(Join-Path $VoiceRoot 'voice-env'))}
$TorchIndex = if ($Cuda) {'https://download.pytorch.org/whl/cu124'} else {'https://download.pytorch.org/whl/cpu'}
Invoke-Checked $VoicePython @('-m','pip','install','torch==2.6.0','torchaudio==2.6.0','--index-url',$TorchIndex)
Invoke-Checked $VoicePython @('-m','pip','install','-r','requirements-chatterbox.in')
# Upstream retained version 0.1.7, so a normal install can keep the older PyPI code.
$VoiceRevision = (Get-Content 'requirements-chatterbox.in' | Where-Object {$_ -match '^chatterbox-tts @ '})
Invoke-Checked $VoicePython @('-m','pip','install','--force-reinstall','--no-deps',$VoiceRevision)
Invoke-Checked $VoicePython @('-m','pip','check')
if ($DownloadModels) {
    Write-Host 'Downloading the pinned Chatterbox Multilingual V3 voice (about 3.2 GB). Interrupted downloads resume.'
    Invoke-Checked $Python @('scripts/download_runtime.py','chatterbox','--destination',(Join-Path $VoiceRoot 'chatterbox-v3'))
}
$Arguments = @('scripts/configure_runtime.py','--tts-root',$VoiceRoot)
if ($Cuda) {$Arguments += '--cuda'}
Invoke-Checked $Python $Arguments
Write-Host 'Local voice paths configured. In EMS Voice Settings, select Chatterbox after download completes. Restart EMS to load changed runtimes.'
