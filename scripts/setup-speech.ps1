param([switch]$DownloadModels,[switch]$Cuda,[switch]$LegacyKemeTone)
. "$PSScriptRoot/common.ps1"
Assert-Tools
$SpeechRoot=Join-Path $env:LOCALAPPDATA 'EMS-AI-Assistant/runtime'
$SpeechPython=Join-Path $SpeechRoot 'speech-env/Scripts/python.exe'
New-Item -ItemType Directory -Force $SpeechRoot | Out-Null
if (!(Test-Path $SpeechPython)) {Invoke-Checked $Python @('-m','venv',(Join-Path $SpeechRoot 'speech-env'))}
$TorchIndex=if ($Cuda) {'https://download.pytorch.org/whl/cu124'} else {'https://download.pytorch.org/whl/cpu'}
Invoke-Checked $SpeechPython @('-m','pip','install','torch==2.6.0','torchaudio==2.6.0','--index-url',$TorchIndex)
Invoke-Checked $SpeechPython @('-m','pip','install','-r','requirements-speech.in')
if ($DownloadModels) {
    if ($LegacyKemeTone) {Invoke-Checked $Python @('scripts/download_runtime.py','kemetone','--destination',(Join-Path $SpeechRoot 'kemetone'))}
    Invoke-Checked $Python @('scripts/download_runtime.py','whisper','--destination',(Join-Path $SpeechRoot 'whisper'))
}
$RuntimeArguments = @('scripts/configure_runtime.py','--speech-root',$SpeechRoot)
if ($Cuda) {$RuntimeArguments += '--cuda'}
Invoke-Checked $Python $RuntimeArguments
Write-Host 'Recognition and Windows English runtime configured. Use setup-chatterbox.ps1 -DownloadModels for the new Arabic voice. Restart EMS if it is open.'
