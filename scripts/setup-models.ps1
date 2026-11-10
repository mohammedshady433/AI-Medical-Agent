param([switch]$InstallOllama,[switch]$Experimental120B,[switch]$AcceptSlow120B,[switch]$EmbeddingsOnly,
    [ValidateSet('7B','20B','Both')][string]$AssistantModel='20B',[switch]$IncludeVision)
. "$PSScriptRoot/common.ps1"
if ($EmbeddingsOnly -and $Experimental120B) {throw 'Choose either -EmbeddingsOnly or -Experimental120B, not both.'}
if ($Experimental120B -and $AssistantModel -ne '20B') {throw 'Experimental120B cannot be combined with AssistantModel.'}
Assert-Tools
$RuntimeRoot=Join-Path $env:LOCALAPPDATA 'EMS-AI-Assistant/runtime'
New-Item -ItemType Directory -Force $RuntimeRoot | Out-Null
$Ollama=Join-Path $RuntimeRoot 'ollama/ollama.exe'
Invoke-Checked $Python @('scripts/hardware_check.py',$(if ($EmbeddingsOnly) {'--embeddings-only'} elseif ($Experimental120B) {'--experimental'} else {'--recommended'}))
if ($Experimental120B -and !$AcceptSlow120B) {throw '120B needs approximately 65 GB storage and much more memory than this laptop. Rerun with -AcceptSlow120B only if you explicitly accept this experimental limitation.'}
if (!(Test-Path $Ollama)) {
    if (!$InstallOllama) {throw 'Private Ollama is missing. Rerun with -InstallOllama to download the verified runtime (~1.5 GB).'}
    Invoke-Checked $Python @('scripts/download_runtime.py','ollama','--destination',$RuntimeRoot)
}
$env:OLLAMA_HOST='127.0.0.1:11435'
$env:OLLAMA_MODELS=Join-Path $RuntimeRoot 'ollama-models'
$env:OLLAMA_NO_CLOUD='1'
$env:OLLAMA_MAX_LOADED_MODELS='2'
$env:OLLAMA_NUM_PARALLEL='1'
$Server=$null
try {
    try {Invoke-RestMethod 'http://127.0.0.1:11435/api/version' -TimeoutSec 2 | Out-Null}
    catch {
        $Server=Start-Process -FilePath $Ollama -ArgumentList 'serve' -WindowStyle Hidden -PassThru -RedirectStandardOutput 'logs/ollama-setup.log' -RedirectStandardError 'logs/ollama-setup-errors.log'
        Start-Sleep -Seconds 3
    }
    if (!$EmbeddingsOnly) {
        $Models=if ($Experimental120B) {@('gpt-oss:120b')} elseif ($AssistantModel -eq 'Both') {@('qwen2.5:7b-instruct','gpt-oss:20b')} elseif ($AssistantModel -eq '7B') {@('qwen2.5:7b-instruct')} else {@('gpt-oss:20b')}
        foreach ($Model in $Models) {Invoke-Checked $Ollama @('pull',$Model)}
    }
    if ($IncludeVision) {Invoke-Checked $Ollama @('pull','qwen2.5vl:3b')}
    Invoke-Checked $Ollama @('pull','bge-m3:567m')
    $ConfigArgs=@('scripts/configure_runtime.py','--ollama-root',$RuntimeRoot)
    if (!$EmbeddingsOnly -and !$Experimental120B) {$ConfigArgs += @('--assistant-model',$(if ($AssistantModel -eq '20B') {'gpt-oss:20b'} else {'qwen2.5:7b-instruct'}))}
    if ($IncludeVision) {$ConfigArgs += '--vision-model'}
    Invoke-Checked $Python $ConfigArgs
} finally {if ($null -ne $Server -and !$Server.HasExited) {Stop-Process -Id $Server.Id -ErrorAction SilentlyContinue}}
if ($EmbeddingsOnly) {Write-Host 'Local document embeddings installed. No offline assistant model was downloaded. Keep ONLINE selected for cloud answers.'}
else {Write-Host 'Models installed. EMS will start its configured private Ollama automatically; the chosen mode remains unchanged.'}
