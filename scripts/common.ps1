$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
New-Item -ItemType Directory -Force logs,build,.runtime/tmp | Out-Null
$env:TEMP = Join-Path $ProjectRoot '.runtime/tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.runtime/pip-cache'
$Python = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')"}
}
function Assert-Tools {
    if (!(Test-Path $Python)) {throw 'Python environment missing. Run scripts/setup.ps1 first.'}
    Invoke-Checked $Python @('-c',"import sys; assert sys.version_info[:2] == (3,12), 'EMS requires Python 3.12'")
}
