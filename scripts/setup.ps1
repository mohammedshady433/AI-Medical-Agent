param([string]$PythonExecutable = 'py')
. "$PSScriptRoot/common.ps1"
Start-Transcript -Path 'logs/setup.log' -Append
try {
    if (!(Test-Path $Python)) {
        if ($PythonExecutable -eq 'py') {Invoke-Checked 'py' @('-3.12','-m','venv','.venv')}
        else {Invoke-Checked $PythonExecutable @('-m','venv','.venv')}
    }
    Assert-Tools
    Invoke-Checked $Python @('-m','pip','install','--upgrade','pip==26.2.2')
    Invoke-Checked $Python @('-m','pip','install','-r','requirements.lock')
    Invoke-Checked $Python @('scripts/validate.py')
    Write-Host 'Development dependencies installed. No models or paid API calls were started.'
    Write-Host 'Start the AI core test console with: .venv\Scripts\python.exe scripts\test_console.py'
} finally {Stop-Transcript}
