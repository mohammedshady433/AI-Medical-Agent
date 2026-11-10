. "$PSScriptRoot/common.ps1"
Assert-Tools
Start-Transcript -Path 'logs/tests.log'
try {
    Invoke-Checked $Python @('scripts/validate.py')
    Invoke-Checked $Python @('-m','pytest','-q','--basetemp=build/pytest-temp','--junitxml=logs/backend-tests.xml')
    Write-Host 'Automated tests passed. Mocked provider tests do not establish live model, microphone, or voice quality.'
} finally {Stop-Transcript}
