$ErrorActionPreference = "Stop"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) {
    $venvPython
} else {
    (Get-Command python -ErrorAction Stop).Source
}
$viewer = Join-Path $PSScriptRoot "scripts\view_baseline.py"

& $python $viewer
exit $LASTEXITCODE
