$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:TEMP=Join-Path $PSScriptRoot 'cache'; $env:TMP=$env:TEMP
$env:MPLCONFIGDIR=Join-Path $PSScriptRoot 'cache\matplotlib'
$python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $python -u render_report_visuals.py
if($LASTEXITCODE -ne 0) {throw 'Report visualization failed'}
