$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:TEMP=Join-Path $PSScriptRoot 'cache'; $env:TMP=$env:TEMP
$env:TORCH_HOME=Join-Path $PSScriptRoot 'cache\torch'
$env:MPLCONFIGDIR=Join-Path $PSScriptRoot 'cache\matplotlib'
$python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $python -u analyze_background.py
if($LASTEXITCODE -ne 0) {throw 'Background analysis failed'}
