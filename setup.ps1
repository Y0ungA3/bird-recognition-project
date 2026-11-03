$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:TEMP=Join-Path $PSScriptRoot 'cache'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $PSScriptRoot 'cache\pip'
$env:TORCH_HOME=Join-Path $PSScriptRoot 'cache\torch'
if (!(Test-Path -LiteralPath '.venv\Scripts\python.exe')) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'GPU package installation failed' }
& .\.venv\Scripts\python.exe -m pip install matplotlib pillow numpy
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& .\.venv\Scripts\python.exe check_gpu.py
if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed' }
