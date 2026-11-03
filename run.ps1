param([ValidateSet('baseline','crop','smoke','resume','evaluate')][string]$Mode='baseline')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:TORCH_HOME=Join-Path $PSScriptRoot 'cache\torch'
$env:TEMP=Join-Path $PSScriptRoot 'cache'
$env:TMP=$env:TEMP
$env:PYTHONIOENCODING='utf-8'
$python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$latestFile=Join-Path $PSScriptRoot 'runs\latest_baseline.txt'
$baseline='runs/baseline_full'
if(Test-Path -LiteralPath $latestFile) { $baseline=(Get-Content -LiteralPath $latestFile -Raw).Trim() }
switch ($Mode) {
 'baseline' {
   $newRun='runs/baseline_full'
   if(Test-Path -LiteralPath (Join-Path $PSScriptRoot "$newRun/config.json")) { $newRun='runs/baseline_'+(Get-Date -Format 'yyyyMMdd_HHmmss') }
   $newRun | Set-Content -LiteralPath $latestFile
   & $python -u train.py --run $newRun
 }
 'crop' { & $python -u train.py --crop --run ('runs/crop_'+(Get-Date -Format 'yyyyMMdd_HHmmss')) }
 'smoke' { & $python -u train.py --smoke --epochs 1 --workers 0 --run ('runs/smoke_'+(Get-Date -Format 'yyyyMMdd_HHmmss')) }
 'resume' { & $python -u train.py --resume --run $baseline }
 'evaluate' { & $python -u train.py --evaluate --run $baseline }
}
exit $LASTEXITCODE
