$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:TEMP=Join-Path $PSScriptRoot 'cache'
$env:TMP=$env:TEMP
$env:PYTHONIOENCODING='utf-8'
$python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
foreach($seed in @(42,43,44)) {
 foreach($mode in @('full','crop')) {
  $run=if($mode -eq 'full' -and $seed -eq 42) {'runs/baseline_full'} elseif($mode -eq 'crop' -and $seed -eq 42) {'runs/crop_margin010'} else {"runs/comparison_${mode}_seed${seed}"}
  if(Test-Path -LiteralPath "$run/completed.json") { Write-Output "Already complete: $run"; continue }
  $trainArgs=@('-u','train.py','--seed',"$seed",'--split-seed','42','--run',$run)
  if($mode -eq 'crop') { $trainArgs+='--crop' }
  if(Test-Path -LiteralPath "$run/config.json") { $trainArgs+='--resume' }
  Write-Output "Starting $run"
  & $python @trainArgs
  if($LASTEXITCODE -ne 0) { throw "Training failed: $run" }
 }
}
& $python -u compare_experiments.py
if($LASTEXITCODE -ne 0) { throw 'Comparison analysis failed' }
