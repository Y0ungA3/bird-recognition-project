$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:TEMP=Join-Path $PSScriptRoot 'cache'; $env:TMP=$env:TEMP
$env:TORCH_HOME=Join-Path $PSScriptRoot 'cache\torch'
$env:MPLCONFIGDIR=Join-Path $PSScriptRoot 'cache\matplotlib'
$python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
foreach($seed in @(42,43,44)) {
    $run="runs/full_letterbox_seed${seed}"
    if(Test-Path -LiteralPath "$run/completed.json") {continue}
    $trainArgs=@('-u','train.py','--resize-mode','letterbox','--seed',"$seed",'--split-seed','42','--epochs','10','--batch-size','32','--workers','0','--run',$run)
    if(Test-Path -LiteralPath "$run/config.json") {$trainArgs+='--resume'}
    & $python @trainArgs
    if($LASTEXITCODE -ne 0) {throw "Training failed: $run"}
}
& $python -u analyze_full_letterbox.py
if($LASTEXITCODE -ne 0) {throw 'Controlled comparison failed'}
