$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:TEMP=Join-Path $PSScriptRoot 'cache'; $env:TMP=$env:TEMP
$python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
foreach($seed in @(42,43,44)) {
 $run="runs/letterbox_seed${seed}"
 if(Test-Path -LiteralPath "$run/completed.json") {continue}
 $trainArgs=@('-u','train.py','--crop','--resize-mode','letterbox','--seed',"$seed",'--split-seed','42','--run',$run)
 if(Test-Path -LiteralPath "$run/config.json") {$trainArgs+='--resume'}
 & $python @trainArgs
 if($LASTEXITCODE -ne 0) {throw "Training failed: $run"}
}
& $python -u compare_letterbox.py
if($LASTEXITCODE -ne 0) {throw 'Letterbox analysis failed'}
