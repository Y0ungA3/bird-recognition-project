$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:TEMP=Join-Path $PSScriptRoot 'cache'; $env:TMP=$env:TEMP
$env:TORCH_HOME=Join-Path $PSScriptRoot 'cache\torch'
$env:MPLCONFIGDIR=Join-Path $PSScriptRoot 'cache\matplotlib'
$python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
foreach($policy in @(
    @{prefix='freeze_head';freeze=10},
    @{prefix='finetune_all';freeze=0}
)) {
    foreach($seed in @(42,43,44)) {
        $run="runs/$($policy.prefix)_seed${seed}"
        if(Test-Path -LiteralPath "$run/completed.json") {continue}
        $trainArgs=@('-u','train.py','--crop','--resize-mode','letterbox','--seed',"$seed",'--split-seed','42','--epochs','10','--freeze-epochs',"$($policy.freeze)",'--batch-size','32','--workers','0','--run',$run)
        if(Test-Path -LiteralPath "$run/config.json") {$trainArgs+='--resume'}
        & $python @trainArgs
        if($LASTEXITCODE -ne 0) {throw "Training failed: $run"}
    }
}
& $python -u analyze_freeze_finetune.py
if($LASTEXITCODE -ne 0) {throw 'Freeze/fine-tune analysis failed'}
