$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:TEMP=Join-Path $PSScriptRoot 'cache'; $env:TMP=$env:TEMP
$env:TORCH_HOME=Join-Path $PSScriptRoot 'cache\torch'
$env:MPLCONFIGDIR=Join-Path $PSScriptRoot 'cache\matplotlib'
$python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$conditions=@(
    @{label='000'; margin='0.0'},
    @{label='005'; margin='0.05'},
    @{label='020'; margin='0.2'}
)
foreach($condition in $conditions) {
    foreach($seed in @(42,43,44)) {
        $run="runs/margin$($condition.label)_seed${seed}"
        if(Test-Path -LiteralPath "$run/completed.json") {continue}
        $trainArgs=@('-u','train.py','--crop','--resize-mode','letterbox','--margin',$condition.margin,'--seed',"$seed",'--split-seed','42','--epochs','10','--freeze-epochs','0','--batch-size','32','--workers','0','--run',$run)
        if(Test-Path -LiteralPath "$run/config.json") {$trainArgs+='--resume'}
        & $python @trainArgs
        if($LASTEXITCODE -ne 0) {throw "Training failed: $run"}
    }
}
& $python -u analyze_margin_ablation.py
if($LASTEXITCODE -ne 0) {throw 'Margin ablation analysis failed'}
