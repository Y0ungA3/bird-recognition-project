$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$run='runs/baseline_full'
if(Test-Path -LiteralPath 'runs/latest_baseline.txt') { $run=(Get-Content -LiteralPath 'runs/latest_baseline.txt' -Raw).Trim() }
Write-Output "Run: $run"
$history=Join-Path $run 'history.csv'
if(Test-Path -LiteralPath $history) { Import-Csv -LiteralPath $history | Select-Object epoch,phase,seconds,val_accuracy,val_top5,val_macro_f1 | Format-Table }
$done=Join-Path $run 'completed.json'
if(Test-Path -LiteralPath $done) { Get-Content -LiteralPath $done } else { Write-Output 'Training has not completed. Check the training window or saved logs.' }
