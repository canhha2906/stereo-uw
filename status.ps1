# Training matrix status board. Runs from anywhere — locates its own folder.
$ErrorActionPreference = "SilentlyContinue"
if ($PSScriptRoot) { Set-Location -Path $PSScriptRoot }
else { Set-Location -Path "C:\Users\canhh\Workspace\code\stereo-uw" }
$stages = @(
  @{n="agg2d      pretrain"; f="log_agg2d_pre.log"},
  @{n="agg2d      finetune"; f="log_agg2d_ft.log"},
  @{n="agg2d_ctx  pretrain"; f="log_agg2dctx_pre.log"},
  @{n="agg2d_ctx  finetune"; f="log_agg2dctx_ft.log"},
  @{n="ref        pretrain"; f="log_ref_pre.log"},
  @{n="ref        finetune"; f="log_ref_ft.log"},
  @{n="ref_ctx    pretrain"; f="log_refctx_pre.log"},
  @{n="ref_ctx    finetune"; f="log_refctx_ft.log"}
)
Write-Host "`n=== STEREO MATRIX STATUS ===" -ForegroundColor Cyan
foreach ($s in $stages) {
  if (Test-Path $s.f) {
    $vals = Select-String -Path $s.f -Pattern "val EPE = ([0-9.]+)\s+\(best so far ([0-9.inf]+)\)"
    if ($vals) {
      $lastEp = ($vals | Select-Object -Last 1).Line.Trim()
      $best = ($vals | ForEach-Object { [double]$_.Matches.Groups[1].Value } | Measure-Object -Minimum).Minimum
      Write-Host ("{0} | epochs done: {1,2} | BEST EPE: {2:N4} | {3}" -f $s.n, $vals.Count, $best, $lastEp)
    } else {
      $tail = (Get-Content $s.f -Tail 1)
      Write-Host ("{0} | starting... | {1}" -f $s.n, $tail)
    }
  } else {
    Write-Host ("{0} | (not started)" -f $s.n) -ForegroundColor DarkGray
  }
}
Write-Host "---"
$p = Get-Process python -ErrorAction SilentlyContinue
if ($p) {
  Write-Host ("RUNNING: python PID {0} (started {1})" -f ($p.Id -join ','), $p[0].StartTime) -ForegroundColor Green
  # show live progress of the most recently written log
  $active = Get-ChildItem log_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  Write-Host ("ACTIVE LOG: {0}" -f $active.Name) -ForegroundColor Green
  Write-Host ("  " + (Get-Content $active.FullName -Tail 1))
} else {
  Write-Host "no python running (matrix idle or finished)" -ForegroundColor Yellow
}
Write-Host ""
