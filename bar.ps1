# Live loading bar for the currently-training config. Ctrl+C to stop watching.
if ($PSScriptRoot) { Set-Location -Path $PSScriptRoot }
else { Set-Location -Path "C:\Users\canhh\Workspace\code\stereo-uw" }

while ($true) {
  $active = Get-ChildItem log_*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $active) { Write-Progress -Activity "matrix" -Status "no logs yet"; Start-Sleep 3; continue }

  $stage = ($active.BaseName -replace '^log_','')
  # grab the most recent non-empty line
  $line = (Get-Content $active.FullName -Tail 1)
  if (-not $line) { $line = (Get-Content $active.FullName -Tail 3 | Where-Object {$_} | Select-Object -Last 1) }

  if ($line -match 'ep(\d+):\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]') {
    $ep=$matches[1]; $pct=[int]$matches[2]; $cur=$matches[3]; $tot=$matches[4]; $eta=$matches[5]
    Write-Progress -Activity ("{0}  -  epoch {1}" -f $stage,$ep) `
                   -Status ("iter {0}/{1}   [{2}]" -f $cur,$tot,$eta) `
                   -PercentComplete $pct
  }
  elseif ($line -match 'val EPE = ([0-9.]+)') {
    Write-Progress -Activity $stage -Status ("epoch done - val EPE $($matches[1]) (validating / next epoch)") -PercentComplete 100
  }
  else {
    Write-Progress -Activity $stage -Status $line -PercentComplete 0
  }
  Start-Sleep -Seconds 2
}
