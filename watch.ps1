# Live auto-refreshing training dashboard. Ctrl+C to stop (does NOT stop training).
if ($PSScriptRoot) { Set-Location -Path $PSScriptRoot }
else { Set-Location -Path "C:\Users\canhh\Workspace\code\stereo-uw" }
$statusScript = Join-Path $PSScriptRoot "status.ps1"
if (-not (Test-Path $statusScript)) { $statusScript = "C:\Users\canhh\Workspace\code\stereo-uw\status.ps1" }
while ($true) {
  Clear-Host
  & $statusScript
  Write-Host "(auto-refresh every 15s - Ctrl+C to stop watching; training keeps running)" -ForegroundColor DarkGray
  Start-Sleep -Seconds 15
}
