# Starts one Chrome per Kaggle account, each with its own persistent profile + CDP debug port.
# Profiles live OUTSIDE the repo (cookies = login). Log in once per profile; they stay logged in across restarts
# as long as the profile folders are kept and Chrome is closed normally (not killed).
$chrome = (Get-Command chrome.exe -ErrorAction SilentlyContinue).Source
if (-not $chrome) { $chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe" }
$root = "E:\train-lora\_tools"
$map = @(@(9222,'profile'), @(9223,'profile_c'), @(9224,'profile_b'), @(9225,'profile_d'))
foreach ($m in $map) {
  $up = $false
  try { Invoke-WebRequest "http://127.0.0.1:$($m[0])/json/version" -UseBasicParsing -TimeoutSec 2 | Out-Null; $up = $true } catch {}
  if ($up) { "port $($m[0]) already running"; continue }
  Start-Process $chrome -ArgumentList "--remote-debugging-port=$($m[0])", "--user-data-dir=$root\$($m[1])", "--restore-last-session", "--window-size=1500,950", "https://www.kaggle.com/"
  Start-Sleep -Seconds 3
}
