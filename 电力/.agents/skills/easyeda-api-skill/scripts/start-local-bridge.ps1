param([switch]$LaunchEda)
$ErrorActionPreference = 'Stop'
function Find-LocalBridge {
    foreach ($port in 49620..49629) {
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 1
            if ($health.service -eq 'easyeda-bridge') {
                return [pscustomobject]@{ Port = $port; Health = $health }
            }
        } catch {}
    }
    return $null
}
$bridge = Find-LocalBridge
if (-not $bridge) {
    $bridgePath = Join-Path $PSScriptRoot 'bridge-server.mjs'
    $node = (Get-Command node -ErrorAction Stop).Source
    $logDir = Join-Path $env:TEMP 'agtc-eda-bridge'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    Start-Process -FilePath $node -ArgumentList ('"' + $bridgePath + '"') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'stdout.log') -RedirectStandardError (Join-Path $logDir 'stderr.log') | Out-Null
    # Wait for this startup to finish before considering any further action.
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        Start-Sleep -Milliseconds 300
        $bridge = Find-LocalBridge
        if ($bridge) { break }
    }
    if (-not $bridge) { throw "Bridge did not become healthy; inspect $logDir" }
}
if ($LaunchEda -and -not (Get-Process -Name lceda-pro -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath 'C:\Program Files\lceda-pro\lceda-pro.exe' -WindowStyle Hidden | Out-Null
}
if ($LaunchEda -and -not $bridge.Health.edaConnected) {
    # The extension connects asynchronously after the desktop process starts.
    $healthUri = "http://127.0.0.1:$($bridge.Port)/health"
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 400
        try {
            $bridge.Health = Invoke-RestMethod $healthUri -TimeoutSec 1
            if ($bridge.Health.edaConnected) { break }
        } catch {}
    }
}
[pscustomobject]@{
    Port = $bridge.Port
    EdaConnected = $bridge.Health.edaConnected
    WindowCount = $bridge.Health.edaWindowCount
    Note = if (-not $bridge.Health.edaConnected) { 'Use API Gateway > reconnect in the existing EDA client.' }
} | ConvertTo-Json
