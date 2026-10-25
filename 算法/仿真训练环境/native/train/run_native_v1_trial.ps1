<#
.SYNOPSIS
  Formal native V1.1 runner with an immutable 1 ms/50 recipe.

.DESCRIPTION
  Verifies the frozen protocol, records the full unfiltered training log, preserves
  the trainer exit code, and validates the generated task/physics contract.
  Set -Python or PYTHON to select the Python runtime. Extra arguments are limited
  to live-state viewer options.

.EXAMPLE
  .\run_native_v1_trial.ps1 -Tag pilot02
  .\run_native_v1_trial.ps1 -Tag pilot02 -LiveStateFile D:\tmp\native_live.json
#>
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Python = "",
    [double]$SimDt = 0.001,
    [int]$Decimation = 50,
    [string]$LiveStateFile = "",
    [int]$LiveStateInterval = 5,
    [double]$LiveStateSpacing = 2.5,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$caller = (Get-Location).Path
if (-not $Python) {
    if ($env:PYTHON) {
        $Python = $env:PYTHON
    } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        $Python = (Get-Command python3).Source
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $Python = (Get-Command python).Source
    } else {
        [Console]::Error.WriteLine("ERROR: set -Python or PYTHON to a Python executable")
        exit 2
    }
}
$helper = Join-Path $here "trial_checks.py"
$protocolSha = Join-Path $here "PROTOCOL_NATIVE_V1.1.sha256"
& $Python $helper require-official-physics --sim-dt "$SimDt" --decimation "$Decimation"
$physicsCheckRc = $LASTEXITCODE
if ($physicsCheckRc -ne 0) { exit 2 }

$viewerArgs = @()
$seen = @{}
if ($LiveStateFile) {
    if (-not [IO.Path]::IsPathRooted($LiveStateFile)) {
        $LiveStateFile = [IO.Path]::GetFullPath((Join-Path $caller $LiveStateFile))
    }
    $viewerArgs += @("--live-state-file", $LiveStateFile)
    $seen["--live-state-file"] = $true
    $viewerArgs += @("--live-state-interval", "$LiveStateInterval")
    $seen["--live-state-interval"] = $true
    $viewerArgs += @("--live-state-spacing", "$LiveStateSpacing")
    $seen["--live-state-spacing"] = $true
}
for ($i = 0; $i -lt $ExtraArgs.Count; $i++) {
    $flag = $ExtraArgs[$i]
    if ($flag -notin @("--live-state-file", "--live-state-interval", "--live-state-spacing")) {
        [Console]::Error.WriteLine("ERROR: unsupported extra argument: $flag")
        exit 2
    }
    if ($seen.ContainsKey($flag)) {
        [Console]::Error.WriteLine("ERROR: duplicate viewer option: $flag")
        exit 2
    }
    if ($i + 1 -ge $ExtraArgs.Count -or [string]::IsNullOrWhiteSpace($ExtraArgs[$i + 1])) {
        [Console]::Error.WriteLine("ERROR: $flag requires a value")
        exit 2
    }
    $value = $ExtraArgs[$i + 1]
    if ($flag -eq "--live-state-file" -and -not [IO.Path]::IsPathRooted($value)) {
        $value = [IO.Path]::GetFullPath((Join-Path $caller $value))
    }
    $viewerArgs += @($flag, $value)
    $seen[$flag] = $true
    $i++
}

Set-Location $here
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$trial = Join-Path (Join-Path $here "runs") "native_v1_1_${Tag}_${stamp}"
New-Item -ItemType Directory -Force -Path $trial | Out-Null

Write-Host "=== protocol SHA check (V1.1) ==="
& $Python (Join-Path $here "protocol_sha.py") --check
$protocolRc = $LASTEXITCODE
if ($protocolRc -ne 0) {
    [Console]::Error.WriteLine("PROTOCOL_SHA_MISMATCH")
    exit 2
}
Copy-Item -LiteralPath @(
    (Join-Path $here "PROTOCOL_NATIVE_V1.1.md"),
    (Join-Path $here "PROTOCOL_NATIVE_V1.1.sha256"),
    (Join-Path $here "PROTOCOL_NATIVE_V1.md"),
    (Join-Path $here "PROTOCOL_NATIVE_V1.sha256")
) -Destination $trial

$trainArgs = @(
    "--policy", "lstm", "--lstm-hidden", "512", "--num-envs", "64", "--num-steps", "32",
    "--iterations", "300", "--seed", "0", "--threads", "8", "--device", "cuda",
    "--physics", "newton_xpbd", "--task-backend", "fast_stop_v2", "--lidar", "warp",
    "--feedback", "vqf", "--depth", "analytic", "--detections", "analytic", "--detections-warp",
    "--sim-dt", "0.001", "--decimation", "50", "--save-interval", "20", "--out", $trial
) + $viewerArgs
"PHYSICS sim_dt=0.001 decimation=50" | Set-Content -LiteralPath (Join-Path $trial "physics.txt") -Encoding utf8
(($Python, "-u", (Join-Path $here "train_wsl.py")) + $trainArgs) -join " " |
    Set-Content -LiteralPath (Join-Path $trial "cli_args.txt") -Encoding utf8

Write-Host "=== formal trial train: $trial ==="
& $Python -u (Join-Path $here "train_wsl.py") @trainArgs 2>&1 |
    Tee-Object -FilePath (Join-Path $trial "train.log") | Out-Host
$trainRc = $LASTEXITCODE

& $Python $helper write-manifest --trial-dir $trial --train-exit $trainRc --protocol-sha $protocolSha
$manifestRc = $LASTEXITCODE
Write-Host "TRIAL_DIR=$trial train_exit=$trainRc manifest_check=$manifestRc"
if ($trainRc -ne 0) { exit $trainRc }
exit $manifestRc