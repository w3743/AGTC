<#
.SYNOPSIS
  Formal native V1.1 trial runner for Windows (mirror of run_native_v1_trial.sh).

.DESCRIPTION
  Verifies the frozen protocol SHA (V1.1), creates the trial directory, copies the
  protocol files, runs the formal LSTM recipe on 1 ms/50 physics, and writes
  trial_manifest.json. Requires the native stack Python environment on PATH
  (or pass -Python).

.EXAMPLE
  .\run_native_v1_trial.ps1 -Tag pilot02
  .\run_native_v1_trial.ps1 -Tag pilot02 -LiveStateFile D:\tmp\native_live.json -LiveStateInterval 5
#>
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Python = "python",
    [double]$SimDt = 0.001,
    [int]$Decimation = 50,
    [string]$LiveStateFile = "",
    [int]$LiveStateInterval = 5,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$trial = Join-Path "runs" "native_v1_1_${Tag}_${stamp}"
New-Item -ItemType Directory -Force -Path $trial | Out-Null

Write-Host "=== protocol SHA check (V1.1) ==="
& $Python "protocol_sha.py" --check
if ($LASTEXITCODE -ne 0) { Write-Error "PROTOCOL_SHA_MISMATCH"; exit 2 }
Copy-Item "PROTOCOL_NATIVE_V1.1.md", "PROTOCOL_NATIVE_V1.1.sha256", "PROTOCOL_NATIVE_V1.md", "PROTOCOL_NATIVE_V1.sha256" -Destination $trial

$trainArgs = @(
    "--policy", "lstm", "--num-envs", "64", "--num-steps", "32", "--iterations", "300",
    "--seed", "0", "--threads", "8", "--device", "cuda", "--physics", "newton_xpbd",
    "--lidar", "warp", "--feedback", "vqf", "--depth", "analytic", "--detections", "analytic",
    "--detections-warp", "--sim-dt", "$SimDt", "--decimation", "$Decimation", "--profile",
    "--save-interval", "20", "--out", "$trial"
)
if ($LiveStateFile) { $trainArgs += @("--live-state-file", $LiveStateFile, "--live-state-interval", "$LiveStateInterval") }
if ($ExtraArgs.Count -gt 0) { $trainArgs += $ExtraArgs }

"PHYSICS sim_dt=$SimDt decimation=$Decimation`n" | Tee-Object -FilePath (Join-Path $trial "physics.txt")
($trainArgs -join " ") | Set-Content -Path (Join-Path $trial "cli_args.txt") -Encoding utf8

Write-Host "=== formal trial train: $trial ==="
& $Python "-u" "train_wsl.py" @trainArgs 2>&1 | Tee-Object -FilePath (Join-Path $trial "train.log") | Select-String -Pattern "IDENTITY|ITER|SUMMARY|Error|Traceback" | Select-Object -First 8
$rc = $LASTEXITCODE

$protocol = Get-Content (Join-Path $trial "PROTOCOL_NATIVE_V1.1.sha256") -Raw | ConvertFrom-Json
$contractPath = Join-Path $trial "observation_contract.json"
$contractSha = $null
if (Test-Path $contractPath) { $contractSha = (Get-Content $contractPath -Raw | ConvertFrom-Json).observation_contract_sha256 }
$manifest = [ordered]@{
    protocol_sha256      = $protocol.protocol_sha256
    protocol_file        = "PROTOCOL_NATIVE_V1.1.md"
    protocol_version     = "V1.1"
    created_at           = (Get-Date).ToString("o")
    cli_args_file        = "cli_args.txt"
    exit_code            = $rc
    observation_contract_sha256 = $contractSha
    capability_metrics_allowed  = $false
    phase                = "formal_native_v1_trial"
    physics              = [ordered]@{ sim_dt = $SimDt; decimation = $Decimation; experimental = $false }
    checkpoints          = @(Get-ChildItem $trial -Filter "*.pt" | ForEach-Object { $_.Name } | Sort-Object)
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $trial "trial_manifest.json") -Encoding utf8
Write-Host ("TRIAL_MANIFEST " + ($manifest | ConvertTo-Json -Depth 5 -Compress))
Write-Host "TRIAL_DIR=$trial exit=$rc"
exit $rc
