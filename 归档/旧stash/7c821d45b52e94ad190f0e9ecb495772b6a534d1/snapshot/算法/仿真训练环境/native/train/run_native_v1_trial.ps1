<#
.SYNOPSIS
  Formal native V1.1 trial runner with fixed protocol and accurate run status.

.DESCRIPTION
  Verifies the frozen V1.1 protocol SHA, creates the trial folder, runs the
  LSTM-512 recipe on 1 ms / 50 fast_stop_v2 physics, and records the actual
  observation-contract physics/task values in trial_manifest.json.

.EXAMPLE
  .\run_native_v1_trial.ps1 -Tag pilot02
  .\run_native_v1_trial.ps1 -Tag pilot02 -LiveStateFile C:\tmp\native_live.json
#>
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Python = "",
    [double]$SimDt = 0.001,
    [int]$Decimation = 50,
    [string]$LiveStateFile = "",
    [int]$LiveStateInterval = 5,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $here

if (-not $Python -and $env:PYTHON) { $Python = $env:PYTHON }
if (-not $Python -and $env:VIRTUAL_ENV) {
    $candidate = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate) { $Python = $candidate }
}
if (-not $Python) {
    foreach ($candidate in @("python3", "python", "py")) {
        $resolved = Get-Command -Name $candidate -ErrorAction SilentlyContinue
        if ($resolved) { $Python = $candidate; break }
    }
}
if (-not $Python) {
    [Console]::Error.WriteLine("Python not found; pass -Python or set PYTHON.")
    exit 2
}
$pythonCommand = Get-Command -Name $Python -ErrorAction SilentlyContinue
if (-not $pythonCommand -and (Test-Path -LiteralPath $Python)) {
    $pythonCommand = Get-Item -LiteralPath $Python
}
if (-not $pythonCommand) {
    [Console]::Error.WriteLine("Python executable not found: $Python")
    exit 2
}
$pythonExe = if ($pythonCommand.Source) { $pythonCommand.Source } else { $pythonCommand.FullName }
$pythonPrefixArgs = @()
if ([IO.Path]::GetFileName($pythonExe) -ieq "py.exe") { $pythonPrefixArgs = @("-3.12") }

if ([Math]::Abs($SimDt - 0.001) -gt 1e-12 -or $Decimation -ne 50) {
    [Console]::Error.WriteLine("Formal V1.1 trials require -SimDt 0.001 and -Decimation 50.")
    exit 2
}

$allowedViewerFlags = @("--live-state-file", "--live-state-interval", "--live-state-spacing")
$extraValues = @()
$extraSeen = @{}
for ($i = 0; $i -lt $ExtraArgs.Count;) {
    $flag = $ExtraArgs[$i]
    if ($allowedViewerFlags -notcontains $flag) {
        [Console]::Error.WriteLine("Unsupported extra argument for a frozen formal trial: $flag")
        exit 2
    }
    if ($extraSeen.ContainsKey($flag)) {
        [Console]::Error.WriteLine("Duplicate viewer option: $flag")
        exit 2
    }
    if ($i + 1 -ge $ExtraArgs.Count -or $ExtraArgs[$i + 1] -like "--*") {
        [Console]::Error.WriteLine("Missing value for viewer option: $flag")
        exit 2
    }
    $value = $ExtraArgs[$i + 1]
    if ($flag -eq "--live-state-interval") {
        $intervalValue = 0
        if (-not [int]::TryParse($value, [ref]$intervalValue) -or $intervalValue -lt 1) {
            [Console]::Error.WriteLine("--live-state-interval must be a positive integer.")
            exit 2
        }
    }
    if ($flag -eq "--live-state-spacing") {
        $spacingValue = 0.0
        if (-not [double]::TryParse($value, [ref]$spacingValue) -or $spacingValue -le 0) {
            [Console]::Error.WriteLine("--live-state-spacing must be positive.")
            exit 2
        }
    }
    $extraSeen[$flag] = $true
    $extraValues += @($flag, $value)
    $i += 2
}
if ($LiveStateFile -and $extraSeen.ContainsKey("--live-state-file")) {
    [Console]::Error.WriteLine("Pass --live-state-file either with -LiveStateFile or through -ExtraArgs, not both.")
    exit 2
}
if ($LiveStateFile -and $extraSeen.ContainsKey("--live-state-interval")) {
    [Console]::Error.WriteLine("Pass --live-state-interval either with -LiveStateInterval or through -ExtraArgs, not both.")
    exit 2
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$trial = Join-Path "runs" "native_v1_1_$($Tag)_$stamp"
New-Item -ItemType Directory -Force -Path $trial | Out-Null

Write-Host "=== protocol SHA check (V1.1) ==="
& $pythonExe @pythonPrefixArgs "protocol_sha.py" --check --protocol "PROTOCOL_NATIVE_V1.1.md"
if ($LASTEXITCODE -ne 0) {
    [Console]::Error.WriteLine("PROTOCOL_SHA_MISMATCH")
    exit 2
}
Copy-Item "PROTOCOL_NATIVE_V1.1.md", "PROTOCOL_NATIVE_V1.1.sha256", "PROTOCOL_NATIVE_V1.md", "PROTOCOL_NATIVE_V1.sha256" -Destination $trial

$trainArgs = @(
    "--policy", "lstm", "--num-envs", "64", "--num-steps", "32", "--iterations", "300",
    "--seed", "0", "--threads", "8", "--device", "cuda", "--physics", "newton_xpbd",
    "--task-backend", "fast_stop_v2", "--lidar", "warp", "--feedback", "vqf",
    "--depth", "analytic", "--detections", "analytic", "--detections-warp",
    "--sim-dt", "0.001", "--decimation", "50", "--save-interval", "20", "--out", "$trial"
)
("PHYSICS sim_dt=0.001 decimation=50 task_backend=fast_stop_v2" + [Environment]::NewLine) |
    Tee-Object -FilePath (Join-Path $trial "physics.txt")
($trainArgs -join " ") | Set-Content -Path (Join-Path $trial "cli_args.txt") -Encoding utf8

Write-Host "=== formal trial train: $trial ==="
& $pythonExe @pythonPrefixArgs "-u" "train_wsl.py" @trainArgs 2>&1 |
    Tee-Object -FilePath (Join-Path $trial "train.log")
$trainExitCode = $LASTEXITCODE

$contractPath = Join-Path $trial "observation_contract.json"
$identityPath = Join-Path $trial "identity.json"
$contract = $null
$identity = $null
if (Test-Path -LiteralPath $contractPath) {
    try { $contract = Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json }
    catch { $contract = $null }
}
if (Test-Path -LiteralPath $identityPath) {
    try { $identity = Get-Content -LiteralPath $identityPath -Raw | ConvertFrom-Json }
    catch { $identity = $null }
}
$protocol = Get-Content -LiteralPath (Join-Path $trial "PROTOCOL_NATIVE_V1.1.sha256") -Raw | ConvertFrom-Json
$physicsDt = if ($contract) { $contract.physics_dt_s } else { $null }
$physicsDecimation = if ($contract) { $contract.decimation } else { $null }
$physicsExperimental = if ($contract) { $contract.physics_experimental } else { $null }
$taskBackend = if ($contract) { $contract.task_backend } else { $null }
if (-not $taskBackend -and $contract -and $contract.task) { $taskBackend = $contract.task.backend }
$taskIdentitySha = if ($contract) { $contract.task_identity_sha256 } else { $null }
$identityTaskSha = if ($identity -and $identity.env) { $identity.env.task_identity_sha256 } else { $null }
$profileEnabled = if ($contract) { $contract.profile_enabled } else { $null }
$channels = if ($contract) { $contract.channels } else { $null }
$channelsValid = [bool]($channels -and
    $channels.lidar.mode -eq "real" -and $channels.depth.mode -eq "real" -and
    $channels.detections.mode -eq "real" -and $channels.state.mode -eq "real" -and
    $channels.feedback.mode -eq "real" -and $channels.lidar.backend -eq "warp_kernel" -and
    $channels.depth.backend -eq "analytic_warp" -and
    $channels.detections.backend -eq "analytic_detector_warp" -and
    $channels.feedback.backend -eq "vqf_reference")
$physicsValid = $false
if ($null -ne $physicsDt -and $null -ne $physicsDecimation -and $null -ne $physicsExperimental) {
    $physicsValid = ([Math]::Abs([double]$physicsDt - 0.001) -le 1e-12 -and
        [int]$physicsDecimation -eq 50 -and $physicsExperimental -eq $false)
}
$formalEligible = [bool]($trainExitCode -eq 0 -and $contract -and $physicsValid -and
    $taskBackend -eq "fast_stop_v2" -and $taskIdentitySha -and
    $taskIdentitySha -eq $identityTaskSha -and $channelsValid)
$manifest = [ordered]@{
    protocol_sha256 = $protocol.protocol_sha256
    protocol_file = "PROTOCOL_NATIVE_V1.1.md"
    protocol_version = "V1.1"
    created_at = (Get-Date).ToString("o")
    cli_args_file = "cli_args.txt"
    exit_code = $trainExitCode
    observation_contract_sha256 = if ($contract) { $contract.observation_contract_sha256 } else { $null }
    capability_metrics_allowed = $false
    phase = if ($formalEligible) { "formal_native_v1_trial" } else { "invalid_formal_trial" }
    physics = [ordered]@{ sim_dt = $physicsDt; decimation = $physicsDecimation; experimental = $physicsExperimental }
    task_backend = $taskBackend
    task_identity_sha256 = $taskIdentitySha
    channels_all_real = $channelsValid
    profile_enabled = $profileEnabled
    formal_eligible = $formalEligible
    contract_file = if ($contract) { "observation_contract.json" } else { $null }
    checkpoints = @(Get-ChildItem -LiteralPath $trial -Filter "*.pt" -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Name } | Sort-Object)
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $trial "trial_manifest.json") -Encoding utf8
Write-Host ("TRIAL_MANIFEST " + ($manifest | ConvertTo-Json -Depth 8 -Compress))

$finalExitCode = $trainExitCode
if ($finalExitCode -eq 0 -and -not $formalEligible) { $finalExitCode = 3 }
Write-Host "TRIAL_DIR=$trial exit=$finalExitCode"
exit $finalExitCode
