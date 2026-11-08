#Requires -Version 5.1
<#
.SYNOPSIS
  Windows-native launcher for the frozen Native V2 recipe.

.DESCRIPTION
  Mirrors run_native_v2_trial.sh without importing Isaac/Kit:
  - verifies the frozen protocol SHA, official physics and dependency versions;
  - writes the same trial evidence (protocol copy, cli_args.txt, physics.txt);
  - runs train_wsl.py with the fixed 64 env x 32 steps x 300 iterations recipe;
  - keeps the process alive under a VRAM watchdog. If the GPU is under memory
    pressure and the trainer has not produced a new iteration for more than
    StallSeconds, or the process is unresponsive for HardStallSeconds, the
    launcher preserves the last atomic checkpoint/logs/metrics in
    watchdog_preserved/ and terminates the trainer (exit 90);
  - after a clean exit it runs trial_checks.py write-manifest.

  This file is a Windows launcher only. It does not modify the frozen trainer,
  protocol, reward or task sources.

.EXAMPLE
  .\run_native_v2_trial.ps1 -Tag win01 -Python D:\Apps\DSHproject\Rk3588\agtc-native\venv\Scripts\python.exe
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Python = "",
    [string]$RuntimeRoot = 'D:\Apps\DSHproject\Rk3588\agtc-native',
    [double]$VramPercentLimit = 90.0,
    [int]$StallSeconds = 60,
    [int]$HardStallSeconds = 180,
    [int]$StartupGraceSeconds = 900,
    [string]$LiveStateFile = "",
    [int]$LiveStateInterval = 5,
    [ValidateSet('cuda', 'cpu')][string]$PolicyDevice = 'cuda',
    [switch]$Smoke
)

$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$TrainDir = $PSScriptRoot
$NativeRoot = Split-Path -Parent $TrainDir
$FieldRoot = Split-Path -Parent $NativeRoot
$ProtocolFile = Join-Path $TrainDir 'PROTOCOL_NATIVE_V2.md'
$ProtocolShaFile = Join-Path $TrainDir 'PROTOCOL_NATIVE_V2.sha256'
$TrainScript = Join-Path $TrainDir 'train_wsl.py'
$Wrapper = Join-Path $TrainDir 'run_train_windows.py'
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $RuntimeRoot 'venv\Scripts\python.exe'
}
foreach ($required in @($Python, $ProtocolFile, $ProtocolShaFile, $TrainScript, $Wrapper)) {
    if (!(Test-Path -LiteralPath $required -PathType Leaf)) {
        [Console]::Error.WriteLine("ERROR: required file not found: $required")
        exit 2
    }
}

$cacheRoot = Join-Path $RuntimeRoot 'cache'
$tempRoot = Join-Path $cacheRoot 'tmp'
$runsRoot = Join-Path $TrainDir 'runs'
foreach ($dir in @($cacheRoot, $tempRoot, (Join-Path $cacheRoot 'warp'),
                   (Join-Path $cacheRoot 'cuda'), (Join-Path $cacheRoot 'torch'),
                   (Join-Path $cacheRoot 'pip'), (Join-Path $cacheRoot 'matplotlib'),
                   $runsRoot)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# All downloads, compiler caches and temporary files stay on the runtime drive.
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:TMPDIR = $tempRoot
$env:PIP_CACHE_DIR = Join-Path $cacheRoot 'pip'
$env:WARP_CACHE_PATH = Join-Path $cacheRoot 'warp'
# Warp mempool + mixed native sensor kernels crashes with 0xC0000005 on this
# Windows x64 / sm_120 host; the default allocator is stable. The mempool is
# not part of the task identity, so the frozen contract SHA is unchanged.
$env:WARP_ENABLE_MEMPOOLS_AT_INIT = '0'
$env:CUDA_CACHE_PATH = Join-Path $cacheRoot 'cuda'
$env:TORCH_HOME = Join-Path $cacheRoot 'torch'
$env:MPLCONFIGDIR = Join-Path $cacheRoot 'matplotlib'
$env:XDG_CACHE_HOME = $cacheRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = $FieldRoot
# Windows x64 / sm_120 workaround: disable cuDNN's internal workspaces/streams
# and Warp's mempool; the frozen task identity is unchanged.
$env:AGTC_DISABLE_CUDNN = '1'
$env:AGTC_SHARED_STREAM = '0'
$env:AGTC_SYNC_CUDA = '0'
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
Remove-Item Env:CUDA_MODULE_LOADING -ErrorAction SilentlyContinue
Remove-Item Env:CUDA_LAUNCH_BLOCKING -ErrorAction SilentlyContinue
Remove-Item Env:CUDA_DEVICE_MAX_CONNECTIONS -ErrorAction SilentlyContinue
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$trial = Join-Path $runsRoot ("native_v2_windows_{0}_{1}" -f $Tag, $stamp)
New-Item -ItemType Directory -Force -Path $trial | Out-Null
$launcherLog = Join-Path $trial 'launcher.log'
Start-Transcript -Path $launcherLog -Force | Out-Null

function Write-Phase([string]$message) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'o'), $message
    Write-Host $line
}

function Invoke-PythonChecked([string[]]$arguments, [string]$label) {
    & $Python @arguments
    $rc = $LASTEXITCODE
    if ($rc -ne 0) {
        [Console]::Error.WriteLine("ERROR: $label failed with exit code $rc")
        return $false
    }
    return $true
}

$exitCode = 2
$watchdogTriggered = $false
$watchdogReason = $null
try {
    Write-Phase "trial=$trial"
    Write-Phase "python=$Python"
    Write-Phase "field_root=$FieldRoot"
    Write-Phase "policy_device=$PolicyDevice"

    $versionCode = @'
from importlib.metadata import version
expected = {"newton": "1.7.0", "warp-lang": "1.18.0", "mujoco": "3.15.0", "mujoco-warp": "3.15.0"}
actual = {key: version(key) for key in expected}
print(actual)
assert actual == expected, "native V2 dependency versions differ"
'@
    $versionFile = Join-Path $trial 'dependency_check.py'
    Set-Content -LiteralPath $versionFile -Value $versionCode -Encoding utf8
    if (!(Invoke-PythonChecked @($versionFile) 'dependency version check')) { exit 2 }

    Write-Phase '=== protocol SHA check (V2) ==='
    if (!(Invoke-PythonChecked @((Join-Path $TrainDir 'protocol_sha.py'), '--protocol', 'PROTOCOL_NATIVE_V2.md', '--check') 'protocol_sha.py')) { exit 2 }
    if (!(Invoke-PythonChecked @((Join-Path $TrainDir 'trial_checks.py'), 'check-protocol', '--protocol-version', 'V2', '--protocol-sha', $ProtocolShaFile) 'check-protocol')) { exit 2 }
    if (!(Invoke-PythonChecked @((Join-Path $TrainDir 'trial_checks.py'), 'require-official-physics', '--sim-dt', '0.001', '--decimation', '50') 'require-official-physics')) { exit 2 }

    Copy-Item -LiteralPath $ProtocolFile -Destination (Join-Path $trial 'PROTOCOL_NATIVE_V2.md') -Force
    Copy-Item -LiteralPath $ProtocolShaFile -Destination (Join-Path $trial 'PROTOCOL_NATIVE_V2.sha256') -Force
    Set-Content -LiteralPath (Join-Path $trial 'physics.txt') -Value 'PHYSICS sim_dt=0.001 decimation=50' -Encoding utf8

    $iterations = 300
    $phaseName = 'formal'
    if ($Smoke) {
        $iterations = 1
        $phaseName = 'smoke'
    }
    $trainArgs = @(
        '-u', $Wrapper,
        '--policy', 'lstm', '--lstm-hidden', '512',
        '--num-envs', '64', '--num-steps', '32', '--iterations', "$iterations",
        '--seed', '0', '--threads', '8', '--epochs', '4', '--lr', '0.0003',
        '--device', 'cuda', '--policy-device', $PolicyDevice,
        '--physics', 'newton_mujoco_warp', '--task-backend', 'fast_stop_v2',
        '--lidar', 'warp', '--feedback', 'vqf', '--depth', 'analytic',
        '--detections', 'analytic', '--detections-warp',
        '--sim-dt', '0.001', '--decimation', '50', '--save-interval', '20',
        '--nonfinite-limit', '0', '--out', $trial
    )
    if (![string]::IsNullOrWhiteSpace($LiveStateFile)) {
        $trainArgs += @('--live-state-file', $LiveStateFile,
                        '--live-state-interval', "$LiveStateInterval")
    }
    $cliText = ($trainArgs | ForEach-Object {
        if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' '
    Set-Content -LiteralPath (Join-Path $trial 'cli_args.txt') -Value $cliText -Encoding utf8

    # Detailed source snapshot (raw bytes + SHA256) for the run evidence.
    $snapshot = Join-Path $trial 'source_snapshot'
    New-Item -ItemType Directory -Force -Path $snapshot | Out-Null
    $sourceFiles = @(
        (Join-Path $TrainDir 'train_wsl.py'), (Join-Path $TrainDir 'run_train_windows.py'),
        (Join-Path $TrainDir 'trial_checks.py'),
        (Join-Path $TrainDir 'contract_identity.py'), (Join-Path $TrainDir 'policy_lstm.py'),
        (Join-Path $TrainDir 'protocol_sha.py'), (Join-Path $TrainDir 'PROTOCOL_NATIVE_V2.md'),
        (Join-Path $TrainDir 'PROTOCOL_NATIVE_V2.sha256'),
        (Join-Path $NativeRoot 'requirements-native-v2.txt'),
        (Join-Path $NativeRoot 'proto\native_env.py'),
        (Join-Path $NativeRoot 'proto\observation_provider.py'),
        (Join-Path $NativeRoot 'proto\newton_backend.py'),
        (Join-Path $NativeRoot 'proto\newton_mujoco_backend.py'),
        (Join-Path $NativeRoot 'proto\physics_backend.py'),
        (Join-Path $NativeRoot 'proto\reward_fast_stop.py'),
        (Join-Path $NativeRoot 'proto\depth_caster.py'),
        (Join-Path $NativeRoot 'proto\detections_caster.py'),
        (Join-Path $NativeRoot 'proto\train_m0.py'),
        (Join-Path $NativeRoot 'proto\policy_m0.py')
    )
    $hashRows = @()
    foreach ($source in $sourceFiles) {
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            $rel = $source.Substring($NativeRoot.Length).TrimStart('\')
            $target = Join-Path $snapshot $rel
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
            Copy-Item -LiteralPath $source -Destination $target -Force
            $hashRows += [ordered]@{
                path = $rel
                sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
                size_bytes = (Get-Item -LiteralPath $source).Length
            }
        }
    }
    try {
        $gitCommit = (& git -C (Split-Path -Parent $NativeRoot) rev-parse HEAD 2>$null)
        if ($LASTEXITCODE -ne 0) { $gitCommit = 'unavailable' }
    } catch { $gitCommit = 'unavailable' }
    $sourceManifest = [ordered]@{
        created_at = (Get-Date).ToString('o')
        repo_root = (Split-Path -Parent $NativeRoot)
        git_commit = ("$gitCommit").Trim()
        wrapper = [ordered]@{
            path = $PSCommandPath
            sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        files = $hashRows
    }
    $sourceManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $trial 'source_manifest.json') -Encoding utf8

    $outFile = Join-Path $trial 'train.out'
    $errFile = Join-Path $trial 'train.err'
    $gpuCsv = Join-Path $trial 'gpu_samples.csv'
    'timestamp_utc,memory_used_mib,memory_total_mib,memory_pct,train_out_bytes,metrics_bytes,process_cpu_s' |
        Set-Content -LiteralPath $gpuCsv -Encoding utf8

    if (Test-Path -LiteralPath $outFile) { Remove-Item -LiteralPath $outFile -Force }
    if (Test-Path -LiteralPath $errFile) { Remove-Item -LiteralPath $errFile -Force }

    Write-Phase ("=== native V2 {0} train: 64 env x 32 steps x {1} iterations ===" -f $phaseName, $iterations)
    $startTime = Get-Date
    $process = Start-Process -FilePath $Python -ArgumentList $trainArgs -WorkingDirectory $TrainDir `
        -NoNewWindow -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile

    $metricsPath = Join-Path $trial 'metrics.jsonl'
    $lastOutBytes = 0L
    $lastMetricsBytes = 0L
    $lastProgress = $startTime
    $seenProgress = $false
    $lastIterSeen = ""
    $gpuAvailable = $null -ne (Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue)

    while (-not $process.HasExited) {
        Start-Sleep -Seconds 2
        $now = Get-Date
        $outBytes = 0L
        if (Test-Path -LiteralPath $outFile) { $outBytes = (Get-Item -LiteralPath $outFile).Length }
        $metricsBytes = 0L
        if (Test-Path -LiteralPath $metricsPath) { $metricsBytes = (Get-Item -LiteralPath $metricsPath).Length }
        if ($outBytes -gt $lastOutBytes -or $metricsBytes -gt $lastMetricsBytes) {
            $lastOutBytes = $outBytes
            $lastMetricsBytes = $metricsBytes
            $lastProgress = $now
            if ($metricsBytes -gt 0) { $seenProgress = $true }
        }
        if (-not $seenProgress -and $outBytes -gt 0) {
            $tail = Get-Content -LiteralPath $outFile -Tail 20 -ErrorAction SilentlyContinue
            if ($tail -match '^ITER ') {
                $seenProgress = $true
                $lastProgress = $now
                $lastIterSeen = ($tail | Where-Object { $_ -match '^ITER ' } | Select-Object -Last 1)
            }
        }

        $used = 0.0
        $total = 0.0
        $pct = 0.0
        if ($gpuAvailable) {
            try {
                $gpuLine = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
                if ($gpuLine) {
                    $parts = $gpuLine -split '\s*,\s*'
                    if ($parts.Count -ge 2) {
                        $used = [double]::Parse(($parts[0] -replace '[^0-9\.]', ''))
                        $total = [double]::Parse(($parts[1] -replace '[^0-9\.]', ''))
                        if ($total -gt 0) { $pct = 100.0 * $used / $total }
                    }
                }
            } catch { }
        }
        $cpu = ''
        try { $cpu = [string](Get-Process -Id $process.Id -ErrorAction Stop).CPU } catch { }

        $csvRow = "{0},{1},{2},{3},{4},{5},{6}" -f `
            $now.ToUniversalTime().ToString('o'), [math]::Round($used, 1), [math]::Round($total, 1),
            [math]::Round($pct, 2), $outBytes, $metricsBytes, $cpu
        Add-Content -LiteralPath $gpuCsv -Value $csvRow

        $elapsed = ($now - $startTime).TotalSeconds
        if ($seenProgress -or $elapsed -ge $StartupGraceSeconds) {
            $stall = ($now - $lastProgress).TotalSeconds
            if ($pct -ge $VramPercentLimit -and $stall -ge $StallSeconds) {
                $watchdogTriggered = $true
                $watchdogReason = "vram_pressure_stall: gpu={0:F2}% stall={1:F1}s" -f $pct, $stall
            } elseif ($stall -ge $HardStallSeconds) {
                $watchdogTriggered = $true
                $watchdogReason = "unresponsive_stall: stall={0:F1}s gpu={1:F2}%" -f $stall, $pct
            }
        }
        if ($watchdogTriggered) { break }
    }

    $endTime = Get-Date
    $preserved = Join-Path $trial 'watchdog_preserved'
    if ($watchdogTriggered) {
        Write-Phase "WATCHDOG $watchdogReason"
        New-Item -ItemType Directory -Force -Path $preserved | Out-Null
        foreach ($pattern in @('latest.pt', 'checkpoint_*.pt', 'metrics.jsonl',
                              'train.out', 'train.err', 'identity.json',
                              'observation_contract.json', 'failure.json')) {
            Get-ChildItem -LiteralPath $trial -Filter $pattern -File -ErrorAction SilentlyContinue |
                ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $preserved -Force }
        }
        $lastMetrics = ""
        if (Test-Path -LiteralPath $metricsPath) {
            $lastMetrics = (Get-Content -LiteralPath $metricsPath -Tail 1 -ErrorAction SilentlyContinue)
        }
        $record = [ordered]@{
            triggered_at = $endTime.ToString('o')
            reason = $watchdogReason
            vram_percent_limit = $VramPercentLimit
            stall_seconds = $StallSeconds
            hard_stall_seconds = $HardStallSeconds
            gpu_memory_percent = $pct
            last_iter_line = $lastIterSeen
            last_metrics_line = $lastMetrics
            preserved = $preserved
            note = 'The frozen trainer cannot be checkpointed externally; the last atomic latest.pt/checkpoint_*.pt and logs are preserved before termination.'
        }
        $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $preserved 'watchdog.json') -Encoding utf8
        try { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue } catch { }
        Start-Sleep -Seconds 5
        try { $process.Refresh() } catch { }
    }

    $processExit = $null
    $exitInferred = $false
    try {
        $process.WaitForExit()
        $process.Refresh()
        if ($process.HasExited) { $processExit = $process.ExitCode }
    } catch { }
    if ($null -eq $processExit -and (Test-Path -LiteralPath (Join-Path $trial 'summary.json'))) {
        # The venv launcher occasionally loses ExitCode while the trainer has
        # already written summary.json; use it only as a defensive fallback.
        $processExit = 0
        $exitInferred = $true
    }
    $combined = Join-Path $trial 'train.log'
    if (Test-Path -LiteralPath $outFile) { Get-Content -LiteralPath $outFile | Set-Content -LiteralPath $combined -Encoding utf8 }
    if (Test-Path -LiteralPath $errFile) { Get-Content -LiteralPath $errFile | Add-Content -LiteralPath $combined -Encoding utf8 }

    $exitRecord = [ordered]@{
        exit_code = $processExit
        started_at = $startTime.ToString('o')
        finished_at = $endTime.ToString('o')
        duration_s = [math]::Round(($endTime - $startTime).TotalSeconds, 3)
        watchdog_triggered = $watchdogTriggered
        watchdog_reason = $watchdogReason
        exit_inferred_from_summary = $exitInferred
        trial = $trial
    }
    $exitRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $trial 'process_exit.json') -Encoding utf8

    if ($watchdogTriggered) {
        $exitCode = 90
    } elseif ($processExit -ne 0) {
        $exitCode = if ($null -eq $processExit) { 1 } else { [int]$processExit }
    } elseif ($Smoke) {
        Write-Phase "TRIAL_DIR=$trial smoke_exit=$processExit (manifest skipped)"
        $exitCode = 0
    } else {
        Write-Phase '=== trial_checks write-manifest (V2) ==='
        & $Python (Join-Path $TrainDir 'trial_checks.py') write-manifest `
            --trial-dir $trial --train-exit $processExit `
            --protocol-sha $ProtocolShaFile --protocol-version V2
        $manifestRc = $LASTEXITCODE
        Write-Phase "TRIAL_DIR=$trial train_exit=$processExit manifest_check=$manifestRc"
        $exitCode = [int]$manifestRc
    }
} finally {
    try { Stop-Transcript | Out-Null } catch { }
}

exit $exitCode
