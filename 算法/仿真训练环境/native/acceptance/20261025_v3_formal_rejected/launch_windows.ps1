param(
    [switch]$Start,
    [string]$RootReleaseNotifiedAt = ''
)
$ErrorActionPreference = 'Stop'
$taskBase = 'C:\pioneer\agtc-native-acceptance-backup\fixed_physics_v3_prep_20261025_184052'
$taskLaunchWsl = '/mnt/c/pioneer/agtc-native-acceptance-backup/fixed_physics_v3_prep_20261025_184052/launch.py'
$taskDistro = 'Ubuntu-24.04-Arm64'
if (-not $Start) {
    Write-Output 'PREPARED_ONLY: no WSL process or GPU work started. Pass -Start after explicit root GPU-release notice.'
    exit 0
}
if ([string]::IsNullOrWhiteSpace($RootReleaseNotifiedAt)) { throw 'RootReleaseNotifiedAt is required when Start is specified.' }
$taskRequest = [ordered]@{ root_release_notified_at = $RootReleaseNotifiedAt; tag = 'fixed_physics_v3_20261025_184052'; start_requested_at_utc = [DateTimeOffset]::UtcNow.ToString('o'); start_requested_at_beijing = [DateTimeOffset]::Now.ToString('o') }
[System.IO.File]::WriteAllText((Join-Path $taskBase 'windows_launch_request.json'), ($taskRequest | ConvertTo-Json -Depth 4) + "`n", [System.Text.UTF8Encoding]::new($false))
$taskApp = Join-Path $env:SystemRoot 'System32\wsl.exe'
$taskStdout = Join-Path $taskBase 'launch_controller.stdout.rawlog'
$taskStderr = Join-Path $taskBase 'launch_controller.stderr.rawlog'
$taskArgs = @('-d', $taskDistro, '--', '/usr/bin/env', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', '/opt/venv/bin/python', '-B', $taskLaunchWsl, '--root-release-notified-at', $RootReleaseNotifiedAt)
$taskProc = Start-Process -FilePath $taskApp -ArgumentList $taskArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr
$taskProcState = Get-Process -Id $taskProc.Id -ErrorAction SilentlyContinue
$taskWinRecord = [ordered]@{ windows_pid = $taskProc.Id; process_name = 'wsl.exe'; observed_alive_after_start = ($null -ne $taskProcState); started_at_utc = [DateTimeOffset]::UtcNow.ToString('o'); started_at_beijing = [DateTimeOffset]::Now.ToString('o'); root_release_notified_at = $RootReleaseNotifiedAt; tag = 'fixed_physics_v3_20261025_184052'; arguments = $taskArgs }
[System.IO.File]::WriteAllText((Join-Path $taskBase 'windows_launcher.json'), ($taskWinRecord | ConvertTo-Json -Depth 5) + "`n", [System.Text.UTF8Encoding]::new($false))
$taskWinRecord | ConvertTo-Json -Depth 5