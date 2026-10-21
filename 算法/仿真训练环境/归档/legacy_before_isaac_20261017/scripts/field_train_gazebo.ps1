[CmdletBinding()]
param(
    [string]$Distribution,
    [ValidateSet('all', 'smoke', 'train', 'benchmark')]
    [string]$Command = 'all',
    [ValidateSet('gazebo', 'fieldsim')]
    [string]$Backend = 'gazebo',
    [ValidateSet('mixed', 'search', 'goto')]
    [string]$Mode = 'mixed',
    [int]$Steps = 512,
    [int]$NSteps = 128,
    [int]$BenchmarkSteps = 128,
    [double]$PacedSeconds = 5,
    [int]$Seed = 7,
    [string]$Output,
    [string]$Checkpoint,
    [int]$MonitorPort = 0,
    [switch]$FlatGround
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$projectPath = Split-Path $PSScriptRoot -Parent
$settingsPath = Join-Path $projectPath '.local.json'
$settings = if (Test-Path -LiteralPath $settingsPath) {
    Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
} else {
    [pscustomobject]@{ distribution = 'Ubuntu-24.04' }
}
if (-not $Distribution) { $Distribution = $settings.distribution }
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { throw 'WSL is unavailable.' }
if ($Steps -le 0 -or $NSteps -le 0 -or $BenchmarkSteps -le 0 -or $PacedSeconds -le 0) {
    throw 'Training and benchmark step counts and paced seconds must be positive.'
}
if ($MonitorPort -ne 0 -and ($MonitorPort -lt 1024 -or $MonitorPort -gt 65535)) {
    throw 'MonitorPort must be 1024..65535.'
}

function ConvertTo-WslPath([string]$WindowsPath) {
    $normalized = [System.IO.Path]::GetFullPath($WindowsPath).Replace('\', '/')
    $converted = (& wsl.exe -d $Distribution --exec wslpath -u $normalized).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $converted) {
        throw "Cannot convert path for WSL: $WindowsPath"
    }
    return $converted
}

if (-not $Output) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $Output = Join-Path $projectPath "results\gazebo_field_training\run_$stamp`_$PID"
} elseif (-not [System.IO.Path]::IsPathRooted($Output)) {
    $Output = Join-Path $projectPath $Output
}
$Output = [System.IO.Path]::GetFullPath($Output)
$outputParent = Split-Path -Parent $Output
if (-not (Test-Path -LiteralPath $outputParent)) {
    New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
}
if (Test-Path -LiteralPath $Output) {
    if (@(Get-ChildItem -LiteralPath $Output -Force).Count -gt 0) {
        throw "Training output directory must be empty: $Output"
    }
} else {
    New-Item -ItemType Directory -Path $Output | Out-Null
}

$linuxProject = ConvertTo-WslPath $projectPath
$linuxOutput = ConvertTo-WslPath $Output
$linuxScript = "$linuxProject/scripts/field_train_gazebo.sh"
$linuxArgs = @($Command, '--backend', $Backend, '--mode', $Mode,
               '--steps', [string]$Steps, '--n-steps', [string]$NSteps,
               '--benchmark-steps', [string]$BenchmarkSteps,
               '--paced-seconds', [string]$PacedSeconds,
               '--seed', [string]$Seed, '--output', $linuxOutput)
if ($Checkpoint) {
    $linuxArgs += @('--checkpoint', (ConvertTo-WslPath $Checkpoint))
}
if ($MonitorPort -ne 0) { $linuxArgs += @('--monitor-port', [string]$MonitorPort) }
if ($FlatGround) { $linuxArgs += '--flat-ground' }

Write-Host "Backend: $Backend; mode: $Mode; command: $Command"
Write-Host "Output: $Output"
& wsl.exe -d $Distribution --exec bash $linuxScript @linuxArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
