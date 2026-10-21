[CmdletBinding()]
param(
    [string]$Distribution,
    [int]$Seconds = 10,
    [int]$Seed = 7,
    [string]$SceneJson,
    [string]$StateJson,
    [string]$Output,
    [switch]$Gui,
    [switch]$Unlimited,
    [switch]$NoMotion,
    [switch]$WithUwb
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

$fieldPython = Join-Path $projectPath '.venv-field\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $fieldPython)) {
    throw "Field simulation Python environment not found: $fieldPython"
}
if ($Seconds -le 0) { throw '-Seconds must be positive.' }
if ([bool]$SceneJson -ne [bool]$StateJson) {
    throw 'Pass both -SceneJson and -StateJson, or pass neither to materialize a seed-based scene.'
}

$runRoot = Join-Path $projectPath 'results\gazebo_field_runs'
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$runName = '{0}_{1}' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'), $PID
$runDir = Join-Path $runRoot $runName
if (Test-Path -LiteralPath $runDir) { throw "Run directory already exists: $runDir" }
New-Item -ItemType Directory -Path $runDir | Out-Null
$inputDir = Join-Path $runDir 'input'
$worldDir = Join-Path $runDir 'world'
$worldPath = Join-Path $worldDir 'world.sdf'

Push-Location $projectPath
try {
    if ($SceneJson) {
        $scenePath = [System.IO.Path]::GetFullPath($SceneJson)
        $statePath = [System.IO.Path]::GetFullPath($StateJson)
        if (-not (Test-Path -LiteralPath $scenePath) -or -not (Test-Path -LiteralPath $statePath)) {
            throw 'Both -SceneJson and -StateJson must point to existing files.'
        }
    } else {
        & $fieldPython -m simulation.export_field_payload --output-dir $inputDir --seed $Seed
        if ($LASTEXITCODE -ne 0) { throw "Field scene materialization failed (exit $LASTEXITCODE)." }
        $scenePath = Join-Path $inputDir 'scene.json'
        $statePath = Join-Path $inputDir 'state.json'
    }

    $exportArgs = @(
        '-m', 'simulation.field_world',
        '--scene-json', $scenePath,
        '--state-json', $statePath,
        '--output', $worldDir,
        '--mass', '30',
        '--com-height', '0.15'
    )
    if ($WithUwb) { $exportArgs += '--with-uwb' }
    & $fieldPython @exportArgs
    if ($LASTEXITCODE -ne 0) { throw "Gazebo field export failed (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}

$telemetryPath = if ($Output) {
    [System.IO.Path]::GetFullPath($Output)
} else {
    Join-Path $runDir 'telemetry'
}
if (Test-Path -LiteralPath $telemetryPath) {
    if (@(Get-ChildItem -LiteralPath $telemetryPath -Force).Count -gt 0) {
        throw "Telemetry output must be empty: $telemetryPath"
    }
} else {
    $telemetryParent = Split-Path -Parent $telemetryPath
    if (-not (Test-Path -LiteralPath $telemetryParent)) {
        New-Item -ItemType Directory -Force -Path $telemetryParent | Out-Null
    }
    New-Item -ItemType Directory -Path $telemetryPath | Out-Null
}

function ConvertTo-WslPath([string]$WindowsPath) {
    $normalized = [System.IO.Path]::GetFullPath($WindowsPath).Replace('\', '/')
    $converted = (& wsl.exe -d $Distribution --exec wslpath -u $normalized).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $converted) {
        throw "Cannot convert path for WSL: $WindowsPath"
    }
    return $converted
}

$linuxProject = ConvertTo-WslPath $projectPath
$linuxWorld = ConvertTo-WslPath $worldPath
$linuxOutput = ConvertTo-WslPath $telemetryPath
$linuxScript = "$linuxProject/scripts/field_gazebo.sh"
$linuxArgs = @('--world', $linuxWorld, '--output', $linuxOutput,
               '--seconds', [string]$Seconds)
if ($Gui) { $linuxArgs += '--gui' }
if ($Unlimited) { $linuxArgs += '--unlimited' }
if ($NoMotion) { $linuxArgs += '--no-motion' }

Write-Host "World: $worldPath"
Write-Host "Telemetry: $telemetryPath"
& wsl.exe -d $Distribution --exec bash $linuxScript @linuxArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
