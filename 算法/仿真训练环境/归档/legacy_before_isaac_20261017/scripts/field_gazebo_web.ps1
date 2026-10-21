[CmdletBinding()]
param([int]$Port=8768, [int]$Seed=7, [string]$Checkpoint)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path $PSScriptRoot -Parent
$settings = Get-Content -LiteralPath (Join-Path $projectPath '.local.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$linuxRoot = (& wsl.exe -d $settings.distribution --exec wslpath -u $projectPath.Replace('\','/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Unable to resolve the WSL project path.' }
$webArgs = @('--port', [string]$Port, '--seed', [string]$Seed)
if ($Checkpoint) {
    $checkpointPath = [System.IO.Path]::GetFullPath($Checkpoint)
    $linuxCheckpoint = (& wsl.exe -d $settings.distribution --exec wslpath -u $checkpointPath.Replace('\','/')).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Unable to resolve the checkpoint path.' }
    $webArgs += @('--checkpoint', $linuxCheckpoint)
}
Write-Host "Gazebo browser: http://127.0.0.1:$Port/ (initial scene loading may take a moment)"
& wsl.exe -d $settings.distribution --exec bash "$linuxRoot/scripts/field_gazebo_web.sh" @webArgs
exit $LASTEXITCODE
