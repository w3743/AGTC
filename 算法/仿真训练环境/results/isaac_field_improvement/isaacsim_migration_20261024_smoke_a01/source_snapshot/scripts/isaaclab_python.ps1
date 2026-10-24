[CmdletBinding()]
param(
    [ValidateSet('viewer', 'module', 'script', 'python')]
    [string]$Mode = 'viewer',
    [string]$Module = 'isaac_field',
    [string]$Script = '',
    [string[]]$Arguments = @(),
    [string]$TrainingRoot = 'C:\pioneer\isaac-training'
)

$ErrorActionPreference = 'Stop'
$fieldRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path.TrimEnd('\')
$isaacLabRoot = Join-Path $fieldRoot 'external\IsaacLab'
$trainingRoot = [System.IO.Path]::GetFullPath($TrainingRoot).TrimEnd('\')
$runtimePython = Join-Path $trainingRoot '.venv\Scripts\python.exe'
$statusPath = Join-Path $trainingRoot 'install_status.json'

if (!(Test-Path -LiteralPath $runtimePython -PathType Leaf) -or !(Test-Path -LiteralPath $statusPath -PathType Leaf)) {
    throw "Isaac Lab runtime is not installed. Run scripts\install_isaaclab_windows.ps1 first."
}
$status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
if ($status.status -ne 'ready' -or $status.isaaclab_commit -ne '5870804e4f6aee57bdef68d59f67d328b990d1e1') {
    throw "Training environment status does not match the fixed IsaacLab checkout: $statusPath"
}

$env:ISAACLAB_ROOT = $isaacLabRoot
$env:WARP_CACHE_PATH = Join-Path $trainingRoot 'cache\warp'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = $fieldRoot + ';' + $env:PYTHONPATH
} else {
    $env:PYTHONPATH = $fieldRoot
}

switch ($Mode) {
    'viewer' {
        $pythonArguments = @('-m', 'isaac_field', 'viewer') + $Arguments
    }
    'module' {
        if ($Module -notmatch '^[A-Za-z_][A-Za-z0-9_.]*$') {
            throw "Invalid Python module name: $Module"
        }
        $pythonArguments = @('-m', $Module) + $Arguments
    }
    'script' {
        if ([string]::IsNullOrWhiteSpace($Script)) {
            throw 'Mode script requires -Script with a project-relative Python file.'
        }
        $scriptPath = [System.IO.Path]::GetFullPath((Join-Path $fieldRoot $Script))
        $projectPrefix = $fieldRoot + '\'
        if (!$scriptPath.StartsWith($projectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'Python script must remain inside the project checkout.'
        }
        if (!(Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
            throw "Python script not found: $scriptPath"
        }
        $pythonArguments = @($scriptPath) + $Arguments
    }
    'python' {
        $pythonArguments = @($Arguments)
    }
}

& $runtimePython @pythonArguments
$exitCode = $LASTEXITCODE
exit $exitCode
