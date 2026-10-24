[CmdletBinding()]
param(
    [string]$TrainingRoot = 'C:\pioneer\isaac-training',
    [string]$PythonExe = 'C:\pioneer\isaac-sim-compatibility\.venv\Scripts\python.exe',
    [switch]$IncludeLegacySB3
)

$ErrorActionPreference = 'Stop'
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$fieldRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path.TrimEnd('\')
$isaacLabRoot = Join-Path $fieldRoot 'external\IsaacLab'
$expectedCommit = '5870804e4f6aee57bdef68d59f67d328b990d1e1'
$trainingRoot = [System.IO.Path]::GetFullPath($TrainingRoot).TrimEnd('\')

if ($trainingRoot.StartsWith($fieldRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'TrainingRoot must stay outside the project checkout.'
}
if (!(Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
if (!(Test-Path -LiteralPath (Join-Path $isaacLabRoot 'uv.lock') -PathType Leaf)) {
    throw "Fixed IsaacLab checkout is unavailable: $isaacLabRoot"
}

$commit = (& git -C $isaacLabRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $commit -ne $expectedCommit) {
    throw "IsaacLab checkout must remain at $expectedCommit; found '$commit'."
}
$pythonProbe = & $PythonExe -c 'import sys,sysconfig; print("{}.{}|{}".format(sys.version_info.major, sys.version_info.minor, sysconfig.get_platform()))'
if ($LASTEXITCODE -ne 0 -or ($pythonProbe -join '').Trim() -ne '3.12|win-amd64') {
    throw "Expected the existing Python 3.12 win-amd64 interpreter; found '$($pythonProbe -join '')'."
}

New-Item -ItemType Directory -Path $trainingRoot -Force | Out-Null
$logRoot = Join-Path $trainingRoot 'logs'
$cacheRoot = Join-Path $trainingRoot 'cache'
$toolVenv = Join-Path $trainingRoot 'tooling\.venv'
$toolPython = Join-Path $toolVenv 'Scripts\python.exe'
$uvExe = Join-Path $toolVenv 'Scripts\uv.exe'
$runtimeVenv = Join-Path $trainingRoot '.venv'
$runtimePython = Join-Path $runtimeVenv 'Scripts\python.exe'
New-Item -ItemType Directory -Path $logRoot,(Join-Path $cacheRoot 'uv'),(Join-Path $cacheRoot 'warp') -Force | Out-Null

function Invoke-Logged {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $logPath = Join-Path $logRoot ($Name + '.log')
    [System.IO.File]::WriteAllText($logPath, "COMMAND: $Executable $($Arguments -join ' ')`r`n", $utf8NoBom)
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    [System.IO.File]::WriteAllText(($logPath + '.exit'), [string]$exitCode, $utf8NoBom)
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. See $logPath"
    }
}

if (!(Test-Path -LiteralPath $uvExe -PathType Leaf)) {
    if (!(Test-Path -LiteralPath $toolPython -PathType Leaf)) {
        Invoke-Logged 'create-tool-venv' $PythonExe @('-m', 'venv', $toolVenv)
    }
    Invoke-Logged 'install-uv' $toolPython @('-m', 'pip', 'install', '--disable-pip-version-check', 'uv==0.13.1')
}
if (!(Test-Path -LiteralPath $runtimePython -PathType Leaf)) {
    Invoke-Logged 'create-training-venv' $uvExe @('venv', '--python', $PythonExe, '--seed', $runtimeVenv)
}

$env:UV_PROJECT_ENVIRONMENT = $runtimeVenv
$env:UV_CACHE_DIR = Join-Path $cacheRoot 'uv'
$env:UV_LINK_MODE = 'copy'
$env:WARP_CACHE_PATH = Join-Path $cacheRoot 'warp'

$rawRequirements = Join-Path $trainingRoot 'isaaclab-windows-x64-requirements.txt'
$adaptedRequirements = Join-Path $trainingRoot 'isaaclab-windows-arm64-x64-emulation-requirements.txt'
$compatRequirements = Join-Path $trainingRoot 'isaacsim-6.2-windows-x64-compat-requirements.txt'
$exportArgs = @(
    'export', '--project', $isaacLabRoot, '--python', $runtimePython, '--locked',
    '--no-default-groups', '--extra', 'rsl-rl', '--extra', 'viser', '--extra', 'isaacsim', '--extra', 'test',
    '--format', 'requirements.txt', '--output-file', $rawRequirements
)
if ($IncludeLegacySB3) {
    $exportArgs += @('--extra', 'sb3')
}
Invoke-Logged 'isaaclab-locked-export' $uvExe $exportArgs

$rawText = [System.IO.File]::ReadAllText($rawRequirements)
$windowsX64Marker = "platform_machine == 'AMD64' and sys_platform == 'win32'"
if (!$rawText.Contains($windowsX64Marker)) {
    throw "Expected Windows x64 markers were not found in $rawRequirements; refusing an unreviewed platform adaptation."
}
$adaptedText = $rawText.Replace($windowsX64Marker, "platform_machine == 'ARM64' and sys_platform == 'win32'")
[System.IO.File]::WriteAllText($adaptedRequirements, $adaptedText, $utf8NoBom)
$adapterRecord = [ordered]@{
    upstream_commit = $expectedCommit
    upstream_lock = (Join-Path $isaacLabRoot 'uv.lock')
    raw_requirements_sha256 = (Get-FileHash -LiteralPath $rawRequirements -Algorithm SHA256).Hash.ToLowerInvariant()
    adapted_requirements_sha256 = (Get-FileHash -LiteralPath $adaptedRequirements -Algorithm SHA256).Hash.ToLowerInvariant()
    change = 'In the exported requirements copy only, map the upstream Windows AMD64 environment marker to this Windows ARM64 host marker. The interpreter reports win-amd64 wheel ABI; upstream source and uv.lock remain unchanged.'
    torch_cuda_runtime_probed = $false
} | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText((Join-Path $logRoot 'isaaclab-platform-adaptation.json'), $adapterRecord, $utf8NoBom)

$syncArgs = @('pip', 'sync', '--python', $runtimePython, '--index-strategy', 'unsafe-best-match', $adaptedRequirements)
Push-Location $isaacLabRoot
try {
    Invoke-Logged 'isaaclab-locked-sync' $uvExe $syncArgs
} finally {
    Pop-Location
}

$compatPins = @(
    '# Local compatibility overlay for Isaac Sim 6.2.0.0 Windows x64-emulation metadata.',
    '# Restore four Isaac Sim exact pins, its packaging upper bound, and the GPU ONNX runtime omitted by the upstream export.',
    'numpy==2.3.1',
    'fsspec==2026.2.0',
    'pywin32==306',
    'urllib3==2.7.0',
    'packaging==26.0',
    'onnxruntime-gpu==1.26.0'
)
[System.IO.File]::WriteAllLines($compatRequirements, $compatPins, $utf8NoBom)
Invoke-Logged 'isaacsim-compatibility-overlay' $uvExe @(
    'pip', 'install', '--python', $runtimePython, '--reinstall', '-r', $compatRequirements
)
$compatRecord = [ordered]@{
    isaacsim_version = '6.2.0.0'
    isaaclab_commit = $expectedCommit
    requirements_file = $compatRequirements
    requirements_sha256 = (Get-FileHash -LiteralPath $compatRequirements -Algorithm SHA256).Hash.ToLowerInvariant()
    pins = [ordered]@{
        numpy = '2.3.1'
        fsspec = '2026.2.0'
        pywin32 = '306'
        urllib3 = '2.7.0'
        packaging = '26.0'
        onnxruntime_gpu = '1.26.0'
    }
    reason = 'The installed Isaac Sim 6.2.0.0 Windows wheel metadata has exact requirements that differ from four versions selected by the fixed IsaacLab uv.lock. This explicit local overlay keeps those vendor requirements satisfiable; upstream files and the original lock export are preserved.'
    upstream_lock_modified = $false
    gpu_probed = $false
} | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText((Join-Path $logRoot 'isaacsim-compatibility-overlay.json'), $compatRecord, $utf8NoBom)

Invoke-Logged 'project-runtime-install' $uvExe @(
    'pip', 'install', '--python', $runtimePython, '--no-deps',
    '--editable', $fieldRoot, 'vqf==2.1.2'
)
if ($IncludeLegacySB3) {
    Invoke-Logged 'legacy-sb3-install' $uvExe @(
        'pip', 'install', '--python', $runtimePython, '--no-deps', 'sb3-contrib==2.9.0'
    )
}

Invoke-Logged 'pip-check' $runtimePython @('-m', 'pip', 'check')
$verifyCode = @'
import importlib.metadata as metadata
import json
import platform
import sys
import sysconfig
import torch

packages = {
    "isaacsim": metadata.version("isaacsim"),
    "isaaclab": metadata.version("isaaclab"),
    "rsl-rl-lib": metadata.version("rsl-rl-lib"),
    "viser": metadata.version("viser"),
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "torchvision": metadata.version("torchvision"),
    "warp-lang": metadata.version("warp-lang"),
    "newton": metadata.version("newton"),
    "vqf": metadata.version("vqf"),
    "onnxruntime-gpu": metadata.version("onnxruntime-gpu"),
    "numpy": metadata.version("numpy"),
    "fsspec": metadata.version("fsspec"),
    "pywin32": metadata.version("pywin32"),
    "urllib3": metadata.version("urllib3"),
    "packaging": metadata.version("packaging"),
    "pytest": metadata.version("pytest"),
}
assert packages["isaacsim"] == "6.2.0.0", packages
assert packages["rsl-rl-lib"] == "5.5.1", packages
assert packages["torch"].endswith("+cu130"), packages
assert packages["torch_cuda_build"] == "13.0", packages
assert packages["torchvision"].endswith("+cu130"), packages
assert packages["warp-lang"] == "1.17.0", packages
assert packages["newton"] == "1.6.1", packages
assert packages["vqf"] == "2.1.2", packages
assert packages["onnxruntime-gpu"] == "1.26.0", packages
assert packages["numpy"] == "2.3.1", packages
assert packages["fsspec"] == "2026.2.0", packages
assert packages["pywin32"] == "306", packages
assert packages["urllib3"] == "2.7.0", packages
assert packages["packaging"] == "26.0", packages
print(json.dumps({
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "python_platform_tag": sysconfig.get_platform(),
    "os_machine": platform.machine(),
    "gpu_probed": False,
    "packages": packages,
}, indent=2))
'@
Invoke-Logged 'runtime-metadata-check' $runtimePython @('-c', $verifyCode)

$status = [ordered]@{
    status = 'ready'
    python = $runtimePython
    isaaclab_root = $isaacLabRoot
    isaaclab_commit = $expectedCommit
    include_legacy_sb3 = [bool]$IncludeLegacySB3
    isaacsim_compatibility_overlay = (Join-Path $logRoot 'isaacsim-compatibility-overlay.json')
    pip_check_exit = 0
    torch_cuda_runtime_probed = $false
    metadata_log = (Join-Path $logRoot 'runtime-metadata-check.log')
    installed_utc = [DateTime]::UtcNow.ToString('o')
} | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText((Join-Path $trainingRoot 'install_status.json'), $status, $utf8NoBom)
Write-Host "Isaac Lab environment ready: $runtimePython"
