<#
.SYNOPSIS
    运行 AGTC 仓库自检（离线、只读）。

.DESCRIPTION
    定位 git 与 Python 后调用 .github/scripts/repository-self-check.py。
    优先使用 PATH 中的程序；找不到时回退到本机已安装位置（含 codex 运行时自带 git）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .github/scripts/run-repository-self-check.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .github/scripts/run-repository-self-check.ps1 -Range "HEAD~1..HEAD" -SkipNative -Verbose
#>
param(
    [string]$Range = '',
    [switch]$SkipNative,
    [switch]$SkipLinks,
    [switch]$Verbose
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Find-Executable {
    param([string]$Name, [string[]]$Candidates)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    foreach ($candidate in $Candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    throw "找不到 $Name；请安装或加入 PATH。"
}

$gitCandidates = @(
    "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe",
    'C:\Program Files\Git\cmd\git.exe',
    'C:\Program Files (x86)\Git\cmd\git.exe'
)
$pythonCandidates = @(
    (Join-Path $root '算法\rnn研究\.venv\Scripts\python.exe')
)
$git = Find-Executable -Name 'git' -Candidates $gitCandidates
$python = Find-Executable -Name 'python' -Candidates $pythonCandidates

$env:PATH = (Split-Path -Parent $git) + ';' + $env:PATH

$checker = Join-Path $root '.github\scripts\repository-self-check.py'
$arguments = @('-B', $checker, '--python', $python)
if ($Range) { $arguments += @('--range', $Range) }
if ($SkipNative) { $arguments += '--skip-native' }
if ($SkipLinks) { $arguments += '--skip-links' }
if ($Verbose) { $arguments += '--verbose' }

Push-Location $root
try {
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
