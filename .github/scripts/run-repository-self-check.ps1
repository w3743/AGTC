<#
.SYNOPSIS
    运行 AGTC 仓库自检（离线、只读）。

.DESCRIPTION
    定位 git 与 Python 后调用 .github/scripts/repository-self-check.py。
    优先使用仓库内 .venv 的 Python 和 PATH 中的 git；找不到时回退到本机已安装位置
    （含 codex 运行时自带 git）。Windows 应用商店的 python 占位程序会被跳过。

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
$env:PYTHONIOENCODING = 'utf-8'

function Resolve-Git {
    $command = Get-Command git -ErrorAction SilentlyContinue
    $candidates = @()
    if ($command) { $candidates += $command.Source }
    $candidates += @(
        "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe",
        'C:\Program Files\Git\cmd\git.exe',
        'C:\Program Files (x86)\Git\cmd\git.exe'
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $version = & $candidate --version 2>&1
            if ($LASTEXITCODE -eq 0) { return $candidate }
        }
    }
    throw '找不到可用的 git，请安装 Git 或加入 PATH。'
}

function Resolve-Python {
    $candidates = @(
        (Join-Path $root '算法\rnn研究\.venv\Scripts\python.exe'),
        (Join-Path $root '.venv\Scripts\python.exe')
    )
    foreach ($name in @('python', 'python3')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { $candidates += $command.Source }
    }
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        $version = & $candidate --version 2>&1
        if ($LASTEXITCODE -eq 0) { return $candidate }
    }
    throw '找不到可用的 Python，请安装 Python 或先创建仓库内 .venv。'
}

$git = Resolve-Git
$python = Resolve-Python
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
