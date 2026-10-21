$ErrorActionPreference = 'Stop'
$sourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$tempDir = Join-Path $tempRoot ('foc-drive-codec-' + [guid]::NewGuid().ToString('N'))
$tempDirFull = [System.IO.Path]::GetFullPath($tempDir)
if (-not $tempDirFull.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    (Split-Path -Leaf $tempDirFull) -notmatch '^foc-drive-codec-[0-9a-f]{32}$') {
    throw 'Refusing to use an unexpected temporary build directory.'
}
$gccPath = (Get-Command gcc -ErrorAction Stop).Source
New-Item -ItemType Directory -Path $tempDir | Out-Null
try {
    $exePath = Join-Path $tempDir 'drive_codec_tests.exe'
    & $gccPath -std=c11 -Wall -Wextra -Werror -pedantic `
        (Join-Path $sourceDir '速度反馈接口.c') `
        (Join-Path $sourceDir '速度反馈接口测试.c') `
        -o $exePath
    if ($LASTEXITCODE -ne 0) { throw "gcc failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) { throw 'gcc did not produce the test executable.' }
    & $exePath
    if ($LASTEXITCODE -ne 0) { throw "host tests failed with exit code $LASTEXITCODE" }
}
finally {
    if (Test-Path -LiteralPath $tempDirFull) {
        $resolvedTempDir = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $tempDirFull).Path)
        if ($resolvedTempDir -eq $tempDirFull -and
            (Split-Path -Leaf $resolvedTempDir) -match '^foc-drive-codec-[0-9a-f]{32}$') {
            Remove-Item -LiteralPath $resolvedTempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
