param(
    [string]$Python = 'C:\pioneer\mnist-local-neuron-env\Scripts\python.exe',
    [int]$Port = 8765,
    [string]$DataRoot = 'C:\pioneer\local-neuron-runs',
    [int]$Seed = 0
)

$serverPath = Join-Path $PSScriptRoot 'server.py'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python. Pass -Python with the environment's python.exe path."
}

$serverArguments = @(
    $serverPath,
    '--port', $Port,
    '--data-root', $DataRoot,
    '--seed', $Seed
)

& $Python @serverArguments
exit $LASTEXITCODE
