param(
    [string]$Python = 'C:\pioneer\mnist-local-neuron-env\Scripts\python.exe',
    [int]$Port = 8765,
    [string]$DataRoot = 'C:\pioneer\local-neuron-runs',
    [int]$Seed = 0,
    [ValidateSet('uoro', 'window')]
    [string]$Mode = 'uoro',
    [int]$UoroRank = 4,
    [double]$LearningRate = 0.003,
    [double]$InputSparsity = 0.02
)

$serverPath = Join-Path $PSScriptRoot 'server.py'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python. Pass -Python with the environment's python.exe path."
}

$serverArguments = @(
    $serverPath,
    '--host', 'localhost',
    '--port', $Port,
    '--data-root', $DataRoot,
    '--seed', $Seed,
    '--mode', $Mode
)
if ($Mode -eq 'uoro') {
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $serverArguments += @(
        '--uoro-rank', $UoroRank,
        '--learning-rate', $LearningRate.ToString($culture),
        '--input-sparsity', $InputSparsity.ToString($culture)
    )
} elseif ($PSBoundParameters.ContainsKey('LearningRate')) {
    $serverArguments += @('--learning-rate', $LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture))
}

& $Python @serverArguments
exit $LASTEXITCODE
