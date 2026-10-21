[CmdletBinding(DefaultParameterSetName='Inline')]
param(
    [Parameter(Mandatory=$true)]
    [ValidateNotNullOrEmpty()]
    [string]$Session,

    [Parameter(Mandatory=$true, ParameterSetName='Inline')]
    [AllowEmptyString()]
    [string]$Code,

    [Parameter(Mandatory=$true, ParameterSetName='File')]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$CodeFile,

    [ValidateRange(1, 60000)]
    [int]$TimeoutMs = 60000,

    [string]$Executable = 'D:\lceda-pro\lceda-pro.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw 'lceda-pro executable was not found.'
}
if ($PSCmdlet.ParameterSetName -eq 'File') {
    $Code = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $CodeFile).Path)
}

$arguments = [System.Collections.Generic.List[string]]::new()
foreach ($argument in @(
    'invoke',
    '--session', $Session,
    '--ext-uuid', 'eda',
    '--code', $Code,
    '--timeout', $TimeoutMs.ToString([Globalization.CultureInfo]::InvariantCulture)
)) {
    [void]$arguments.Add([string]$argument)
}

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = (Resolve-Path -LiteralPath $Executable).Path
$startInfo.WorkingDirectory = Split-Path -Parent $PSCommandPath
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
foreach ($argument in $arguments) {
    [void]$startInfo.ArgumentList.Add($argument)
}

$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $startInfo
if (-not $process.Start()) {
    throw 'Could not start lceda-pro CLI.'
}
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()
$waitBudgetMs = [Math]::Min(75000, $TimeoutMs + 15000)
if (-not $process.WaitForExit($waitBudgetMs)) {
    try { $process.Kill($true) } catch {}
    throw "lceda-pro CLI timed out after $waitBudgetMs ms."
}
$process.WaitForExit()
$stdout = $stdoutTask.GetAwaiter().GetResult()
$stderr = $stderrTask.GetAwaiter().GetResult()
$processExitCode = $process.ExitCode
$process.Dispose()

$logDirectory = Join-Path (Split-Path -Parent $PSCommandPath) 'logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
# Keep one current API log; reviewed evidence belongs in the merged check record.
$logPath = Join-Path $logDirectory '最近API调用.json'

try {
    $response = ConvertFrom-Json -InputObject $stdout -AsHashtable -ErrorAction Stop
} catch {
    $failure = [ordered]@{
        ok = $false
        error = [ordered]@{
            code = 'INVALID_CLI_RESPONSE'
            message = 'CLI stdout was not valid JSON.'
        }
        processExitCode = $processExitCode
        stdoutBytes = [System.Text.Encoding]::UTF8.GetByteCount([string]$stdout)
        stderrBytes = [System.Text.Encoding]::UTF8.GetByteCount([string]$stderr)
    }
    $safeJson = ConvertTo-Json -InputObject $failure -Depth 20
    [System.IO.File]::WriteAllText($logPath, $safeJson, [System.Text.UTF8Encoding]::new($false))
    Write-Output $safeJson
    throw "CLI invocation failed; inspect the redacted JSON log at $logPath."
}

function Protect-JsonValue {
    param([AllowNull()]$Value)

    if ($null -eq $Value) { return $null }

    if ($Value -is [System.Collections.IDictionary]) {
        $safe = [ordered]@{}
        foreach ($key in $Value.Keys) {
            if ([string]$key -match '(?i)(token|secret|password|passwd|cookie|authorization|credential|api.?key|username|user.?id|email|phone)') {
                $safe[$key] = '[REDACTED]'
            } else {
                $safe[$key] = Protect-JsonValue -Value $Value[$key]
            }
        }
        return ,$safe
    }

    if ($Value -is [System.Array]) {
        $safeItems = [System.Collections.Generic.List[object]]::new()
        foreach ($item in $Value) {
            $safeItems.Add((Protect-JsonValue -Value $item))
        }
        return ,@($safeItems.ToArray())
    }

    return $Value
}

$safeResponse = Protect-JsonValue -Value $response
$safeJson = ConvertTo-Json -InputObject $safeResponse -Depth 100
[System.IO.File]::WriteAllText($logPath, $safeJson, [System.Text.UTF8Encoding]::new($false))

Write-Output $safeJson
if ($processExitCode -ne 0 -or $response.ok -ne $true) {
    throw "CLI invocation returned a failure; inspect the redacted JSON log at $logPath."
}
