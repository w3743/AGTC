[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateNotNullOrEmpty()]
    [string[]]$CliArguments,

    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')]
    [string]$LogName,

    [ValidateRange(1000, 1810000)]
    [int]$ProcessTimeoutMs = 180000,

    [string]$Executable = 'D:\lceda-pro\lceda-pro.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw 'lceda-pro executable was not found.'
}
if ($CliArguments.Count -eq 0) {
    throw 'Provide at least one CLI argument.'
}

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = (Resolve-Path -LiteralPath $Executable).Path
$startInfo.WorkingDirectory = Split-Path -Parent $PSCommandPath
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
foreach ($argument in $CliArguments) {
    if ($null -eq $argument) { throw 'CLI arguments cannot contain null values.' }
    [void]$startInfo.ArgumentList.Add([string]$argument)
}

$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $startInfo
if (-not $process.Start()) {
    throw 'Could not start lceda-pro CLI.'
}
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()
if (-not $process.WaitForExit($ProcessTimeoutMs)) {
    try { $process.Kill($true) } catch {}
    throw "lceda-pro CLI timed out after $ProcessTimeoutMs ms."
}
$process.WaitForExit()
$stdout = $stdoutTask.GetAwaiter().GetResult()
$stderr = $stderrTask.GetAwaiter().GetResult()
$processExitCode = $process.ExitCode
$process.Dispose()

$logDirectory = Join-Path (Split-Path -Parent $PSCommandPath) 'logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
# Keep one current CLI log; reviewed evidence belongs in the merged check record.
$logPath = Join-Path $logDirectory '最近CLI调用.json'

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
    throw "CLI response was invalid; inspect the JSON log at $logPath."
}

function Protect-JsonValue {
    param(
        [AllowNull()]$Value,
        [string]$KeyName = '',
        [switch]$OmitBinary
    )

    if ($null -eq $Value) { return $null }

    if ($KeyName -match '(?i)(token|secret|password|passwd|cookie|authorization|credential|api.?key|username|user.?id|email|phone|account|owner|author|endpoint)') {
        return '[REDACTED]'
    }

    if ($Value -is [System.Collections.IDictionary]) {
        $safe = [ordered]@{}
        foreach ($key in $Value.Keys) {
            $safe[$key] = Protect-JsonValue -Value $Value[$key] -KeyName ([string]$key) -OmitBinary:$OmitBinary
        }
        return ,$safe
    }

    if ($Value -is [System.Array]) {
        $safeItems = [System.Collections.Generic.List[object]]::new()
        foreach ($item in $Value) {
            $safeItems.Add((Protect-JsonValue -Value $item -OmitBinary:$OmitBinary))
        }
        return ,@($safeItems.ToArray())
    }

    if ($OmitBinary -and $Value -is [string] -and $Value.Length -gt 4096) {
        $looksBase64 = $Value -match '^(?:[A-Za-z0-9+/]{100,}={0,2})$'
        $looksDataUri = $Value -match '^data:[^,]*;base64,[A-Za-z0-9+/]{100,}={0,2}$'
        $binaryKey = $KeyName -match '(?i)(b64|base64|blob|image.?data|data.?uri|binary|file.?data)'
        if ($looksBase64 -or $looksDataUri -or ($binaryKey -and $Value -match '^[A-Za-z0-9+/=]+$')) {
            return "[binary payload omitted from console; $($Value.Length) characters saved in log]"
        }
    }

    return $Value
}

$logResponse = Protect-JsonValue -Value $response
$logJson = ConvertTo-Json -InputObject $logResponse -Depth 100
[System.IO.File]::WriteAllText($logPath, $logJson, [System.Text.UTF8Encoding]::new($false))

$displayResponse = Protect-JsonValue -Value $response -OmitBinary
$displayJson = ConvertTo-Json -InputObject $displayResponse -Depth 100
Write-Output $displayJson
Write-Verbose "Saved response JSON to $logPath"

if ($processExitCode -ne 0 -or $response.ok -ne $true) {
    throw "lceda-pro returned a failure; inspect the redacted JSON log at $logPath."
}
