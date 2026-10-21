param(
    [ValidateRange(1, 5)]
    [int]$Samples = 3,
    [ValidateRange(1, 10)]
    [int]$IntervalSeconds = 1,
    [int[]]$FocusProcessIds = @(),
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $OutputPath) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutputPath = Join-Path $scriptDir "windows_perf_$stamp.json"
}
$outputDir = Split-Path -Parent $OutputPath
if ($outputDir -and -not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$samplesOut = [System.Collections.Generic.List[object]]::new()
$focusPids = @($FocusProcessIds | Sort-Object -Unique)
$previousCaptureTime = $null

for ($sampleIndex = 0; $sampleIndex -lt $Samples; $sampleIndex++) {
    $captureTime = Get-Date
    $capturedAt = $captureTime.ToString('o')
    $sampleDeltaSeconds = if ($previousCaptureTime) { [math]::Round(($captureTime - $previousCaptureTime).TotalSeconds, 2) } else { $null }
    $previousCaptureTime = $captureTime
    $engineRows = @(Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine)
    $parsed = [System.Collections.Generic.List[object]]::new()
    foreach ($row in $engineRows) {
        if ($row.Name -match '^pid_(?<pid>\d+)_luid_(?<luid>0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)_phys_(?<phys>\d+)_eng_(?<eng>\d+)_engtype_(?<type>.+)$') {
            $parsed.Add([pscustomobject]@{
                pid = [int]$Matches.pid
                raw_luid = $Matches.luid
                engine_type = $Matches.type
                physical_index = [int]$Matches.phys
                engine_index = [int]$Matches.eng
                utilization_pct = [double]$row.UtilizationPercentage
            })
        }
    }

    $allGroups = @($parsed | Group-Object -Property pid, raw_luid, engine_type | ForEach-Object {
        $items = @($_.Group)
        [pscustomobject]@{
            pid = $items[0].pid
            raw_luid = $items[0].raw_luid
            engine_type = $items[0].engine_type
            utilization_pct_max = [math]::Round((($items | Measure-Object utilization_pct -Maximum).Maximum), 1)
            utilization_pct_sum = [math]::Round((($items | Measure-Object utilization_pct -Sum).Sum), 1)
            raw_engines = @($items | ForEach-Object {
                [pscustomobject]@{
                    phys = $_.physical_index
                    eng = $_.engine_index
                    utilization_pct = $_.utilization_pct
                }
            })
        }
    } | Sort-Object -Property utilization_pct_max -Descending)
    $groups = @($allGroups | Select-Object -First 8)

    $wantedPids = @($focusPids + @($groups | ForEach-Object { $_.pid }) | Sort-Object -Unique)
    $processRows = @()
    if ($wantedPids.Count -gt 0) {
        # Request only non-sensitive process fields; CommandLine is not queried.
        $processRows = @(Get-CimInstance -ClassName Win32_Process -Property ProcessId, Name, WorkingSetSize | Where-Object { $wantedPids -contains [int]$_.ProcessId })
    }
    $processMap = @{}
    foreach ($processRow in $processRows) {
        $processMap[[int]$processRow.ProcessId] = [pscustomobject]@{
            pid = [int]$processRow.ProcessId
            name = $processRow.Name
            working_set_mib = [math]::Round(([double]$processRow.WorkingSetSize / 1MB), 1)
        }
    }
    $top8 = @($groups | ForEach-Object {
        $proc = $processMap[[int]$_.pid]
        [pscustomobject]@{
            pid = $_.pid
            process = if ($proc) { $proc.name } else { $null }
            raw_luid = $_.raw_luid
            engine_type = $_.engine_type
            utilization_pct_max = $_.utilization_pct_max
            utilization_pct_sum = $_.utilization_pct_sum
            raw_engines = $_.raw_engines
        }
    })
    $focus = @($focusPids | ForEach-Object {
        $p = [int]$_
        if ($processMap.ContainsKey($p)) {
            $mapped = $processMap[$p]
            $counterRows = @($allGroups | Where-Object { [int]$_.pid -eq $p })
            [pscustomobject]@{
                pid = $p
                process = $mapped.name
                working_set_mib = $mapped.working_set_mib
                gpu_engines = @($counterRows | ForEach-Object {
                    [pscustomobject]@{
                        raw_luid = $_.raw_luid
                        engine_type = $_.engine_type
                        utilization_pct_max = $_.utilization_pct_max
                        utilization_pct_sum = $_.utilization_pct_sum
                        raw_engines = $_.raw_engines
                    }
                })
            }
        } else {
            [pscustomobject]@{ pid = $p; process = $null; working_set_mib = $null; gpu_engines = @() }
        }
    })

    $cpuRows = @(Get-CimInstance -ClassName Win32_Processor)
    $cpuMean = if ($cpuRows.Count) { [math]::Round((($cpuRows | Measure-Object -Property LoadPercentage -Average).Average), 1) } else { $null }
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $ram = [pscustomobject]@{
        total_mib = [math]::Round(([double]$os.TotalVisibleMemorySize * 1KB / 1MB), 1)
        free_mib = [math]::Round(([double]$os.FreePhysicalMemory * 1KB / 1MB), 1)
    }
    $nvidia = $null
    $nvidiaSmi = Get-Command -Name nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $nvidiaText = & $nvidiaSmi.Source '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu' '--format=csv,noheader,nounits' 2>$null
        if ($LASTEXITCODE -eq 0 -and $nvidiaText) {
            $nvidia = @($nvidiaText | ForEach-Object {
                $parts = $_ -split ',\s*'
                if ($parts.Count -ge 5) {
                    [pscustomobject]@{
                        name = $parts[0]
                        utilization_pct = $parts[1]
                        memory_used_mib = $parts[2]
                        memory_total_mib = $parts[3]
                        temperature_c = $parts[4]
                    }
                }
            })
        }
    }
    $adapterInventory = @(Get-CimInstance -ClassName Win32_VideoController | Select-Object Name, DriverVersion)

    $samplesOut.Add([pscustomobject]@{
        captured_at = $capturedAt
        seconds_since_previous_capture = $sampleDeltaSeconds
        cpu_load_pct_mean = $cpuMean
        ram = $ram
        nvidia_smi = $nvidia
        focus_processes = $focus
        top8_gpu_engines = $top8
        parsed_gpu_engine_counter_count = $parsed.Count
        adapter_inventory = $adapterInventory
    })

    if ($sampleIndex -lt ($Samples - 1)) { Start-Sleep -Seconds $IntervalSeconds }
}

$report = [pscustomobject]@{
    schema = 'windows-performance-monitor-snapshot-v1'
    collected_at = (Get-Date).ToString('o')
    read_only = $true
    sample_count = $Samples
    interval_seconds = $IntervalSeconds
    method = 'CIM GPU engine, processor, OS memory, and video controller counters; nvidia-smi read-only query when available.'
    privacy = 'No process command lines, URLs, or full process inventory are collected or stored.'
    gpu_luid_note = 'Raw LUID values are preserved. Win32_VideoController inventory does not directly bind each LUID to an adapter name.'
    samples = @($samplesOut)
}
$json = $report | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($OutputPath), $json, [System.Text.UTF8Encoding]::new($false))
Write-Output ([System.IO.Path]::GetFullPath($OutputPath))
