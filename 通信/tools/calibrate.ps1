# One-shot in-situ calibration + rebuild for SLE Range Link (server side).
#
# Usage (keep both boards fixed, do NOT touch/shield them while it runs):
#   & D:\Project\NearLink\Tools\calibrate.ps1 -Port COM22 -ActualMm 500 -Seconds 120
#
# What it does:
#   1. capture N seconds of "[RL] server D=... D1=..." lines from the server port
#   2. take the median of raw D1 (library smoothed distance, before app calibration)
#   3. CALIB_MM = median(D1) - ActualMm  -> write into .config
#   4. rebuild server firmware and copy fwpkg to Artifacts\<OutName>\server
#   5. print the flash command (hold board Reset -> run -> release)
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads BOM-less UTF-8 as
# ANSI, so non-ASCII text here breaks parsing when the script is dot-sourced.

param(
    [string]$Port = "COM22",
    [int]$ActualMm = 500,
    [int]$Seconds = 120,
    [string]$OutName = "range_link_v24"
)

$ErrorActionPreference = "Stop"
$root = "D:\Project\NearLink"
$sdk  = Join-Path $root "SDK\bearpi-pico_h2821e"
$cfg  = Join-Path $sdk "build\config\target_config\bs21e\menuconfig\acore\standard_bs21e_1100e.config"
$tc   = Join-Path $root "ToolChain"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "[1/4] Capturing $Seconds s of raw distance (D1) from $Port ..." -ForegroundColor Cyan
$sp = New-Object System.IO.Ports.SerialPort $Port, 115200, "None", 8, "One"
$sp.ReadTimeout = 200
$sb = New-Object System.Text.StringBuilder
try {
    $sp.Open()
    $end = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $end) {
        try { [void]$sb.Append($sp.ReadExisting()) } catch {}
        Start-Sleep -Milliseconds 80
    }
} finally {
    if ($sp.IsOpen) { $sp.Close() }
}
$rawFile = Join-Path $root ("Docs\range_link\capture_$stamp.txt")
Set-Content -Path $rawFile -Value $sb.ToString() -NoNewline
Write-Host "      raw log -> $rawFile"

$lines = @($sb.ToString() -split "`r?`n" | Where-Object { $_ -match "\[RL\] server D=" })
$d1 = @()
$dOut = @()
$reboots = @($sb.ToString() -split "`r?`n" | Where-Object { $_ -match "Flashboot Init|Reboot cause|exception:|Oops:NMI" })
foreach ($l in $lines) {
    if ($l -match "D1=(-?\d+)mm") { $v = [int]$Matches[1]; if ($v -gt 0) { $d1 += $v } }
    if ($l -match " D=(-?\d+)mm")  { $dOut += [int]$Matches[1] }
}
if ($d1.Count -lt 10) {
    throw "Too few samples ($($d1.Count) valid D1). Check: both boards connected, port not held by another tool, server firmware >= v21."
}
$sorted = $d1 | Sort-Object
$median = $sorted[[int]($sorted.Count / 2)]
$calib  = $median - $ActualMm
$mean   = ($d1 | Measure-Object -Average).Average
$sd     = 0.0
foreach ($x in $d1) { $sd += ($x - $mean) * ($x - $mean) }
$sd = [math]::Sqrt($sd / $d1.Count)

Write-Host ("[2/4] samples={0}  D1 median={1}mm  mean={2}mm  sigma={3}mm  min={4}  max={5}" -f `
    $d1.Count, $median, [math]::Round($mean, 1), [math]::Round($sd, 1), $sorted[0], $sorted[-1]) -ForegroundColor Green
if ($dOut.Count -gt 0) {
    $ds = $dOut | Sort-Object
    Write-Host ("      current output D: median={0}mm  (target {1}mm, error {2}mm)" -f `
        $ds[[int]($ds.Count / 2)], $ActualMm, ($ds[[int]($ds.Count / 2)] - $ActualMm)) -ForegroundColor Green
}
if ($reboots.Count -gt 0) {
    Write-Host ("      WARNING: {0} reset/exception lines found in the capture (watchdog?)" -f $reboots.Count) -ForegroundColor Yellow
}
Write-Host ("      actual={0}mm  =>  CONFIG_RANGE_LINK_CALIB_MM={1}" -f $ActualMm, $calib) -ForegroundColor Green

Write-Host "[3/4] Writing .config and rebuilding server ..." -ForegroundColor Cyan
$txt = Get-Content $cfg -Raw
$txt = $txt -replace "CONFIG_RANGE_LINK_CALIB_MM=\d+", "CONFIG_RANGE_LINK_CALIB_MM=$calib"
$txt = $txt -replace "CONFIG_RANGE_LINK_CALIB_SCALE_PM=\d+", "CONFIG_RANGE_LINK_CALIB_SCALE_PM=1000"
$txt = $txt -replace "# CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER is not set", "CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER=y"
$txt = $txt -replace "CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT=y", "# CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT is not set"
Set-Content -Path $cfg -Value $txt -NoNewline

$env:Path = "$tc\tools\python;$tc\tools\python\Scripts;$tc\tools\Windows\ninja;$tc\tools\Windows\cc_riscv32_win_env;" + $env:Path
Push-Location $sdk
try {
    $blog = Join-Path $root ("Docs\range_link\build_$stamp.log")
    & python build.py -c -j8 standard-bs21e-1100e *> $blog
    if ($LASTEXITCODE -ne 0) { throw "build failed, see $blog" }
} finally { Pop-Location }

$fw  = Join-Path $sdk "tools\pkg\fwpkg\bs21e\bs21e_all.fwpkg"
$mem = Join-Path $sdk "output\bs21e\acore\standard-bs21e-1100e\application.mem"
$outDir = Join-Path $root ("Artifacts\$OutName\server")
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
Copy-Item $fw  (Join-Path $outDir "bs21e_all.fwpkg") -Force
if (Test-Path $mem) { Copy-Item $mem (Join-Path $outDir "application.mem") -Force }
$buildInfo = @(
    "out=$OutName",
    "built=$stamp",
    "calib_mm=$calib",
    "scale_pm=1000",
    "actual_mm=$ActualMm",
    "d1_median_mm=$median",
    "d1_mean_mm=$([math]::Round($mean,1))",
    "d1_sigma_mm=$([math]::Round($sd,1))",
    "samples=$($d1.Count)",
    "capture=$rawFile"
)
Set-Content -Path (Join-Path $outDir "build_info.txt") -Value $buildInfo

Write-Host "[4/4] Build OK -> $outDir" -ForegroundColor Green
Write-Host "Flash (hold board Reset -> run -> release Reset):" -ForegroundColor Yellow
Write-Host ("  & '{0}\tools\hsflash\hsflash.exe' write -c bs21e -p {1} -b 750000 -f '{2}\bs21e_all.fwpkg' --after-reset" -f $tc, $Port, $outDir)
