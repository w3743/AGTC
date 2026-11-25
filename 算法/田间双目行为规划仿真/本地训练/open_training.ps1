$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$sim = Split-Path -Parent $root
$cache = Join-Path $sim '.cache'
$python = $env:AGTC_TRAIN_PYTHON
if (!$python) { $python = Join-Path $sim '.venv\Scripts\python.exe' }
if (!(Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Create the project .venv and install requirements.txt, or set AGTC_TRAIN_PYTHON.'
}
$nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (!$nodeCommand) { throw 'Install Node.js and add it to PATH.' }
$node = $nodeCommand.Source
$browser = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
if (!(Test-Path -LiteralPath $browser)) { $browser = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe' }
if (!(Test-Path -LiteralPath $python) -or !(Test-Path -LiteralPath $node) -or !(Test-Path -LiteralPath $browser)) {
    throw 'Missing local Python, Node.js, or Chrome/Edge.'
}
New-Item -ItemType Directory -Force -Path $cache | Out-Null
$env:TEMP = $cache
$env:TMP = $cache
$env:TORCHINDUCTOR_CACHE_DIR = $cache
$env:PYTHONIOENCODING = 'utf-8'
$env:AGTC_TRAIN_PYTHON = $python
try {
    $npu = Get-PnpDevice -PresentOnly -ErrorAction Stop |
        Where-Object { $_.Class -eq 'ComputeAccelerator' -and $_.FriendlyName -match 'AI Boost|NPU' } |
        Select-Object -First 1
    if ($npu) { $env:AGTC_NPU_NAME = $npu.FriendlyName }
} catch {}

function Ready($url, $property) {
    try {
        $reply = Invoke-RestMethod -Uri $url -TimeoutSec 2
        return $null -ne $reply.$property
    } catch { return $false }
}
function Start-Local($entry,$argument,$folder,$name,$check,$property) {
    if (Ready $check $property) { return }
    $launch = @{
        FilePath = $entry
        ArgumentList = $argument
        WorkingDirectory = $folder
        WindowStyle = 'Hidden'
        RedirectStandardOutput = Join-Path $folder ($name+'.log')
        RedirectStandardError = Join-Path $folder ($name+'.err')
    }
    Start-Process @launch
    for ($i=0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 300
        if (Ready $check $property) { return }
    }
    throw ('Local service failed to start: ' + (Join-Path $folder ($name+'.err')))
}

Start-Local $python @('-B','offroad_server.py') $sim 'server' 'http://127.0.0.1:8922/api/scene' 'heights'
Start-Local $node @('workbench.cjs') $root 'workbench' 'http://127.0.0.1:8924/api/state' 'defaults'
$page = [IO.File]::ReadAllText((Join-Path $root 'workbench.html'),[Text.Encoding]::UTF8)
$title = [regex]::Match($page,'<title>([^<]+)</title>').Groups[1].Value
if (-not ('AgtcTrainingWindow' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class AgtcTrainingWindow {
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] private static extern IntPtr FindWindow(string cls,string title);
    [DllImport("user32.dll")] private static extern bool IsIconic(IntPtr hwnd);
    [DllImport("user32.dll")] private static extern bool ShowWindow(IntPtr hwnd,int command);
    [DllImport("user32.dll")] private static extern bool SetForegroundWindow(IntPtr hwnd);
    public static bool Focus(string title) {
        IntPtr hwnd=FindWindow("Chrome_WidgetWin_1",title);
        if(hwnd==IntPtr.Zero) return false;
        if(IsIconic(hwnd)) ShowWindow(hwnd,9);
        SetForegroundWindow(hwnd);
        return true;
    }
}
'@
}
if ($title -and [AgtcTrainingWindow]::Focus($title)) { return }
$profile = Join-Path $cache 'training-browser'
Start-Process -FilePath $browser -ArgumentList @(('--user-data-dir="' + $profile + '"'),'--no-first-run',
    '--no-default-browser-check','--disable-sync','--window-size=1580,1040','--app=http://127.0.0.1:8924/')
