[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Session,
    [Parameter(Mandatory=$true)][string]$Destination,
    [string]$Executable = 'D:\lceda-pro\lceda-pro.exe'
)

$ErrorActionPreference = 'Stop'
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$outputDir = [System.IO.Path]::GetDirectoryName($destinationPath)
[System.IO.Directory]::CreateDirectory($outputDir) | Out-Null
$code = 'const f=await eda.sch_ManufactureData.getExportDocumentFile("FOC_G431_Driver_preview", "PDF", {theme:"Black on White",lineWidth:"Default"}, "Current Schematic", {outputMethod:"Merged sheet"}); if(!f)throw new Error("PDF export returned no file"); const data=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(",")[1]);r.onerror=reject;r.readAsDataURL(f);}); return {name:f.name,size:f.size,data};'
$responseText = & (Join-Path $PSScriptRoot 'Invoke-Lceda.ps1') -Session $Session -Code $code -TimeoutMs 60000 -Executable $Executable | Out-String
$response = ConvertFrom-Json -InputObject $responseText -ErrorAction Stop
if (-not $response.ok -or -not $response.value.data) { throw 'Official PDF export failed or returned no file data.' }
$bytes = [Convert]::FromBase64String($response.value.data)
if ($bytes.Length -ne [int]$response.value.size) { throw "PDF byte mismatch: got $($bytes.Length), expected $($response.value.size)." }
[System.IO.File]::WriteAllBytes($destinationPath, $bytes)
$fileHash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash
[pscustomobject]@{Path=$destinationPath;Bytes=$bytes.Length;SHA256=$fileHash;NativeName=$response.value.name}|ConvertTo-Json -Compress
