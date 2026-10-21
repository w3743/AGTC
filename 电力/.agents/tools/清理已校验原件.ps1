$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..')).TrimEnd('\')
$recordPath = Join-Path $projectRoot '归档/目录整理.json'
$record = Get-Content -LiteralPath $recordPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([IO.Path]::GetFullPath($record.root).TrimEnd('\') -ne $projectRoot) { throw '归档记录根路径不符' }
if ($record.status -ne 'VERIFIED_READY_FOR_POWERSHELL_CLEANUP') { throw '归档尚未完成逐文件读回' }
$rows = @($record.files | Where-Object cleanup)
if (@($record.files | Where-Object { -not $_.readback }).Count) { throw '存在未校验成员' }
$emptyParents = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$verifiedFiles = @()
$protected = @($record.protected_sha256.PSObject.Properties.Name)
foreach ($archive in $record.archives.PSObject.Properties) {
    $archivePath = [IO.Path]::GetFullPath((Join-Path $projectRoot $archive.Name))
    if (-not $archivePath.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'ZIP路径越界' }
    if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $archive.Value.after_sha256) { throw "已读回ZIP发生变化 $archivePath" }
}
$done = 0
foreach ($row in $rows) {
    $target = [IO.Path]::GetFullPath((Join-Path $projectRoot $row.path))
    if (-not $target.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "路径越界 $target" }
    $item = Get-Item -LiteralPath $target -Force
    if ($row.path -in $protected) { throw '命中保护工程' }
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "拒绝目录或联接 $target" }
    $parent = $item.Directory
    while ($parent -and $parent.FullName -ne $projectRoot) {
        if ($parent.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "父目录联接 $target" }
        [void]$emptyParents.Add($parent.FullName)
        $parent = $parent.Parent
    }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) { throw "源文件已变 $target" }
    $verifiedFiles += [PSCustomObject]@{Path=$target;Length=$item.Length;Ticks=$item.LastWriteTimeUtc.Ticks}
}
Write-Output "全部原件及已读回ZIP的SHA256预检通过，开始清理。"
foreach ($verified in $verifiedFiles) {
    $target = $verified.Path
    $item = Get-Item -LiteralPath $target -Force
    if ($item.Length -ne $verified.Length -or $item.LastWriteTimeUtc.Ticks -ne $verified.Ticks) { throw "预检后原件发生变化 $target" }
    Remove-Item -LiteralPath $target -Force
    $done++
    if ($done % 250 -eq 0) { Write-Output "已清理 $done / $($rows.Count)" }
}
foreach ($folder in @($emptyParents) | Sort-Object Length -Descending) {
    if (-not $folder.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw '空目录路径越界' }
    if ($folder -match '\\.git(\\|$)') { throw '拒绝Git目录' }
    if ((Test-Path -LiteralPath $folder) -and -not (Get-ChildItem -LiteralPath $folder -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $folder -Force
    }
}
Write-Output "完成：$done 个已读回原件清理；未使用递归删除。"
