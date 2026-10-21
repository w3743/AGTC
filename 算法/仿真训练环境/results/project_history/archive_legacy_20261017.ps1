$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$archive = Join-Path $root '归档/legacy_before_isaac_20261017'
$prefix = $root.TrimEnd('\') + '\'
$archivePrefix = [IO.Path]::GetFullPath($archive).TrimEnd('\') + '\'
if (-not $archivePrefix.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Archive outside workspace' }
if (Test-Path -LiteralPath $archive) { throw 'Archive already exists; inspect its manifest instead of rerunning' }

$paths = @(
    'field_sim', 'simulation', 'training', 'config', 'uwb_plugin', 'dist',
    '.venv-field', '.runtime-gazebo', '.runtime-gazebo-wheels',
    '安装环境.cmd', '检查环境.cmd', '启动实验室.cmd',
    '启动农田仿真.cmd', '测试农田性能.cmd', '启动农田Gazebo.cmd', '启动Gazebo短训练.cmd', '启动Gazebo浏览器.cmd',
    'scripts/build_gazebo_control.sh', 'scripts/field_gazebo.ps1', 'scripts/field_gazebo.sh',
    'scripts/field_gazebo_web.ps1', 'scripts/field_gazebo_web.sh',
    'scripts/field_train_gazebo.ps1', 'scripts/field_train_gazebo.sh', 'scripts/gazebo_python.sh',
    'scripts/install.sh', 'scripts/launch.sh', 'scripts/windows.ps1', 'scripts/env.sh',
    'scripts/sources.py', 'scripts/generate_world.py', 'scripts/verify.py', 'scripts/package.py',
    'pelican-bicycle.html', 'ISAAC_CLEANUP_CANDIDATES.json',
    'results/drone_20260923_105822', 'results/drone_20260923_110423',
    'results/field_mvp', 'results/field_policy_improvement',
    'results/gazebo_field', 'results/gazebo_field_runs', 'results/gazebo_field_source',
    'results/gazebo_field_training', 'results/gazebo_training', 'results/gazebo_web',
    'results/learning', 'results/lightstereo', 'results/manifests',
    'results/mapping_acceptance', 'results/mapping_cyclone', 'results/mapping_gpu',
    'results/mapping_gui', 'results/mapping_gui_final',
    'results/policy_device_benchmark_20261017_164208',
    'results/deployment.json', 'results/field_sim_terrain_phase_paired_benchmark_20261017.json',
    'results/install.log', 'results/package.json'
)

# Check every absolute source/target before any move; preserve relative layout.
$entries = foreach ($relative in $paths) {
    $source = [IO.Path]::GetFullPath((Join-Path $root $relative))
    $target = [IO.Path]::GetFullPath((Join-Path $archive $relative))
    if (-not $source.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Source outside workspace: $source" }
    if (-not $target.StartsWith($archivePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Target outside archive: $target" }
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
    if (Test-Path -LiteralPath $target) { throw "Target exists: $target" }
    $item = Get-Item -LiteralPath $source -Force
    $children = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $source -Recurse -Force) } else { @() }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or @($children | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw "Reparse point requires separate review: $source" }
    [pscustomobject]@{ relative = $relative; source = $source; target = $target; status = 'planned' }
}
foreach ($entry in $entries) {
    foreach ($other in $entries) {
        if ($entry -ne $other -and $entry.source.StartsWith($other.source + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Overlapping move roots' }
    }
}

$files = @(foreach ($entry in $entries) {
    $item = Get-Item -LiteralPath $entry.source -Force
    $listing = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $entry.source -Recurse -File -Force) } else { @($item) }
    foreach ($file in $listing) {
        [pscustomobject]@{
            relative = [IO.Path]::GetRelativePath($root, $file.FullName).Replace('\', '/')
            bytes = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
})

$protected = @(foreach ($folder in @('isaac_field', 'tests')) {
    Get-ChildItem -LiteralPath (Join-Path $root $folder) -Recurse -File -Filter '*.py'
}) + @(Get-ChildItem -LiteralPath (Join-Path $root 'scripts') -File | Where-Object { -not ($paths -contains ('scripts/' + $_.Name)) })
$protected += @(foreach ($name in @('pyproject.toml', '安装IsaacLab.cmd', '启动IsaacLab.cmd')) { Get-Item -LiteralPath (Join-Path $root $name) })
$protectedHashes = @(foreach ($file in $protected) {
    [pscustomobject]@{ relative = [IO.Path]::GetRelativePath($root, $file.FullName).Replace('\', '/'); sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
})

New-Item -ItemType Directory -Path $archive | Out-Null
$manifest = [ordered]@{
    schema = 'isaac-legacy-archive-v1'
    started_at = (Get-Date -Format o)
    authorization = 'User requested moving unrelated legacy files to 归档 and explicitly included old models/results.'
    operation = 'Move-Item only; no deletion; original relative paths retained'
    root = $root
    archive = $archive
    status = 'moving'
    entries = @($entries)
    files = $files
    protected_sources = $protectedHashes
}
$manifestPath = Join-Path $archive 'archive_manifest.json'
function Save-Manifest { $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8 }
Save-Manifest
foreach ($entry in $entries) {
    $parent = Split-Path -Parent $entry.target
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -LiteralPath $entry.source -Destination $entry.target
    $entry.status = 'moved'
    Save-Manifest
}

foreach ($file in $files) {
    $target = Join-Path $archive $file.relative
    if ((Get-Item -LiteralPath $target -Force).Length -ne $file.bytes -or (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) { throw "Archived file mismatch: $target" }
}
foreach ($file in $protectedHashes) {
    if ((Get-FileHash -LiteralPath (Join-Path $root $file.relative) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) { throw "Current Isaac source changed: $($file.relative)" }
}
if (@($entries | Where-Object { Test-Path -LiteralPath $_.source }).Count) { throw 'Some original paths still exist' }
$actualFiles = @(foreach ($entry in $entries) {
    $item = Get-Item -LiteralPath $entry.target -Force
    if ($item.PSIsContainer) { Get-ChildItem -LiteralPath $entry.target -Recurse -File -Force } else { $item }
})
if ($actualFiles.Count -ne $files.Count) { throw 'Archived file count mismatch' }
$manifest.status = 'verified'
$manifest['completed_at'] = Get-Date -Format o
$manifest['verification'] = [ordered]@{ moved_roots = $entries.Count; file_count = $files.Count; total_bytes = ($files | Measure-Object bytes -Sum).Sum; all_sha256_match = $true; protected_source_count = $protectedHashes.Count; protected_sources_unchanged = $true }
Save-Manifest
$manifest.verification | ConvertTo-Json -Compress
