$ErrorActionPreference = 'Stop'
$taskWorkspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path.TrimEnd('\')
$taskArchiveBase = [System.IO.Path]::GetFullPath((Join-Path $taskWorkspace '归档/front_guide_free_rotation_20261020'))
$taskSourceDirectories = @(
    'results/isaac_field_diagnosis/front_guide_assets_20261020',
    'results/isaac_field_diagnosis/front_guide_integration_20261020',
    'results/isaac_field_diagnosis/front_guide_step_probe_20261020',
    'results/isaac_field_improvement/random_global_rough_20261020',
    'results/isaac_field_improvement/random_global_rough_20261020_validation',
    'results/isaac_field_viewer/front_guides_rough_20261020'
)
if (Test-Path -LiteralPath $taskArchiveBase) { throw 'Archive destination already exists; refusing to overwrite.' }
if (-not $taskArchiveBase.StartsWith($taskWorkspace + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Archive path escaped the workspace.'
}
$taskMoves = foreach ($taskRelative in $taskSourceDirectories) {
    $taskSource = (Get-Item -LiteralPath (Join-Path $taskWorkspace $taskRelative)).FullName
    $taskTarget = [System.IO.Path]::GetFullPath((Join-Path $taskArchiveBase $taskRelative))
    if (-not $taskSource.StartsWith($taskWorkspace + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $taskTarget.StartsWith($taskArchiveBase + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unverified move target: $taskRelative"
    }
    if (Test-Path -LiteralPath $taskTarget) { throw "Target already exists: $taskTarget" }
    [PSCustomObject]@{ original_relative = $taskRelative; original_absolute = $taskSource; archive_absolute = $taskTarget }
}
$taskFileRows = foreach ($taskMove in $taskMoves) {
    foreach ($taskFile in Get-ChildItem -LiteralPath $taskMove.original_absolute -Recurse -File) {
        $taskSuffix = $taskFile.FullName.Substring($taskMove.original_absolute.Length).TrimStart('\')
        [PSCustomObject]@{
            original_absolute = $taskFile.FullName
            original_relative = $taskFile.FullName.Substring($taskWorkspace.Length + 1)
            archive_absolute = Join-Path $taskMove.archive_absolute $taskSuffix
            bytes = $taskFile.Length
            sha256 = (Get-FileHash -LiteralPath $taskFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
}
New-Item -ItemType Directory -Path $taskArchiveBase | Out-Null
$taskPlan = [ordered]@{
    created_utc = [DateTime]::UtcNow.ToString('o')
    reason = 'User requested coupled front guides and cleanup; superseded free-guide trials retained without byte changes.'
    workspace = $taskWorkspace
    moves = @($taskMoves)
    files = @($taskFileRows)
}
$taskPlan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $taskArchiveBase 'archive_manifest_before.json') -Encoding utf8
foreach ($taskMove in $taskMoves) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $taskMove.archive_absolute) -Force | Out-Null
    Move-Item -LiteralPath $taskMove.original_absolute -Destination $taskMove.archive_absolute
}
$taskVerified = foreach ($taskRow in $taskFileRows) {
    $taskAfter = Get-Item -LiteralPath $taskRow.archive_absolute
    $taskAfterHash = (Get-FileHash -LiteralPath $taskAfter.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskAfter.Length -ne $taskRow.bytes -or $taskAfterHash -ne $taskRow.sha256) {
        throw "Archived bytes differ: $($taskRow.original_relative)"
    }
    [PSCustomObject]@{ path = $taskRow.original_relative; bytes = $taskAfter.Length; sha256 = $taskAfterHash; matched = $true }
}
$taskResult = [ordered]@{
    actual_exit_code = 0
    verified_utc = [DateTime]::UtcNow.ToString('o')
    directories_moved = @($taskMoves).Count
    files_verified = @($taskVerified).Count
    bytes_verified = ($taskVerified | Measure-Object -Property bytes -Sum).Sum
    files = @($taskVerified)
}
$taskResult | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskArchiveBase 'archive_verification.json') -Encoding utf8
[PSCustomObject]@{ archive = $taskArchiveBase; directories = @($taskMoves).Count; files_verified = @($taskVerified).Count; actual_exit_code = 0 } | ConvertTo-Json
