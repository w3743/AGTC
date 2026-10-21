$ErrorActionPreference = 'Stop'
$snapshotRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $snapshotRoot '..\..\..\..\..')).Path
$originalRoot = Join-Path $snapshotRoot 'original'
$manifestPath = Join-Path $snapshotRoot 'source_snapshot_manifest.json'
if ((Test-Path -LiteralPath $originalRoot) -or (Test-Path -LiteralPath $manifestPath)) {
    throw "Refusing to overwrite source snapshot under $snapshotRoot"
}

$relativePaths = @(
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/runtime_pi.py',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/test_pi_counterfactual_math.py',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/pi_counterfactual_probe.py',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/compare_pi_counterfactual_runs.py',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_pi_probe.sh',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_native_v5_regression_attempt1.sh',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_paired_capped_p_and_pi.sh',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_cpu_math_tests.sh',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_cpu_math_tests_attempt2.sh',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_post_side_static_checks.sh',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_final_static_validation.sh',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/run_review_static_validation.sh',
    'results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/source_freeze_sha256.txt',
    'tests/newton_effort_capture.py',
    'tests/test_env_integration.py'
)

New-Item -ItemType Directory -Path $originalRoot | Out-Null
$entries = foreach ($relative in $relativePaths) {
    $source = Join-Path $projectRoot $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing snapshot source: $relative" }
    $destination = Join-Path $originalRoot $relative
    $destinationParent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
    $snapshotHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
    if ($sourceHash -ne $snapshotHash) { throw "Snapshot hash mismatch: $relative" }
    [pscustomobject]@{
        relative_path = $relative
        sha256 = $sourceHash
        snapshot_relative_path = [IO.Path]::GetRelativePath($snapshotRoot, $destination)
        source_matches_snapshot = $true
    }
}
$snapshotScriptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $MyInvocation.MyCommand.Path).Hash.ToLowerInvariant()
$manifest = [pscustomobject]@{
    schema = 'pi-counterfactual-source-snapshot-v1'
    captured_utc = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    capture_script_sha256 = $snapshotScriptHash
    original_source_count = @($entries).Count
    original_sources = @($entries)
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Output "SOURCE_SNAPSHOT_FILES=$(@($entries).Count)"
Write-Output "SOURCE_SNAPSHOT_MANIFEST=$manifestPath"
