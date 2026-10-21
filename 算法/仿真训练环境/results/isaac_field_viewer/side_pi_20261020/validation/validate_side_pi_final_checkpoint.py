"""Read-only CPU validation of the final fresh side-PI v6 checkpoint.

This entry point reuses the existing baseline/checkpoint validator without
changing its 8192-step expectations. It pins the trial protocol, source count,
baseline hash, and final expected step count for the 196608-step artifact.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback
import validate_side_pi_checkpoint as shared


EXPECTED_PROTOCOL_SHA256 = "0a356055c2ff08e13133ebfe12c71e90c63612a8efce7ea0db8b1187aafd10b8"
EXPECTED_BASELINE_SHA256 = "bdb944d4dbba038536687810089448aecb132d07979dd5493c8a33ebbe9e0129"
EXPECTED_SOURCE_COUNT = 41
EXPECTED_FINAL_STEPS = 196608


def _path_signature(repo_root: Path, path: Path) -> dict:
    resolved = shared._inside(path, repo_root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return shared.signature(resolved)


def validate(args) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = args.repo_root.resolve()
    trial = shared._inside(args.trial_dir, root)
    protocol_path = shared._inside(args.protocol, root)
    manifest_path = shared._inside(args.manifest, root)
    status_path = shared._inside(args.run_status, root)
    baseline_path = shared._inside(args.baseline, root)
    checkpoint_path = shared._inside(args.checkpoint, root)
    reference_path = shared._inside(args.protected_reference, root)
    output_path = shared._inside(args.output, Path(__file__).resolve().parent)

    if args.expected_steps != EXPECTED_FINAL_STEPS:
        raise ValueError(
            f"this final validator is pinned to {EXPECTED_FINAL_STEPS} steps, "
            f"got {args.expected_steps}"
        )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite validation output: {output_path}")

    stable_paths = {
        "protocol": protocol_path,
        "manifest": manifest_path,
        "run_status": status_path,
        "baseline": baseline_path,
        "checkpoint": checkpoint_path,
        "protected_reference": reference_path,
    }
    files_before = {
        name: _path_signature(root, path) for name, path in stable_paths.items()
    }
    protocol = shared.read_json(protocol_path)
    sources_before = shared.source_snapshot_check(root, trial, protocol)
    old_trial_before = shared.protected_old_trial_check(root, reference_path)

    # Keep the shared implementation read-only; scope the expected step count
    # to this process and pass the pinned digests explicitly.
    shared.EXPECTED_STEPS = args.expected_steps
    shared.EXPECTED_SOURCE_COUNT = EXPECTED_SOURCE_COUNT
    common_args = argparse.Namespace(
        repo_root=root,
        trial_dir=trial,
        protocol=protocol_path,
        manifest=manifest_path,
        run_status=status_path,
        baseline=baseline_path,
        checkpoint=checkpoint_path,
        baseline_only=False,
        expected_protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        expected_baseline_sha256=EXPECTED_BASELINE_SHA256,
        protected_reference=reference_path,
        output=output_path,
    )
    report = shared.run_validation(common_args)

    files_after = {
        name: _path_signature(root, path) for name, path in stable_paths.items()
    }
    sources_after = shared.source_snapshot_check(root, trial, protocol)
    old_trial_after = shared.protected_old_trial_check(root, reference_path)
    files_stable = files_before == files_after
    sources_stable = sources_before == sources_after
    old_trial_stable = old_trial_before == old_trial_after

    report["schema"] = "fresh-side-pi-v6-readonly-final-checkpoint-validation-v1"
    report["expected_steps"] = args.expected_steps
    report["file_read_stability"] = {
        "before": files_before,
        "after": files_after,
        "all_inputs_unchanged": files_stable,
    }
    report["source_hash_read_stability"] = {
        "before": sources_before,
        "after": sources_after,
        "unchanged_during_read": sources_stable,
    }
    report["protected_old_trial"] = {
        "before": old_trial_before,
        "after": old_trial_after,
        "unchanged_during_read": old_trial_stable,
    }

    checks = report.setdefault("checks", {})
    old_step_check = checks.pop("checkpoint_8192_step_metadata_and_finite", None)
    if old_step_check is not None:
        checks["checkpoint_expected_step_metadata_and_finite"] = old_step_check
    checks.update({
        "pinned_protocol_sha256": report.get("protocol", {}).get("sha256") == EXPECTED_PROTOCOL_SHA256,
        "pinned_41_source_count": sources_after.get("actual_count") == EXPECTED_SOURCE_COUNT,
        "source_hashes_unchanged_during_read": sources_stable,
        "all_trial_and_reference_files_unchanged_during_read": files_stable,
        "protected_old_trial_hashes_match_before_and_after": (
            old_trial_before.get("all_match") and old_trial_after.get("all_match") and old_trial_stable
        ),
    })
    report["status"] = "passed" if all(checks.values()) else "failed"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-status", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--protected-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_path = args.output.resolve()
    try:
        output_path.relative_to(script_dir)
    except ValueError:
        parser.error(f"output must stay under the validation directory: {script_dir}")
    if output_path.exists():
        parser.error(f"refusing to overwrite validation output: {output_path}")

    try:
        report = validate(args)
    except Exception as exc:
        report = {
            "schema": "fresh-side-pi-v6-readonly-final-checkpoint-validation-v1",
            "status": "failed",
            "expected_steps": args.expected_steps,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "expected_steps": args.expected_steps,
        "output": str(output_path),
        "checks": report.get("checks", {}),
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
