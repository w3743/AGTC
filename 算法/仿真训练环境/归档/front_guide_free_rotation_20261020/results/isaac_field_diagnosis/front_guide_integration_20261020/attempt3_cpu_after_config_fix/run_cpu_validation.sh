#!/usr/bin/env bash
set -uo pipefail
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OUT/../../../.." && pwd)"
cd "$ROOT"
unset RUN_ISAAC_PHYSICS
export CUDA_VISIBLE_DEVICES=
sha256sum isaac_field/env.py isaac_field/newton_drive.py tests/test_front_guide_native.py > "$OUT/source_hashes_before.sha256"
bash scripts/isaaclab_python.sh -m pytest -q > "$OUT/full_cpu_pytest.log" 2>&1
pytest_rc=$?
printf '%s\n' "$pytest_rc" > "$OUT/pytest.exit_code"
bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts tests > "$OUT/compileall.log" 2>&1
compile_rc=$?
printf '%s\n' "$compile_rc" > "$OUT/compileall.exit_code"
sha256sum -c "$OUT/source_hashes_before.sha256" > "$OUT/source_hashes_after.log" 2>&1
hash_rc=$?
printf '%s\n' "$hash_rc" > "$OUT/source_hashes.exit_code"
printf '{"pytest":%s,"compileall":%s,"source_hashes":%s}\n' "$pytest_rc" "$compile_rc" "$hash_rc" > "$OUT/validation.exit.json"
if [[ "$pytest_rc" -ne 0 || "$compile_rc" -ne 0 || "$hash_rc" -ne 0 ]]; then exit 1; fi
exit 0