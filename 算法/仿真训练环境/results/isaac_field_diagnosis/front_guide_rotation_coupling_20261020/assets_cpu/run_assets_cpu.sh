#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
OUT_DIR="$ROOT_DIR/results/isaac_field_diagnosis/front_guide_rotation_coupling_20261020/assets_cpu"
LOG_PATH="$OUT_DIR/asset_tests.log"
EXIT_PATH="$OUT_DIR/asset_tests.exit.json"

if [[ -e "$LOG_PATH" || -e "$EXIT_PATH" ]]; then
  printf 'Refusing to overwrite previous CPU asset test output.\n' >&2
  exit 73
fi
cd "$ROOT_DIR"

ASSETS_SHA256="$(sha256sum isaac_field/assets.py | awk '{print $1}')"
TEST_SHA256="$(sha256sum tests/test_isaac_field_assets.py | awk '{print $1}')"
RUNNER_SHA256="$(sha256sum "$BASH_SOURCE" | awk '{print $1}')"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
COMMAND='CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m pytest -q tests/test_isaac_field_assets.py'

set +e
CUDA_VISIBLE_DEVICES= bash scripts/isaaclab_python.sh -m pytest -q tests/test_isaac_field_assets.py > "$LOG_PATH" 2>&1
ACTUAL_EXIT_CODE=$?
set -e
FINISHED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ASSETS_SHA256_AFTER="$(sha256sum isaac_field/assets.py | awk '{print $1}')"
TEST_SHA256_AFTER="$(sha256sum tests/test_isaac_field_assets.py | awk '{print $1}')"

printf '{"started_utc":"%s","finished_utc":"%s","command":"%s","argv":["scripts/isaaclab_python.sh","-m","pytest","-q","tests/test_isaac_field_assets.py"],"cpu_only":true,"cuda_visible_devices":"","actual_exit_code":%d,"status":"%s","assets_sha256_before":"%s","assets_sha256_after":"%s","test_sha256_before":"%s","test_sha256_after":"%s","runner_sha256":"%s","log":"asset_tests.log"}\n' \
  "$STARTED_UTC" "$FINISHED_UTC" "$COMMAND" "$ACTUAL_EXIT_CODE" \
  "$(if [[ $ACTUAL_EXIT_CODE -eq 0 ]]; then printf passed; else printf failed; fi)" \
  "$ASSETS_SHA256" "$ASSETS_SHA256_AFTER" "$TEST_SHA256" "$TEST_SHA256_AFTER" "$RUNNER_SHA256" \
  > "$EXIT_PATH"
exit "$ACTUAL_EXIT_CODE"
