#!/usr/bin/env bash
set -eo pipefail
if [ -z "$PREP_DIR" ] || [ -z "$ATTEMPT_ROOT" ] || [ -z "$AGTC_NWORLD" ]; then
  echo "set PREP_DIR, ATTEMPT_ROOT, and AGTC_NWORLD" >&2
  exit 2
fi
if [ -z "$PYTHON" ]; then PYTHON="/opt/venv_mujoco315_smoke/bin/python"; fi
if [ -z "$REPO_PROTO" ]; then REPO_PROTO="/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"; fi
case "$AGTC_NWORLD" in 1|3|64) ;; *) echo "AGTC_NWORLD must be 1, 3, or 64" >&2; exit 2 ;; esac
mkdir -p "$ATTEMPT_ROOT"
if [ -e "$ATTEMPT_ROOT/result" ]; then echo "result path already exists: $ATTEMPT_ROOT/result" >&2; exit 2; fi
cp "$PREP_DIR/manifest.json" "$ATTEMPT_ROOT/input_snapshot_manifest.json"
cp "$PREP_DIR/preregistration.json" "$ATTEMPT_ROOT/preregistration.json"
if ! cmp -s "$PREP_DIR/manifest.json" "$ATTEMPT_ROOT/input_snapshot_manifest.json"; then echo "input manifest copy mismatch" >&2; exit 2; fi
set +e
PYTHONUNBUFFERED=1 "$PYTHON" -u "$PREP_DIR/probe/probe_mjw_v6_gate.py" \
  --repo-proto "$REPO_PROTO" --nworld "$AGTC_NWORLD" --steps 20 \
  --device cuda:0 \
  --input-manifest "$ATTEMPT_ROOT/input_snapshot_manifest.json" \
  --preregistration "$ATTEMPT_ROOT/preregistration.json" \
  --input-snapshot-root "$PREP_DIR/input_snapshot/source" \
  --output-dir "$ATTEMPT_ROOT/result" \
  >"$ATTEMPT_ROOT/stdout.log" 2>"$ATTEMPT_ROOT/stderr.log"
rc=$?
set -e
printf '%s\n' "$rc" >"$ATTEMPT_ROOT/exit_code.txt"
exit "$rc"