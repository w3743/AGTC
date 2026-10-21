#!/usr/bin/env bash
set -uo pipefail

ATTEMPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$ATTEMPT_DIR/../../../.." && pwd)"
TRIAL_DIR="$ROOT_DIR/results/isaac_field_improvement/random_global_rough_20261020"
RUNNER="$ROOT_DIR/scripts/train_isaac_foundation_fresh.py"
PYTHON_WRAPPER="$ROOT_DIR/scripts/isaaclab_python.sh"
cd "$ROOT_DIR"

for artifact in "$TRIAL_DIR" "$ATTEMPT_DIR/commands.txt" \
    "$ATTEMPT_DIR/prepare.stdout.log" "$ATTEMPT_DIR/prepare.exit_code" \
    "$ATTEMPT_DIR/preflight.stdout.log" "$ATTEMPT_DIR/preflight.exit_code" \
    "$ATTEMPT_DIR/prepare_preflight.exit_code"; do
  if [[ -e "$artifact" || -L "$artifact" ]]; then
    refusal="$ATTEMPT_DIR/refusal_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
    {
      printf 'refused=1\nreason=existing output or attempt artifact; refusing overwrite\n'
      printf 'existing_path=%q\n' "$artifact"
      printf 'trial_output=%q\n' "$TRIAL_DIR"
    } > "$refusal"
    printf 'Refusing to overwrite existing output or attempt artifact: %s\n' "$artifact" >&2
    exit 2
  fi
done

cat > "$ATTEMPT_DIR/commands.txt" <<EOF
prepare: env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= "$PYTHON_WRAPPER" "$RUNNER" --prepare --profile random_rough --output "$TRIAL_DIR"
preflight: env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= "$PYTHON_WRAPPER" "$RUNNER" --preflight --output "$TRIAL_DIR"
Scope: prepare and preflight only; no launch, model creation, training, viewer, or physics simulation.
EOF

if env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= \
    "$PYTHON_WRAPPER" "$RUNNER" --prepare --profile random_rough --output "$TRIAL_DIR" \
    > "$ATTEMPT_DIR/prepare.stdout.log" 2>&1; then
  prepare_rc=0
else
  prepare_rc=$?
fi
printf '%s\n' "$prepare_rc" > "$ATTEMPT_DIR/prepare.exit_code"
if [[ "$prepare_rc" -ne 0 ]]; then
  printf '%s\n' "$prepare_rc" > "$ATTEMPT_DIR/prepare_preflight.exit_code"
  exit "$prepare_rc"
fi

if env -u RUN_ISAAC_PHYSICS CUDA_VISIBLE_DEVICES= \
    "$PYTHON_WRAPPER" "$RUNNER" --preflight --output "$TRIAL_DIR" \
    > "$ATTEMPT_DIR/preflight.stdout.log" 2>&1; then
  preflight_rc=0
else
  preflight_rc=$?
fi
printf '%s\n' "$preflight_rc" > "$ATTEMPT_DIR/preflight.exit_code"
printf '%s\n' "$preflight_rc" > "$ATTEMPT_DIR/prepare_preflight.exit_code"
exit "$preflight_rc"
