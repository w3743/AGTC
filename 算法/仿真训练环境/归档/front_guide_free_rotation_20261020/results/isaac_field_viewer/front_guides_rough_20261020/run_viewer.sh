#!/usr/bin/env bash
set -uo pipefail

VIEWER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$VIEWER_DIR/../../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="$VIEWER_DIR/viewer_replay"
CMD=("$ROOT_DIR/scripts/isaaclab_python.sh" -m isaac_field viewer
  --mode goto --plants 0 --rock-density 0 --front-guides --body-stability
  --max-time 60 --seed 80003 --episode-seed 87001
  --reward-profile protect_plants --port 8781 --output "$OUTPUT_DIR")

existing_path=""
for candidate in "$OUTPUT_DIR" "$VIEWER_DIR/viewer.log" "$VIEWER_DIR/viewer.pid" \
    "$VIEWER_DIR/launcher.pid" "$VIEWER_DIR/viewer.argv.log" "$VIEWER_DIR/viewer.exit.json"; do
  if [[ -e "$candidate" || -L "$candidate" ]]; then
    existing_path="$candidate"
    break
  fi
done

if [[ -n "$existing_path" ]]; then
  refusal_path="$VIEWER_DIR/refusal_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
  {
    printf 'refused=1\nreason=existing artifact; refusing to overwrite\n'
    printf 'existing_path=%q\nlauncher_pid=%s\n' "$existing_path" "$$"
    printf 'intended_command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
  } > "$refusal_path"
  printf 'Refused to launch because an artifact already exists: %s\n' "$existing_path" >&2
  exit 2
fi

{
  printf 'launcher_pid=%s\n' "$$"
  printf 'launcher_executable=%q\n' "${CMD[0]}"
  printf 'python_argc=%s\n' "$(( ${#CMD[@]} - 1 ))"
  for ((i = 1; i < ${#CMD[@]}; i++)); do
    printf 'python_argv[%s]=%q\n' "$((i - 1))" "${CMD[$i]}"
  done
} > "$VIEWER_DIR/viewer.argv.log"

"${CMD[@]}" > "$VIEWER_DIR/viewer.log" 2>&1 &
python_pid=$!
printf '%s\n' "$$" > "$VIEWER_DIR/launcher.pid"
printf '%s\n' "$python_pid" > "$VIEWER_DIR/viewer.pid"

wait "$python_pid"
viewer_rc=$?
printf '{"actual_exit_code":%s,"launcher_pid":%s,"python_pid":%s}\n' \
  "$viewer_rc" "$$" "$python_pid" > "$VIEWER_DIR/viewer.exit.json"
exit "$viewer_rc"
