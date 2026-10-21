#!/usr/bin/env bash
set -u
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$RUN_DIR/../../../../.." && pwd)"
LOG="$RUN_DIR/paired_comparison_cpu.log"
EXIT_JSON="$RUN_DIR/paired_comparison_cpu.exit.json"
if [[ -e "$LOG" || -e "$EXIT_JSON" ]]; then
  printf 'Refusing to overwrite prior CPU summary records.\n' >&2
  exit 2
fi
cd "$REPO_ROOT"
set +e
python3 "$RUN_DIR/summarize_paired_comparison.py" >"$LOG" 2>&1
rc=$?
set -e
python3 - "$rc" "$EXIT_JSON" "$RUN_DIR/summarize_paired_comparison.py" <<'PY'
import datetime, hashlib, json, pathlib, sys
code, output, script = int(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
output.write_text(json.dumps({
    'actual_exit_code': code,
    'command': 'python3 summarize_paired_comparison.py',
    'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
    'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'outputs': ['paired_comparison.json', 'paired_comparison.md'],
    'log': 'paired_comparison_cpu.log',
    'gpu_used': False,
    'production_modified': False,
}, indent=2) + "\n", encoding='utf-8')
PY
exit "$rc"
