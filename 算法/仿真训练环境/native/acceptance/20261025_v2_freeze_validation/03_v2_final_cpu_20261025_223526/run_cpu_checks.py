from __future__ import annotations
import datetime, hashlib, json, os, re, subprocess, sys, traceback
from pathlib import Path

ROOT = Path(r"C:\pioneer\先进通用履带底盘工程")
NATIVE = ROOT / "算法" / "仿真训练环境" / "native"
OUT = Path(r"C:\pioneer\agtc-native-acceptance-backup\audit_stack\v2_final_cpu_20261025_223526")
PINNED_PROTOCOL_SHA256 = "83db5a6e512d7d7244bc47ddf2059720c757504270e9bbf9129d0c7d8f92f87e"
def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()
def inputs():
    files = []
    for p in NATIVE.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(NATIVE)
        if any(x in ("acceptance", "runs", "__pycache__") for x in rel.parts):
            continue
        if p.suffix == ".py" or rel.as_posix() == "train/PROTOCOL_NATIVE_V2.md":
            files.append((rel.as_posix(), sha(p), p.stat().st_size))
    files.sort()
    canonical = json.dumps(files, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"pinned_protocol_sha256": PINNED_PROTOCOL_SHA256,
            "protocol_file_raw_sha256": next((x[1] for x in files if x[0] == "train/PROTOCOL_NATIVE_V2.md"), None),
            "files": [{"path": p, "sha256": h, "size": size} for p,h,size in files],
            "source_set_sha256": hashlib.sha256(canonical).hexdigest()}
def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def run(label, argv, cwd, env=None, timeout=900):
    rec = {"label": label, "argv": [str(x) for x in argv], "cwd": str(cwd),
           "started_utc": utcnow(), "timeout_seconds": timeout}
    try:
        p = subprocess.run([str(x) for x in argv], cwd=str(cwd), env=env,
                           capture_output=True, timeout=timeout, check=False)
        stdout, stderr, code = p.stdout or b"", p.stderr or b"", p.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        code = 124
        rec["timeout"] = True
    except BaseException as exc:
        stdout, stderr, code = b"", traceback.format_exc().encode("utf-8"), 125
        rec["launcher_exception"] = f"{type(exc).__name__}: {exc}"
    rec["finished_utc"] = utcnow()
    rec["exit_code"] = code
    rec["stdout_bytes"] = len(stdout)
    rec["stderr_bytes"] = len(stderr)
    rec["stdout_sha256"] = hashlib.sha256(stdout).hexdigest()
    rec["stderr_sha256"] = hashlib.sha256(stderr).hexdigest()
    (OUT / f"{label}.stdout.log").write_bytes(stdout)
    (OUT / f"{label}.stderr.log").write_bytes(stderr)
    (OUT / f"{label}.exit_code.txt").write_text(f"{code}\n", encoding="ascii")
    save_json(f"{label}.command.json", rec)
    combined = stdout + b"\n" + stderr
    m = re.search(rb"Ran\s+(\d+)\s+tests?", combined)
    sk = re.search(rb"\(([^\r\n]*\bskipped\b[^\r\n]*)\)", combined)
    if m:
        rec["unittest_count"] = int(m.group(1))
    if sk:
        rec["skip_summary"] = sk.group(1).decode("utf-8", errors="replace")
    save_json(f"{label}.command.json", rec)
    print(f"{label}: exit={code}, stdout={len(stdout)} bytes, stderr={len(stderr)} bytes, tests={rec.get('unittest_count')}, skips={rec.get('skip_summary')}")
    return rec

# Persist execution environment and exact input hashes before any tests run.
before = inputs()
save_json("input_snapshot_before.json", before)
env_win = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
env_wsl = ["wsl.exe", "--exec", "env", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1"]
wsl_native = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native"
wsl_out = "/mnt/c/pioneer/agtc-native-acceptance-backup/audit_stack/v2_final_cpu_20261025_223526"
runs = []
runs.append(run("windows_tools_61", [sys.executable, "-m", "unittest", "discover", "-s",
    str(NATIVE / "tools"), "-p", "test_*.py"], ROOT, env_win))
runs.append(run("wsl_tools_61", env_wsl + ["/opt/venv/bin/python", "-m", "unittest", "discover", "-s",
    wsl_native + "/tools", "-p", "test_*.py"], ROOT))
runs.append(run("wsl_nonfinite_guards", env_wsl + ["/opt/venv/bin/python", wsl_out + "/nonfinite_smoke.py"], ROOT))
runs.append(run("cpu10_newton_backends", env_wsl + ["/opt/venv_mujoco315_smoke/bin/python", "-m", "unittest",
    "discover", "-s", wsl_native + "/proto", "-p", "test_newton*backend*.py"], ROOT))
runs.append(run("ast_live_state_fakeenv", env_wsl + ["/opt/venv/bin/python", wsl_out + "/ast_live_state_smoke.py"], ROOT))

after = inputs()
save_json("input_snapshot_after.json", after)
summary = {
    "pinned_protocol_sha256": PINNED_PROTOCOL_SHA256,
    "source_set_before_sha256": before["source_set_sha256"],
    "source_set_after_sha256": after["source_set_sha256"],
    "source_set_unchanged": before["source_set_sha256"] == after["source_set_sha256"],
    "runs": runs,
    "all_processes_exit_zero": all(x["exit_code"] == 0 for x in runs),
    "expected_tools_tests": 61,
    "windows_tools_count_ok": next((x.get("unittest_count") == 61 for x in runs if x["label"] == "windows_tools_61"), False),
    "wsl_tools_count_ok": next((x.get("unittest_count") == 61 for x in runs if x["label"] == "wsl_tools_61"), False),
    "cpu10_count_ok": next((x.get("unittest_count") == 10 for x in runs if x["label"] == "cpu10_newton_backends"), False),
}
save_json("summary.json", summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not summary["all_processes_exit_zero"] or not summary["source_set_unchanged"]:
    raise SystemExit(1)