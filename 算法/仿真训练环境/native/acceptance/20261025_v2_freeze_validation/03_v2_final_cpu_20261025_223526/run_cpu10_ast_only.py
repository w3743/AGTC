from __future__ import annotations
import datetime, hashlib, json, os, re, subprocess, sys
from pathlib import Path
ROOT = Path(r"C:\pioneer\先进通用履带底盘工程")
NATIVE = ROOT / "算法" / "仿真训练环境" / "native"
OUT = Path(r"C:\pioneer\agtc-native-acceptance-backup\audit_stack\v2_final_cpu_20261025_223526")
PIN = "83db5a6e512d7d7244bc47ddf2059720c757504270e9bbf9129d0c7d8f92f87e"
REL = [
    "train/PROTOCOL_NATIVE_V2.md",
    "train/train_wsl.py",
    "proto/newton_backend.py",
    "proto/newton_mujoco_backend.py",
    "proto/physics_backend.py",
    "proto/test_newton_backend_isolation.py",
    "proto/test_newton_mujoco_backend.py",
]
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def snapshot():
    rows=[]
    for rel in REL:
        p=NATIVE/Path(rel)
        b=p.read_bytes()
        rows.append({"path":rel,"size":len(b),"sha256":hashlib.sha256(b).hexdigest()})
    agg=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"pinned_protocol_sha256":PIN,"files":rows,"selected_source_set_sha256":agg}
def run(label, argv):
    start=utc()
    proc=subprocess.run(argv,cwd=str(ROOT),capture_output=True,check=False)
    (OUT/f"{label}.stdout.log").write_bytes(proc.stdout or b"")
    (OUT/f"{label}.stderr.log").write_bytes(proc.stderr or b"")
    (OUT/f"{label}.exit_code.txt").write_text(str(proc.returncode)+"\n",encoding="ascii")
    rec={"label":label,"argv":argv,"started_utc":start,"finished_utc":utc(),"exit_code":proc.returncode,
         "stdout_bytes":len(proc.stdout or b""),"stderr_bytes":len(proc.stderr or b""),
         "stdout_sha256":hashlib.sha256(proc.stdout or b"").hexdigest(),
         "stderr_sha256":hashlib.sha256(proc.stderr or b"").hexdigest()}
    m=re.search(rb"Ran\s+(\d+)\s+tests?",(proc.stdout or b"")+(proc.stderr or b""))
    if m: rec["unittest_count"]=int(m.group(1))
    (OUT/f"{label}.command.json").write_text(json.dumps(rec,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return rec
before=snapshot()
(OUT/"focused_input_snapshot_before.json").write_text(json.dumps(before,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
prefix=["wsl.exe","--exec","env","CUDA_VISIBLE_DEVICES=","PYTHONDONTWRITEBYTECODE=1"]
native="/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native"
runs=[]
runs.append(run("cpu10_newton_backends",prefix+["/opt/venv_mujoco315_smoke/bin/python","-m","unittest","discover","-s",native+"/proto","-p","test_newton*backend*.py"]))
runs.append(run("ast_live_state_fakeenv",prefix+["/opt/venv/bin/python","/mnt/c/pioneer/agtc-native-acceptance-backup/audit_stack/v2_final_cpu_20261025_223526/ast_live_state_smoke.py"]))
after=snapshot()
(OUT/"focused_input_snapshot_after.json").write_text(json.dumps(after,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
summary={"pinned_protocol_sha256":PIN,"source_set_before_sha256":before["selected_source_set_sha256"],
         "source_set_after_sha256":after["selected_source_set_sha256"],
         "source_set_unchanged":before["selected_source_set_sha256"]==after["selected_source_set_sha256"],
         "runs":runs,"cpu10_count_ok":runs[0].get("unittest_count")==10,
         "ast_fakeenv_pass":runs[1]["exit_code"]==0}
(OUT/"focused_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
raise SystemExit(0 if summary["source_set_unchanged"] and all(x["exit_code"]==0 for x in runs) and summary["cpu10_count_ok"] else 1)