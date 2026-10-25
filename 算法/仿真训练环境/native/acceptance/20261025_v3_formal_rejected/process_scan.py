import json, os, time
from pathlib import Path
base=Path('/mnt/c/pioneer/agtc-native-acceptance-backup/fixed_physics_v3_prep_20261025_184052')
out=base/'process_scan.json'
target={655:'controller_numeric_pid',814:'wrapper',827:'trainer'}
records=[]
for pid, label in target.items():
    root=Path('/proc')/str(pid)
    if root.exists():
        try:
            comm=(root/'comm').read_text().strip()
            status=(root/'status').read_text().splitlines()
            ppid=next((int(x.split()[1]) for x in status if x.startswith('PPid:')),None)
            records.append({'pid':pid,'label':label,'present':True,'comm':comm,'ppid':ppid})
        except Exception as exc: records.append({'pid':pid,'label':label,'present':True,'read_error':type(exc).__name__})
    else: records.append({'pid':pid,'label':label,'present':False})
patterns={'native/train/train_wsl.py':'training_entry','llama-server':'llama_server','/fixed_physics_v3_prep_20261025_184052/launch.py':'this_trial_controller'}
matches=[]
for item in Path('/proc').iterdir():
    if not item.name.isdigit(): continue
    pid=int(item.name)
    try:
        comm=(item/'comm').read_text().strip()
        cmd=(item/'cmdline').read_bytes().decode('utf-8','replace').replace('\0',' ')
        status=(item/'status').read_text().splitlines()
        ppid=next((int(x.split()[1]) for x in status if x.startswith('PPid:')),None)
    except Exception: continue
    labels=[label for pattern,label in patterns.items() if pattern in cmd or (label=='llama_server' and pattern in comm)]
    if labels: matches.append({'pid':pid,'ppid':ppid,'comm':comm,'matched_labels':labels})
now=time.time()
import datetime
utc=datetime.datetime.fromtimestamp(now,datetime.timezone.utc)
summary={'schema':'fixed_physics_v3_process_scan_v1','sampled_at_utc':utc.isoformat(timespec='seconds').replace('+00:00','Z'),'sampled_at_beijing':utc.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'target_pid_observations':records,'matched_background_task_processes':matches,'cmdline_policy':'Read privately for matching only; no command lines or arguments are recorded.','interpretation':'Numeric PID 655 is currently login, not evidence that the original controller remains; wrapper/trainer IDs are absent. The final existing trial monitor and GPU sampler records are separately preserved.'}
out.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps({'written':str(out),'target_pid_observations':records,'matched_background_task_processes':matches},sort_keys=True))