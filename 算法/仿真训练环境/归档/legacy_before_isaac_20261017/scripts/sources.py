"""Fetch pinned, clean source trees; do not reset somebody's local edits."""
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
runtime=Path(os.environ.get('ROBOTICS_HOME',str(Path.home()/'robotics-lab')))
versions=json.loads((root/'config/versions.json').read_text())
def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
for key,name in [('px4','PX4-Autopilot'),('uwb','UWBPX4Sim')]:
    spec=versions[key]; path=runtime/'src'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        subprocess.run(['git','init',str(path)],check=True)
        git(path,'remote','add','origin',spec['url'])
    if not (path/'.git').exists():
        raise SystemExit(f'{path} 已存在但不是 Git 仓库，请另设 ROBOTICS_HOME。')
    # An interrupted first fetch leaves an empty repository. A rerun can resume it.
    head=subprocess.run(['git','-C',str(path),'rev-parse','--verify','HEAD'],capture_output=True,text=True)
    if head.returncode:
        if git(path,'remote','get-url','origin')!=spec['url'] or git(path,'status','--porcelain'):
            raise SystemExit(f'{path} 不是安装器创建的空仓库，请另设 ROBOTICS_HOME。')
        git(path,'fetch','--depth','1','origin',spec['commit'])
        git(path,'checkout','--detach',spec['commit'])
    elif head.stdout.strip()!=spec['commit'] or git(path,'status','--porcelain','--untracked-files=no'):
        raise SystemExit(f'{path} 版本不符或有本地修改。请另设 ROBOTICS_HOME；安装器不会强制重置。')
    print(name,spec['commit'],flush=True)
px4=runtime/'src/PX4-Autopilot'
submodules=['Tools/simulation/gz','src/lib/events/libevents','src/modules/mavlink/mavlink',
    'src/modules/uxrce_dds_client/Micro-XRCE-DDS-Client','src/lib/heatshrink/heatshrink',
    'src/lib/cdrstream/cyclonedds','src/lib/cdrstream/rosidl']
subprocess.run(['git','-C',str(px4),'submodule','update','--init','--recursive','--depth','1',*submodules],check=True)
