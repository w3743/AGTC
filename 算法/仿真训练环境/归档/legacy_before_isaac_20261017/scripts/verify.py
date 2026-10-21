#!/usr/bin/env python3
"""Local deployment checks. Report scope explicitly; no hidden long training."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
runtime=Path(os.environ.get('ROBOTICS_HOME',str(Path.home()/'robotics-lab')))
p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); args=p.parse_args()
report={'package_version':'1.0.0','checked_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'scope':'versions, binaries, graphics; optional arm/hand reset-step-render checks; not full flight/mapping or trained-policy evaluation','checks':[]}

def check(name,fn):
    try:
        detail=fn()
        report['checks'].append(dict(name=name,status='PASS',detail=str(detail)))
        print('PASS',name,flush=True)
    except Exception as exc:
        report['checks'].append(dict(name=name,status='FAIL',detail=str(exc)[-2000:]))
        print('FAIL',name,str(exc)[-500:],flush=True)

def run(cmd,clean=False,timeout=60):
    env=os.environ.copy()
    if clean:
        env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
    result=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,env=env,timeout=timeout)
    if result.returncode: raise RuntimeError((result.stdout+result.stderr)[-1500:])
    return result.stdout.strip()

def system():
    info=platform.freedesktop_os_release()
    assert info['ID']=='ubuntu' and info['VERSION_ID']=='24.04'
    assert platform.machine()=='x86_64' and sys.version_info[:2]==(3,12)
    return 'Ubuntu 24.04 / x86_64 / Python 3.12'

def ros_versions():
    expected=(root/'config/ros-packages.lock').read_text().splitlines()
    names=[x.split('=')[0] for x in expected]
    actual=set(run(['dpkg-query','-W','-f=${Package}=${Version}\n',*names]).splitlines())
    missing=set(expected)-actual
    if missing: raise RuntimeError('版本不符：'+', '.join(sorted(missing)[:10]))
    return f'{len(expected)} ROS packages match lock'

def python_versions(kind):
    py=runtime/'venv'/kind/'bin/python'
    actual=json.loads(run([py,'-m','pip','list','--format=json'],clean=True))
    versions={x['name'].lower().replace('_','-'):x['version'] for x in actual}
    mismatch=[]
    for line in (root/'config'/f'{kind}.lock').read_text().splitlines():
        if not line or line.startswith('#'): continue
        name,version=line.split('==',1)
        if versions.get(name.lower().replace('_','-'))!=version: mismatch.append(line)
    if mismatch: raise RuntimeError('版本不符：'+', '.join(mismatch))
    run([py,'-m','pip','check'],clean=True)
    return 'Locked dependencies installed; pip check passed'

def binaries():
    for file in [runtime/'build/uwb/libUWBGazeboPlugin.so',runtime/'src/PX4-Autopilot/build/px4_sitl_default/bin/px4']:
        if not file.is_file(): raise RuntimeError('缺少 '+str(file))
        if 'not found' in run(['ldd',file]): raise RuntimeError('动态库缺失 '+str(file))
    return run(['gz','sim','--versions'])

def sources():
    versions=json.loads((root/'config/versions.json').read_text())
    for key,folder in [('px4','PX4-Autopilot'),('uwb','UWBPX4Sim')]:
        actual=run(['git','-C',runtime/'src'/folder,'rev-parse','HEAD'])
        assert actual==versions[key]['commit'], folder+' commit mismatch'
    return 'PX4 and UWB commits match'

check('system',system)
check('ros_versions',ros_versions)
check('learning_versions',lambda:python_versions('learning'))
check('px4_python_versions',lambda:python_versions('px4'))
check('sources',sources)
check('simulator_binaries',binaries)
check('graphics',lambda:run(['glxinfo','-B']))
if args.smoke:
    for task in ['arm','hand']:
        check(task+'_smoke',lambda task=task:run([runtime/'venv/learning/bin/python',root/'training/robot_learning.py','--task',task,'--mode','smoke','--output',root/'results/learning'],clean=True)[-500:])
report['passed']=all(x['status']=='PASS' for x in report['checks'])
(root/'results').mkdir(exist_ok=True)
(root/'results/deployment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('基础检查通过。' if report['passed'] else '检查未通过，请看 results/deployment.json。')
sys.exit(0 if report['passed'] else 1)
