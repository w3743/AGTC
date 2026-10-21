"""Build a deterministic online deployment bundle from an explicit allowlist."""
import hashlib
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
files=[
    '使用说明.md','安装环境.cmd','启动实验室.cmd','检查环境.cmd',
    'config/versions.json','config/system-packages.txt','config/ros-packages.lock',
    'config/learning.lock','config/px4.lock',
    'scripts/windows.ps1','scripts/install.sh','scripts/sources.py','scripts/env.sh',
    'scripts/launch.sh','scripts/verify.py','scripts/generate_world.py','scripts/package.py',
    'simulation/common.py','simulation/run_lab.py','simulation/run_drone.py','simulation/run_lightstereo.py','simulation/compare_stereo.py','simulation/refine_stereo.py',
    'training/robot_learning.py','uwb_plugin/CMakeLists.txt',
]
version=json.loads((root/'config/versions.json').read_text())['package_version']
out=root/'dist'; out.mkdir(exist_ok=True)
target=out/f'robotics-lab-{version}.zip'
payload={name:(root/name).read_bytes() for name in sorted(files)}
manifest=''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n' for name,data in payload.items())
payload['SHA256SUMS']=manifest.encode('utf-8')
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for name,data in payload.items():
        info=zipfile.ZipInfo('robotics-lab/'+name,date_time=(2026,9,23,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        info.external_attr=0o100644<<16
        z.writestr(info,data)
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
digest=hashlib.sha256(target.read_bytes()).hexdigest()
(out/(target.name+'.sha256')).write_text(f'{digest}  {target.name}\n',encoding='ascii')
print(f'{target}\n{target.stat().st_size} bytes\nSHA256 {digest}')
