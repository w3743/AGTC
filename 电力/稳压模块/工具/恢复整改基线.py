"""仅在自有EDA会话关闭后执行：恢复本轮误删前的三个原生文件。"""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

module=Path(__file__).resolve().parents[1]
root=module.parent
out=module/'检查/原理整改'
state=json.loads((out/'session.json').read_text(encoding='utf-8'))
sessions=json.loads(subprocess.check_output([r'D:\lceda-pro\lceda-pro.exe','session','list']).decode('utf-8-sig'))
assert sessions['ok']
assert not any(s['sessionId']==state['session'] and s.get('status')=='alive' for s in sessions['value']), '会话仍活跃，拒绝恢复'
rows=json.loads((out/'修改前SHA.json').read_text(encoding='utf-8'))
targets=[r for r in rows if Path(r['path']).suffix in ('.eprj3','.esch2','.ecfg')]
assert len(targets)==3
backup=out/'误删中间态备份.zip'
assert not backup.exists(), '中间态已备份；拒绝重复覆盖'
with zipfile.ZipFile(backup,'x',zipfile.ZIP_DEFLATED) as z:
    for row in targets:
        path=(root/row['path']).resolve()
        assert path.is_relative_to((module/'eda工程文件/sc-regulator').resolve())
        z.writestr(row['path'],path.read_bytes())
with zipfile.ZipFile(backup) as z:
    assert z.testzip() is None
with zipfile.ZipFile(out/'修改前源.zip') as z:
    original={r['path']:z.read(r['path']) for r in targets}
    assert all(hashlib.sha256(original[r['path']]).hexdigest()==r['sha256'] for r in targets)
for row in targets:
    (root/row['path']).write_bytes(original[row['path']])
proof=[dict(path=r['path'],expected_sha256=r['sha256'],
            actual_sha256=hashlib.sha256((root/r['path']).read_bytes()).hexdigest()) for r in targets]
assert all(r['expected_sha256']==r['actual_sha256'] for r in proof)
report=dict(status='NATIVE_BASELINE_RESTORED_NO_AFE_IMPLEMENTATION_CLAIM',
            reason='区域清理误触保留均衡网络，完整回退原生文件；新配置/计算/合同保留，后续改动须先审脚本。',
            files=proof,intermediate_backup=backup.name,
            intermediate_backup_sha256=hashlib.sha256(backup.read_bytes()).hexdigest())
(out/'误删回退验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=report['status'],restored_files=len(proof)),ensure_ascii=True))
