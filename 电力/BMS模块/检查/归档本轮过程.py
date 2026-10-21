"""Archive this turn's closed work copy and process files; verify before removal."""
from pathlib import Path
import datetime, hashlib, json, zipfile

P = Path(__file__).resolve().parent
M = P.parent.resolve()
O = P / '电源调速精简落实'
sha = lambda b: hashlib.sha256(b).hexdigest()
keep = {'expected.json', 'final_reopened.enet', 'reopen_check.result.json', '原生发布校验.json',
        '独立预期与重开网表核对.json', 'BMS_current_raw.pdf', 'PSU5_EXT_细节.png',
        'U_AND2_细节.png', 'J_FAN_CTRL_细节.png', 'Q_FAN_PWM_细节.png',
        'R_FAN_PWM_GATE_细节.png', 'SW_RUN_STORE_细节.png'}
files = [f for f in O.iterdir() if f.is_file() and f.name not in keep and not f.name.endswith('_backup.json')]
work = P / 'tmp/bms_implementation_work'
files += [f for f in work.rglob('*') if f.is_file()]
for name in ['落实电源与调速.py', '完成本轮落地.py', '发布最终标注.py', '写入本轮交接状态.py', '更新本轮接口文档.py']:
    f = P / name
    if f.is_file(): files.append(f)
for name in ['bom_before_implementation.zip', 'source_before_implementation.zip',
             '5V电源预算_before_fan_implementation_20261015.md', '温控调速落实验证_before_refresh_20261015.json',
             'solar_regen_three_files_preedit_20261015.zip', 'solar_regen_three_files_preedit_20261015.sha256.json']:
    f = P / 'tmp' / name
    if f.is_file(): files.append(f)
f = M / 'tmp/HLK_原理图准备文档_编辑前_20261015.zip'
if f.is_file(): files.append(f)
files = list(dict.fromkeys(f.resolve() for f in files))
for f in files:
    assert f.is_relative_to(M) and not f.is_symlink(), f
    assert not f.is_relative_to(M / 'eda工程文件'), f
before = {str(f.relative_to(M)): sha(f.read_bytes()) for f in (M / 'eda工程文件').rglob('*') if f.is_file()}
key = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_电源调速精简过程'
archive = M / '归档/历史与调试.zip'
records = []
with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED) as z:
    for f in files:
        b = f.read_bytes()
        name = key + '/' + f.relative_to(M).as_posix()
        z.writestr(name, b)
        assert sha(z.read(name)) == sha(b), name
        records.append({'source': f.relative_to(M).as_posix(), 'archiveEntry': name, 'sha256': sha(b)})
# Independent reopen/readback, then delete only the already verified exact files.
with zipfile.ZipFile(archive) as z:
    for record, f in zip(records, files):
        assert sha(z.read(record['archiveEntry'])) == record['sha256'] == sha(f.read_bytes())
for f in files: f.unlink()
if work.exists():
    for d in sorted([x for x in work.rglob('*') if x.is_dir()], key=lambda x: len(x.parts), reverse=True):
        assert d.resolve().is_relative_to(work.resolve())
        d.rmdir()
    work.rmdir()
after = {str(f.relative_to(M)): sha(f.read_bytes()) for f in (M / 'eda工程文件').rglob('*') if f.is_file()}
assert before == after
report = {'time': datetime.datetime.now().astimezone().isoformat(), 'archive': '归档/历史与调试.zip',
          'verifiedReadbackBeforeRemoval': True, 'formalNativeUnchanged': True,
          'scope': '本轮已关闭工作副本及明确属于本轮的过程文件；未整理其他tmp子树', 'files': records}
(P / '本轮过程归档.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({'archivedFiles': len(records), 'readback': 'PASS', 'formalNativeUnchanged': True}, ensure_ascii=False))
