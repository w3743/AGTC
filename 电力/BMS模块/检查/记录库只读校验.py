"""Run the project library's read-only check and bind its current evidence."""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys

P = Path(__file__).resolve().parent
R = P.parent.parent
script = R / '元件封装库/工具/同步全部.py'
run = subprocess.run([sys.executable, '-B', str(script), '--check'], cwd=R, capture_output=True)
output = run.stdout.decode('utf-8-sig', errors='replace')
lines = [line for line in output.splitlines() if line.strip().startswith('{')]
result = json.loads(lines[-1]) if lines else {}
files = [R / x for x in ['元件封装库/资产索引.json', '元件封装库/检查/同步证据.json', '元件封装库/检查/共享仓库发布.json', 'BMS模块/原理图准备/BOM.csv']]
files += [f for f in (P.parent / 'eda工程文件').rglob('*') if f.is_file()]
report = {'time': datetime.datetime.now().astimezone().isoformat(), 'syncAlreadyRun': True,
          'readOnlyCommand': 'python -B 元件封装库/工具/同步全部.py --check',
          'exitCode': run.returncode, 'result': result,
          'evidence': [{'path': str(f.relative_to(R)).replace('\\', '/'), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]}
(P / '库同步校验.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
assert run.returncode == 0 and result.get('pass') is True, report
print(json.dumps({'exitCode': run.returncode, **result}, ensure_ascii=False))
