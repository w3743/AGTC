"""重新运行已有模型；输出只写本轮审查目录，不覆盖现行模块证据。"""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import unittest

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def read_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

result = {'date': '2026-10-29', 'boundary': '已有参考算法/理想计算复跑，不证明当前电路、固件或实物通过', 'checks': {}}
source = ROOT / '稳压模块/仿真/源码/V1.6-BUSFIX/cell_protection_qualification.py'
mod = read_module('review_busfix', source)
mod.OUTPUT = OUT / 'BUSFIX本轮复跑.json'
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    status = mod.main()
result['checks']['BUSFIX'] = {'source': str(source), 'sha256': sha(source), 'exitCode': status, 'stdout': stdout.getvalue()}

source = ROOT / '水泵驱动模块/检查/泵控制参考状态机.py'
mod = read_module('review_pump_model', source)
suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
log = io.StringIO()
test_result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
result['checks']['PUMP_STATE'] = {'source': str(source), 'sha256': sha(source), 'tests': test_result.testsRun,
    'failures': len(test_result.failures), 'errors': len(test_result.errors), 'passed': test_result.wasSuccessful(), 'log': log.getvalue()}

source = ROOT / '水泵驱动模块/检查/选型计算.py'
mod = read_module('review_pump_calc', source)
result['checks']['PUMP_CALC'] = {'source': str(source), 'sha256': sha(source), 'result': mod.calc()}
(OUT / '参考模型复跑.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(result['checks']['BUSFIX']['stdout'].strip())
print('PUMP_STATE', test_result.testsRun, 'PASS' if test_result.wasSuccessful() else 'FAIL')
