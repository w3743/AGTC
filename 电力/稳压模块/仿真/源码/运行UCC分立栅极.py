"""运行TI未加密PSpice宏模型，保存仿真输出和含模型限制的摘要。"""
from pathlib import Path
import csv,hashlib,json,shutil,subprocess,tempfile,zipfile
ROOT=Path(__file__).resolve().parents[2];POWER=ROOT.parent
exe=POWER/'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
src=ROOT/'仿真/源码/UCC27511_分立栅极负载.cir';package=ROOT/'仿真/输入/厂家模型/slum488.zip'
with tempfile.TemporaryDirectory(prefix='uccprobe_') as tmp:
    d=Path(tmp);(d/'.spiceinit').write_text('set ngbehavior=ps\n',encoding='ascii')
    shutil.copyfile(src,d/'case.cir')
    with zipfile.ZipFile(package) as z:(d/'ucc.lib').write_bytes(z.read('UCC27511_TRANS.LIB'))
    p=subprocess.run([str(exe.resolve()),'-b','case.cir'],cwd=d,capture_output=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
    log=p.stdout.decode('utf-8','replace')+p.stderr.decode('utf-8','replace')
    path=d/'ucc_gate.txt'
    if p.returncode or 'Error:' in log or not path.is_file():raise RuntimeError(log[-3000:])
    rows=[list(map(float,l.split())) for l in path.read_text().splitlines() if len(l.split())>=10]
    if not rows or rows[-1][0]<11.9e-6:raise RuntimeError('波形不完整')
    samples={str(us):min(rows,key=lambda x:abs(x[0]-us*1e-6))[5] for us in (.5,2,4,6,7.2,8,10,12)}
    # 检验三段有效供电下的逻辑极性；掉电行为只报告不假定通过。
    checks={'default_off':samples['0.5']<1,'request_on':samples['2']>10,'request_release_off':samples['4']<1,'second_request_on':samples['6']>10}
    shutil.copyfile(path,ROOT/'仿真/结果/UCC27511_分立栅极波形.txt')
    report={'status':'TI_MACROMODEL_SIMULATED_APPROXIMATE_LINEAR_GATE_LOAD','checks':checks,'gate_V_at_us':samples,
      'inputs':{'model':'TI SLUM488 UCC27511_TRANS','model_zip_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),
      'gate_resistors_ohm':[10,10],'gate_pulldown_ohm':10000,'constant_gate_capacitance_F':11.36e-9},
      'limitations':['Ciss使用固定11.36nF近似，非IRLB4030电压相关电容与Miller模型',
                     'IN−由理想电压源驱动，不覆盖NPN实际电路','PSpice宏模型在ngspice兼容模式运行，低于模型有效范围的掉电结果不得外推',
                     '未连接真实功率MOS、BUS、线束及制动电阻，无法证明制动支路关断']}
    (ROOT/'仿真/结果/UCC27511_分立栅极汇总.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    if not all(checks.values()):raise SystemExit('UCC逻辑测试有失败，请检查')
