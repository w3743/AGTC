"""参数化低边界等效输出网络；不推断掉电保证。"""
from pathlib import Path
import json,subprocess,tempfile,shutil
ROOT=Path(__file__).resolve().parents[2];EXE=ROOT.parent/'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
base=(ROOT/'仿真/源码/LTC7001_等效栅极筛查.cir').read_text(encoding='utf-8');cases=[]
with tempfile.TemporaryDirectory(prefix='ltceq_') as tmp:
 d=Path(tmp)
 for cb_nf,pump_uA in [(500,0),(500,15),(1000,0),(1000,15)]:
  s=base.replace('Cbst bst 0 500n',f'Cbst bst 0 {cb_nf}n').replace('Icp 0 bst 15u',f'Icp 0 bst {pump_uA}u')
  (d/'case.cir').write_text(s,encoding='ascii');wave=d/'ltc_wave.txt';wave.unlink(missing_ok=True)
  p=subprocess.run([str(EXE.resolve()),'-b','case.cir'],cwd=d,capture_output=True,timeout=35,creationflags=subprocess.CREATE_NO_WINDOW)
  log=p.stdout.decode('utf-8','replace')+p.stderr.decode('utf-8','replace')
  if p.returncode or 'Error:' in log or not wave.exists():raise RuntimeError(log[-2000:])
  rows=[list(map(float,l.split())) for l in wave.read_text().splitlines() if len(l.split())>=6]
  if not rows or rows[-1][0]<.002999:raise RuntimeError('仿真不完整')
  active=[x for x in rows if 10.1e-6<x[0]<.001999];rise=next((x[0]-10e-6 for x in active if x[3]>=8),None)
  on_last=active[-1];release=next((x[0]-.002 for x in rows if x[0]>=.002 and x[3]<=1),None)
  cases.append({'bst_C_nF':cb_nf,'assumed_constant_pump_uA':pump_uA,'time_to_8V_s':rise,'gate_before_off_V':on_last[3],
   'bst_before_off_V':on_last[5],'gate_falls_to_1V_after_input_s':release})
  shutil.copyfile(wave,ROOT/f'仿真/结果/LTC7001_等效_{cb_nf}nF_{pump_uA}uA波形.txt')
report={'status':'SIMPLIFIED_GATE_NETWORK_NOT_ADI_MODEL','cases':cases,'assumptions':['TS固定0V；BST初始10V、0或15µA恒流假设，不模拟限压/预充/UVLO/延迟。手册15µA指标条件为TS=20V、BST−TS=10V，不能当成此TS=0V工况的保证。',
 '上拉7Ω和下拉4Ω取手册12V条件下最大值、外串10k和1M栅源；直接套用BST=10V无保证。',
 '两MOS总典型Qg约321nC换算常数电容32.1nF，不代表非线性Qg、Miller平台或极限值',
 '输入理想互补开关，不验证LTC实际逻辑、驱动电源骤降、MOS短路电流或电感能量'],
 'interpretation':'8V和1V仅为波形比较点，不代表MOS导通/关断额定阈值。'}
(ROOT/'仿真/结果/LTC7001_等效栅极筛查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
if any(c['time_to_8V_s'] is None or c['gate_falls_to_1V_after_input_s'] is None for c in cases):raise SystemExit('部分比较点未达到')
