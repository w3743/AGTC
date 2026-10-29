"""固定TS及D、按手册点拟合栅荷的组合敏感性；不是实际开关验证。"""
from pathlib import Path
import hashlib,json,shutil,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[2];EXE=ROOT.parent/'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
base=(ROOT/'仿真/源码/LTC输出_HYG非线性栅荷等效.cir').read_text(encoding='ascii');model=ROOT/'仿真/源码/HYG022_参数化等效模型.lib';cases=[]
with tempfile.TemporaryDirectory(prefix='ltchyg_') as tmp:
 d=Path(tmp);shutil.copyfile(model,d/'hyg.lib')
 for cap in (500,1000):
  for pump in (0,15):
   s=base.replace('Cbst bst 0 500n',f'Cbst bst 0 {cap}n').replace('Icp 0 bst 15u',f'Icp 0 bst {pump}u');(d/'case.cir').write_text(s,encoding='ascii');wave=d/'combined_gate.txt';wave.unlink(missing_ok=True)
   p=subprocess.run([str(EXE.resolve()),'-b','case.cir'],cwd=d,capture_output=True,timeout=40,creationflags=subprocess.CREATE_NO_WINDOW)
   log=p.stdout.decode('utf-8','replace')+p.stderr.decode('utf-8','replace')
   if p.returncode or 'Error:' in log or not wave.exists():raise RuntimeError(log[-2500:])
   rows=[list(map(float,l.split())) for l in wave.read_text().splitlines() if len(l.split())>=6]
   if not rows or rows[-1][0]<.002999:raise RuntimeError('仿真波形不完整')
   active=[x for x in rows if 10.1e-6<x[0]<.001999];rise=next((x[0]-10e-6 for x in active if x[3]>=8),None)
   off=next((x[0]-.002 for x in rows if x[0]>=.002 and x[3]<=1),None)
   cases.append({'bst_C_nF':cap,'assumed_pump_uA':pump,'to_8V_s':rise,'gate_before_off_V':active[-1][3],
                 'bst_before_off_V':active[-1][5],'off_to_1V_s':off})
   shutil.copyfile(wave,ROOT/f'仿真/结果/LTC输出_HYG栅荷_{cap}nF_{pump}uA波形.txt')
result={'status':'LTC_EQUIVALENT_WITH_HYG_ONE_PORT_FIT_NOT_REAL_SWITCHING','cases':cases,
 'HYG_model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
 'assumptions':['LTC为7Ω/4Ω理想开关+10k上拉、1M栅源，非官方宏模型','初始BST=10V，恒流泵0/15µA是假设，未模拟启动/UVLO/延迟','TS及两漏极固定0V，不存在实际负载电流',
  'HYG单口Q(VGS)曲线混合VDS=50V图与80V典型电荷表，Miller平台宽4.2–4.4V为人为平滑；没有VDS耦合Miller或反向恢复',
  '7Ω/4Ω保证值和15µA指标的原厂测试条件与此模型不同，不能当成此工况保证'],
 'interpretation':'8V和1V是便于比较的任意电压点，不表示实际MOS导通/关断。与固定32.1nF模型的差异仅作模型敏感性。'}
(ROOT/'仿真/结果/LTC输出_HYG非线性栅荷等效.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
if any(x['to_8V_s'] is None or x['off_to_1V_s'] is None for x in cases):raise SystemExit('比较点未达到')
