"""理想滞环功率级在母线和电感参数下的模型敏感性，不是实际PWM控制器。"""
from pathlib import Path
import json,shutil,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[2];EXE=ROOT.parent/'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
base=(ROOT/'仿真/源码/初充零伏_理想开关.cir').read_text(encoding='utf-8');result=[]
with tempfile.TemporaryDirectory(prefix='chargecorners_') as tmp:
 d=Path(tmp)
 for bus in (22.5,37.8):
  for inductance_uH in (12.8,20,24):
   s=base.replace('Vbus bus 0 37.8',f'Vbus bus 0 {bus}').replace('Lmain sw n 20u IC=0',f'Lmain sw n {inductance_uH}u IC=0')
   (d/'case.cir').write_text(s,encoding='ascii');wave=d/'charge_wave.txt';wave.unlink(missing_ok=True)
   p=subprocess.run([str(EXE.resolve()),'-b','case.cir'],cwd=d,capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
   log=p.stdout.decode('utf-8','replace')+p.stderr.decode('utf-8','replace')
   if p.returncode or 'Error:' in log or not wave.exists():raise RuntimeError(log[-1500:])
   rows=[list(map(float,l.split())) for l in wave.read_text().splitlines() if len(l.split())>=6]
   if not rows or rows[-1][0]<.000999:raise RuntimeError('波形不完整')
   settled=[x[3] for x in rows if x[0]>=.0001];first=next((x[0] for x in rows if x[3]>=19.5),None)
   result.append({'bus_V':bus,'inductance_uH':inductance_uH,'first_19_5A_s':first,
                  'current_after_100us_A_min_max':[min(settled),max(settled)],
                  'minimum_entire_current_A':min(x[3] for x in rows)})
report={'status':'IDEAL_SWITCH_HYSTERESIS_SENSITIVITY_NOT_REAL_PWM','cases':result,
 'limitations':['20A滞环控制是理想假设，非实际STM32/DRV驱动','忽略开关延迟、死区、自举、磁芯饱和非线性及PCB寄生',
                '零伏初始；1ms片段，不是完整几十秒初充','电感12.8µH属于文档中的偏置筛查点，非全温保证']}
(ROOT/'仿真/结果/初充开关模型_参数扫描.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
