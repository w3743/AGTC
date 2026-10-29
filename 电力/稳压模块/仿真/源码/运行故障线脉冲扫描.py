"""假设电容和开漏等效阻抗的线级敏感性；不模拟DRV去毛刺。"""
from pathlib import Path
import json,shutil,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[2];EXE=ROOT.parent/'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
base=(ROOT/'仿真/源码/故障线_脉冲等效.cir').read_text(encoding='ascii');cases=[]
with tempfile.TemporaryDirectory(prefix='faultpulses_') as tmp:
 d=Path(tmp)
 for cap_pf in (100,1000,10000):
  s=base.replace('Cwire fault 0 1n',f'Cwire fault 0 {cap_pf}p');(d/'case.cir').write_text(s,encoding='ascii');wave=d/'fault_wave.txt';wave.unlink(missing_ok=True)
  p=subprocess.run([str(EXE.resolve()),'-b','case.cir'],cwd=d,capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
  log=p.stdout.decode('utf-8','replace')+p.stderr.decode('utf-8','replace')
  if p.returncode or 'Error:' in log or not wave.exists():raise RuntimeError(log[-2000:])
  rows=[list(map(float,l.split())) for l in wave.read_text().splitlines() if len(l.split())>=4]
  if not rows or rows[-1][0]<119e-6:raise RuntimeError('波形不完整')
  pulses=[]
  for name,start,end in [('1us',10e-6,30e-6),('4.2us',40e-6,70e-6),('10us',80e-6,119e-6)]:
   low=[x[0] for x in rows if start<=x[0]<=end and x[3]<=.8]
   pulses.append({'input_pulse':name,'v_below_0_8V_duration_s':max(low)-min(low) if low else 0,'wire_min_V':min(x[3] for x in rows if start<=x[0]<=end)})
  cases.append({'assumed_wire_cap_pF':cap_pf,'pulses':pulses});shutil.copyfile(wave,ROOT/f'仿真/结果/故障线_脉冲等效_{cap_pf}pF波形.txt')
out={'status':'ASSUMED_RC_TRANSIENT_NOT_DRIVER_BEHAVIOUR','cases':cases,
 'assumptions':['CORE=3V、上拉1k、下拉10k、LTC输入下拉1M','DRV内部250k典型上拉到假设3.3V',
                '外部开漏等效Ron=40Ω、电容100pF/1nF/10nF均是敏感性假设而非实际板参数'],
 'limits':['仅测导线电平低于0.8V的持续时间，未模拟DRV内部1–4.2us去毛刺或门驱',
           '不模拟TIM1 Break/G07锁存、CORE掉电反灌、短故障脉冲的实际识别或关断',
           '10nF只是较大负载敏感性，不应当成当前PCB实测']}
(ROOT/'仿真/结果/故障线_脉冲等效扫描.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
