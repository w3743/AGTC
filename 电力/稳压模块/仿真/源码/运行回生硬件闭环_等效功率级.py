"""真实TI比较器/驱动宏模型与理想开关功率级闭环；30ms峰值功率片段。"""
from pathlib import Path
import hashlib,json,shutil,subprocess,tempfile,zipfile
ROOT=Path(__file__).resolve().parents[2];EXE=ROOT.parent/'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
base=(ROOT/'仿真/源码/母线过压到制动驱动_组合模型.cir').read_text(encoding='utf-8')
source=base.replace('Vbus bus 0 PWL(0 37 1m 37 2m 42 3m 42 4m 37 5.5m 37)',
 '''Cbus bus 0 5.4m IC=37.8
Bregen 0 bus I={495.3/max(v(bus),1)}
Sdump bus rnode gate 0 MOSideal
Rdump rnode 0 0.82
.model MOSideal SW(Ron=0.0043 Roff=1e9 Vt=4.5 Vh=0)''')
source=source.replace('Vpwm pwm 0 PWL(0 0 4.49m 0 4.5m 3.3 4.9m 3.3 4.91m 0 5.5m 0)','Vpwm pwm 0 0')
source=source.replace('tran 1u 5.5m 0 1u','tran 1u 30m 0 1u uic')
source=source.replace('wrdata chain_wave.txt v(bus) v(outa) v(dmpoff) v(gate) v(pwm)','wrdata chain_wave.txt v(bus) v(outa) v(dmpoff) v(gate) v(rnode)')
with tempfile.TemporaryDirectory(prefix='brakeclosed_') as tmp:
 d=Path(tmp);(d/'.spiceinit').write_text('set ngbehavior=ps\n',encoding='ascii');(d/'case.cir').write_text(source,encoding='ascii')
 for pkg,member,dest in [('snvmc71a.zip','tlv6700.lib','tlv.lib'),('slum488.zip','UCC27511_TRANS.LIB','ucc.lib')]:
  with zipfile.ZipFile(ROOT/'仿真/输入/厂家模型'/pkg) as z:(d/dest).write_bytes(z.read(member))
 p=subprocess.run([str(EXE.resolve()),'-b','case.cir'],cwd=d,capture_output=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
 log=p.stdout.decode('utf-8','replace')+p.stderr.decode('utf-8','replace');wave=d/'chain_wave.txt'
 if p.returncode or 'Error:' in log or not wave.exists():raise RuntimeError(log[-2500:])
 rows=[list(map(float,l.split())) for l in wave.read_text().splitlines() if len(l.split())>=10]
 if not rows or rows[-1][0]<.02999:raise RuntimeError('仿真不完整')
 peak=max(x[1] for x in rows);minbus=min(x[1] for x in rows);peak_i=max(x[9]/.82 for x in rows)
 events=[];state=rows[0][7]>4.5
 for x in rows[1:]:
  new=x[7]>4.5
  if new!=state:events.append({'time_s':x[0],'bus_V':x[1],'on':new});state=new
 shutil.copyfile(wave,ROOT/'仿真/结果/回生硬件闭环_等效功率级波形.txt')
 (ROOT/'仿真/源码/回生硬件闭环_等效功率级.cir').write_text(source,encoding='ascii')
 report={'status':'TI_TLV_UCC_CLOSED_LOOP_WITH_IDEAL_POWER_SWITCH','duration_s':.03,'regen_W':495.3,
  'bus_V_min_max':[minbus,peak],'peak_resistor_A':peak_i,'switch_events':events,
  'limitations':['恒定495.3W为30ms短片段，不代表完整停车','电池/超容不吸能；BUS理想5.4mF',
   'TI TLV6700与UCC27511宏模型；通用NPN BF100，线性11.36nF栅容',
   '制动MOS为VT=4.5V、Ron=4.3mΩ的理想开关；不能代表IRLB4030动态、电容或Miller',
   '忽略MOS线束寄生、热和实际固件，未做容差扫描；不能作为实物峰值保证']}
 (ROOT/'仿真/结果/回生硬件闭环_等效功率级汇总.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False))
