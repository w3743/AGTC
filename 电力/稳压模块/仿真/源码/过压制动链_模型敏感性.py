"""两颗TI宏模型+通用NPN的假设敏感性扫描；不是器件全温保证。"""
from pathlib import Path
import json,shutil,subprocess,tempfile,zipfile
ROOT=Path(__file__).resolve().parents[2]; EXE=ROOT.parent/'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
base=(ROOT/'仿真/源码/母线过压到制动驱动_组合模型.cir').read_text(encoding='utf-8');results=[]
with tempfile.TemporaryDirectory(prefix='dumpcorners_') as tmp:
 d=Path(tmp);(d/'.spiceinit').write_text('set ngbehavior=ps\n',encoding='ascii')
 for pkg,member,dest in [('snvmc71a.zip','tlv6700.lib','tlv.lib'),('slum488.zip','UCC27511_TRANS.LIB','ucc.lib')]:
  with zipfile.ZipFile(ROOT/'仿真/输入/厂家模型'/pkg) as z:(d/dest).write_bytes(z.read(member))
 for supply in (11,12,13):
  for beta in (20,100,300):
   source=base.replace('Vdd vdd 0 12',f'Vdd vdd 0 {supply}').replace('BF=100',f'BF={beta}').replace('4.5m 3.3 4.9m 3.3','4.5m 3.0 4.9m 3.0')
   (d/'case.cir').write_text(source,encoding='ascii');wave=d/'chain_wave.txt';wave.unlink(missing_ok=True)
   p=subprocess.run([str(EXE.resolve()),'-b','case.cir'],cwd=d,capture_output=True,timeout=35,creationflags=subprocess.CREATE_NO_WINDOW)
   log=p.stdout.decode('utf-8','replace')+p.stderr.decode('utf-8','replace')
   if p.returncode or 'Error:' in log or not wave.exists():raise RuntimeError(log[-2000:])
   rows=[list(map(float,l.split())) for l in wave.read_text().splitlines() if len(l.split())>=10]
   if not rows or rows[-1][0]<5.49e-3:raise RuntimeError('波形不完整')
   get=lambda t:min(rows,key=lambda x:abs(x[0]-t)); snaps=[get(t)[7] for t in (.0005,.0025,.0042,.0047,.0052)]
   checks=[snaps[0]<1,snaps[1]>.8*supply,snaps[2]<1,snaps[3]>.8*supply,snaps[4]<1]
   results.append({'supply_V':supply,'generic_BF':beta,'pwm_high_V':3.0,'gate_snapshots_V':snaps,'all_five_checks':all(checks)})
report={'status':'MODEL_SENSITIVITY_SWEEP','cases':results,'scope':'11/12/13V与假设NPN BF=20/100/300；未证明全部器件全温与容差，NPN非厂商宏模型，固定11.36nF非真实MOS。'}
(ROOT/'仿真/结果/过压制动链_模型敏感性.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
if not all(x['all_five_checks'] for x in results):raise SystemExit('有模型工况未通过')
