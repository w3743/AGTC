"""无弹窗运行两个理想开关例子；仅验证其理想化假设，不代表实板运行。"""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[2];EXE=ROOT.parent/'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
SETS={'charge':('初充零伏_理想开关.cir','charge_wave.txt','初充零伏_理想开关波形.txt',6,.001),
      'dump':('电阻投入_理想开关.cir','dump_wave.txt','电阻投入_理想开关波形.txt',4,1.052)}
p=argparse.ArgumentParser();p.add_argument('case',choices=SETS);a=p.parse_args()
srcname,wavename,outname,ncols,stop=SETS[a.case];source=ROOT/'仿真/源码'/srcname
with tempfile.TemporaryDirectory(prefix='powersim_') as tmp:
 d=Path(tmp);shutil.copyfile(source,d/'case.cir');run=subprocess.run([str(EXE.resolve()),'-b','case.cir'],cwd=d,capture_output=True,timeout=80,creationflags=subprocess.CREATE_NO_WINDOW)
 log=run.stdout.decode('utf-8','replace')+run.stderr.decode('utf-8','replace');wave=d/wavename
 if run.returncode or 'Error:' in log or not wave.exists():raise RuntimeError(log[-2000:])
 n=0;last=0;mins=[float('inf')]*(ncols//2);maxs=[float('-inf')]*(ncols//2)
 with wave.open() as f:
  for line in f:
   values=line.split()
   if len(values)<ncols:continue
   v=list(map(float,values[:ncols]));n+=1;last=v[0]
   for j in range(ncols//2):mins[j]=min(mins[j],v[2*j+1]);maxs[j]=max(maxs[j],v[2*j+1])
 if not n or last<stop*.999:raise RuntimeError(f'波形不完整: {last}, {n}')
 shutil.copyfile(wave,ROOT/'仿真/结果'/outname)
 report={'status':'NGSPICE_IDEAL_SWITCH_SIMULATED','case':a.case,'sample_count':n,'last_time_s':last,
  'minima':mins,'maxima':maxs,'variable_order':['switch_V','inductor_A','capacitor_V'] if a.case=='charge' else ['bus_V','dump_A'],
  'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'limits':'理想电源/开关与通用二极管；不含实际驱动器、MOS非线性、固件或布线寄生。'}
 (ROOT/'仿真/结果'/('初充零伏_理想开关复跑.json' if a.case=='charge' else '电阻投入_理想开关复跑.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False))
