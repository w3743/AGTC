from pathlib import Path
import json,shutil,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[2];EXE=ROOT.parent/'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
with tempfile.TemporaryDirectory(prefix='muxsim_') as tmp:
 d=Path(tmp);shutil.copyfile(ROOT/'仿真/源码/抽头切换_理想开关.cir',d/'case.cir')
 p=subprocess.run([str(EXE.resolve()),'-b','case.cir'],cwd=d,capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
 log=p.stdout.decode('utf-8','replace')+p.stderr.decode('utf-8','replace');wave=d/'mux_wave.txt'
 if p.returncode or 'Error:' in log or not wave.exists():raise RuntimeError(log[-2500:])
 rows=[list(map(float,l.split())) for l in wave.read_text().splitlines() if len(l.split())>=6]
 if not rows or rows[-1][0]<119e-6:raise RuntimeError('仿真不完整')
 gain=1/(1+68000*(1/10000+1/(10000+4.5))) * (10000/(10000+4.5))
 targets={'high':16*gain,'low':2*gain}; tolerance=3.3/8192
 times={}
 for name,start,end in [('high',1e-6,60e-6),('low',61e-6,120e-6)]:
  matches=[x[0]-start for x in rows if start<=x[0]<=end and abs(x[5]-targets[name])<=tolerance]
  times[name]=min(matches) if matches else None
 result={'status':'IDEAL_MUX_RC_TRANSIENT','gain':gain,'expected_V':targets,'settling_time_s_to_half_12bit_LSB':times,
         'threshold_V':tolerance,'scope':'理想开关+4.5Ω；忽略注入电荷、芯片电容与漏电、ADC采样保持、误差、固件时序和抽头动态。'}
 shutil.copyfile(wave,ROOT/'仿真/结果/抽头切换_理想开关波形.txt')
 (ROOT/'仿真/结果/抽头切换_理想开关汇总.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False))
