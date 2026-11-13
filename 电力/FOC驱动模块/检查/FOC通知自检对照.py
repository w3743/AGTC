"""Screen a proposed PC11 transistor test of the LPV receiver; not qualification."""
import json, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXE=ROOT/'仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
OUT=Path(__file__).with_suffix('.json')
template='''LPV notification self-test hypothetical circuit
.model np NPN(IS=1e-14 BF=100 VAF=80 CJE=8p CJC=4p TF=1n TR=100n)
Vlpv lpv 0 {lpv}
Vcc vcc 0 3.3
Vtest test 0 PWL(0 0 1m 0 1.001m 2.4 1.2m 2.4 1.201m 0)
Rbase lpv base 47.47k
Qnotice in base 0 np
Rpu vcc in 9.9k
Cpin in 0 20p
Rtest test tb 4.747k
Rpd tb 0 99k
Qtest base tb 0 np
{fault}
.control
set noaskquit
set wr_singlescale
tran 0.2u 2m
wrdata wave.txt v(lpv) v(base) v(in) v(test)
quit
.endc
.end
'''
cases={
 'test_healthy':('DC 3.158',''),
 'test_input_short':('DC 3.158','Rshort in 0 1'),
 'loss_during_test':('PWL(0 3.158 1.1m 3.158 1.101m 0)',''),
 'transient_loss_hidden_by_test':('PWL(0 3.158 1.08m 3.158 1.081m 0 1.12m 0 1.121m 3.158)',''),
 'test_transistor_base_short':('DC 3.158','Rshorttest tb 0 1'),
}
res={}
with tempfile.TemporaryDirectory(prefix='foc_notify_test_') as td:
 w=Path(td)
 for name,(lpv,fault) in cases.items():
  circuit=template.format(lpv=lpv,fault=fault)
  # This negative control shorts the test transistor base to ground (test unavailable).
  (w/'test.cir').write_text(circuit,encoding='ascii'); (w/'wave.txt').unlink(missing_ok=True)
  p=subprocess.run([str(EXE),'-b','test.cir'],cwd=w,capture_output=True,text=True,timeout=30)
  if p.returncode or not (w/'wave.txt').exists(): raise RuntimeError(p.stdout+p.stderr)
  rows=[list(map(float,x.split())) for x in (w/'wave.txt').read_text().splitlines()]
  res[name]={str(t):min(rows,key=lambda r:abs(r[0]-t)) for t in (.0009,.00115,.0015)}
 for temp in (0,60,125):
  # Zero-extra-part alternative: PC11 open-drain attached directly to base;
  # 0.4 V is an assumed GPIO VOL, not an all-temperature measured value.
  circuit=template.format(lpv='DC 3.158',fault='')
  circuit=circuit.replace('Rtest test tb 4.747k\nRpd tb 0 99k\nQtest base tb 0 np',
      'Vlow vlo 0 0.4\nSgpio base vlo test 0 sw\n.model sw SW(Ron=1 Roff=1G Vt=1.2 Vh=0)')
  circuit=circuit.replace('.model np NPN',f'.temp {temp}\n.model np NPN')
  (w/'test.cir').write_text(circuit,encoding='ascii'); (w/'wave.txt').unlink(missing_ok=True)
  p=subprocess.run([str(EXE),'-b','test.cir'],cwd=w,capture_output=True,text=True,timeout=30)
  if p.returncode or not (w/'wave.txt').exists(): raise RuntimeError(p.stdout+p.stderr)
  rows=[list(map(float,x.split())) for x in (w/'wave.txt').read_text().splitlines()]
  res[f'direct_od_proxy_{temp}C']={str(t):min(rows,key=lambda r:abs(r[0]-t)) for t in (.0009,.00115,.0015)}
 for label,fault,lpv in [('pd_normal','','DC 3.158'),('pd_transistor_open','Ropen in 0 1G','DC 5.25')]:
  circuit=template.format(lpv=lpv,fault=fault)
  circuit=circuit.replace('Rtest test tb 4.747k\nRpd tb 0 99k\nQtest base tb 0 np',
      'Vlow vlo 0 0.4\nSgpio base vlo test 0 sw\n.model sw SW(Ron=1 Roff=1G Vt=1.2 Vh=0)\nRbasepd base 0 101k')
  if label=='pd_transistor_open':
   circuit=circuit.replace('Qnotice in base 0 np','* transistor open')
   circuit=circuit.replace('Rbase lpv base 47.47k','Rbase lpv base 46.53k')
  (w/'test.cir').write_text(circuit,encoding='ascii'); (w/'wave.txt').unlink(missing_ok=True)
  p=subprocess.run([str(EXE),'-b','test.cir'],cwd=w,capture_output=True,text=True,timeout=30)
  if p.returncode or not (w/'wave.txt').exists(): raise RuntimeError(p.stdout+p.stderr)
  rows=[list(map(float,x.split())) for x in (w/'wave.txt').read_text().splitlines()]
  res[label]={str(t):min(rows,key=lambda r:abs(r[0]-t)) for t in (.0009,.00115,.0015)}
assert res['test_healthy']['0.00115'][3]>3 and res['test_healthy']['0.0015'][3]<.3
assert res['test_input_short']['0.00115'][3]<.3
assert res['loss_during_test']['0.0015'][3]>3
assert res['transient_loss_hidden_by_test']['0.00115'][3]>3 and res['transient_loss_hidden_by_test']['0.0015'][3]<.3
assert res['test_transistor_base_short']['0.00115'][3]<.3
assert res['pd_transistor_open']['0.0009'][2] < 4
OUT.write_text(json.dumps({'status':'PROXY_ONLY_NOT_IMPLEMENTED','columns':['time_s','LPV_V','LPV_DETECT_BASE_V','PC10_V','PC11_V'],'assumptions':'Ideal 3.158 V LPV or ideal fall; generic NPN; 3.3 V, no real wire capacitance; 200 us test pulse; no GPIO, power, silicon or full-temperature tolerances; last negative control shorts test base to ground.','results':res},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Five transistor-test scenarios and five direct-GPIO proxy variants computed; qualification remains HOLD')
