"""LPV two-receiver + AO3400A notification behavioral proxy, not device model."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ENET=ROOT/'eda工程文件/导出/当前交付/FOC_G431_P1.enet'
EXE=ROOT/'仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
OUT=Path(__file__).with_suffix('.json')
d=json.loads(ENET.read_text(encoding='utf-8-sig'))
parts={x['props']['Designator']:x for x in d['components'].values()}
assert not parts['J_MCU_L']['pinInfoMap']['16']['net']
for ref,nets in {'R_BUCK_VALID_BASE':{'LOAD_POWER_VALID','BUCK_RELEASE_BASE'},'R_LOAD_VALID_BIAS':{'LOAD_EN_BIAS_EXT','SAFE_5V'}}.items():
 assert {v['net'] for v in parts[ref]['pinInfoMap'].values()}==nets
cases={'revoke':('DC 4.75','PWL(0 5 1m 5 1.001m 0)','DC 3.3','1.4'),
       'bias_lost':('PWL(0 4.75 1m 4.75 1.001m 0)','DC 5','DC 3.3','1.4'),
       'precharge':('DC 4.75','DC 0','DC 3.3','1.4'),
       'a_3v3_lost':('DC 4.75','DC 5','DC 0','1.4'),
       'a_rail_zero_revoke':('DC 4.75','PWL(0 5 1m 5 1.001m 0)','DC 0','1.4'),
       'slow_0p3v_gate_proxy':('DC 4.75','PWL(0 5 1m 5 1.001m 0)','DC 3.3','0.3')}
out={}
with tempfile.TemporaryDirectory(prefix='foc_mos_notice_') as td:
 w=Path(td)
 for name,(safe,opto,vcca,vt) in cases.items():
  deck=f'''two LPV receivers and AO3400A capacitance SWITCH proxy
.model np NPN(IS=1e-14 BF=100 VAF=80 CJE=8p CJC=4p TF=1n TR=100n)
.model sop SW(Ron=300 Roff=1G Vt=2 Vh=0)
.model smos SW(Ron=.048 Roff=720k Vt={vt} Vh=.05)
Vsafe safe 0 {safe}
Rbias safe optin 1.01k
Ven en 0 {opto}
Sopt optin lpv en 0 sop
Ra lpv ba 4.653k
Rpa ba 0 98.65k
Qa ca ba 0 np
Rca ca 0 100k
Rb lpv bb 4.653k
Rpb bb 0 98.65k
Qb cb bb 0 np
Rcb cb 0 100k
Rga lpv 0 98.65k
Rgb lpv 0 98.65k
Cga lpv 0 973p
Cgb lpv 0 973p
Cgda lpv ina 77p
Cgdb lpv inb 77p
Vcca vcca 0 {vcca}
Vccb vccb 0 3.3
Rpua vcca ina 10.1k
Rpub vccb inb 10.1k
Cina ina 0 20p
Cinb inb 0 20p
Sma ina 0 lpv 0 smos
Smb inb 0 lpv 0 smos
.control
set noaskquit
set wr_singlescale
tran 0.2u 2m
wrdata wave.txt v(lpv) v(ina) v(inb) i(vcca)
quit
.endc
.end
'''
  (w/'test.cir').write_text(deck,encoding='ascii'); (w/'wave.txt').unlink(missing_ok=True)
  p=subprocess.run([str(EXE),'-b','test.cir'],cwd=w,capture_output=True,text=True,timeout=30)
  if p.returncode or not (w/'wave.txt').exists(): raise RuntimeError(p.stdout+p.stderr)
  rows=[list(map(float,x.split())) for x in (w/'wave.txt').read_text().splitlines()]
  if rows[-1][0]<.00199: raise RuntimeError('ngspice did not complete the transient: '+p.stdout+p.stderr)
  item={str(t):min(rows,key=lambda a:abs(a[0]-t)) for t in (.0009,.0019)}
  if name in ('revoke','slow_0p3v_gate_proxy'):
   hit=[r[0] for r in rows if r[0]>=.001001 and r[2]>=2.31]
   item['pc10_rise_to_2p31v_us']=(hit[0]-.001001)*1e6 if hit else None
  if name in ('a_3v3_lost','a_rail_zero_revoke'): item['peak_abs_vcca_source_a']=max(abs(r[4]) for r in rows)
  out[name]=item
assert out['revoke']['0.0009'][2]<.3 and out['revoke']['0.0019'][2]>2.31
assert out['bias_lost']['0.0019'][3]>2.31 and out['precharge']['0.0019'][2]>2.31
assert abs(out['a_3v3_lost']['0.0019'][2])<.01
body={'status':'BEHAVIORAL_PROXY_UNQUALIFIED','manufacturer':'UMW AO3400A C347475',
 'manufacturer_url':'https://www.umw-ic.com/static/pdf/db97d75d2121bceb9305ec17148391f8.pdf',
 'source_enet_sha256':hashlib.sha256(ENET.read_bytes()).hexdigest(),
 'columns':['time_s','LPV_V','PC10_A_V','PC10_B_V','VCCA_source_A'],
 'limitations':'Idealized switch model; Ron 48mOhm, Roff 720kOhm, Vt 1.4V or 0.3V and 0.05V hysteresis are sensitivity assumptions, not AO3400A transfer data. Cgs 973pF plus Cgd 77pF is based on max Ciss 1050pF at Vds15V/1MHz; not guaranteed for actual voltages/temperature. Source 300ohm proxy not VO617A CTR. No actual off leakage at partial Vgs, power rail, GPIO or hold Buck model. Delay measured from ideal switch control at 1.001ms, excludes BMS LED, opto and MCU.',
 'results':out}
OUT.write_text(json.dumps(body,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Six MOS behavioral cases, not manufacturer qualification')
