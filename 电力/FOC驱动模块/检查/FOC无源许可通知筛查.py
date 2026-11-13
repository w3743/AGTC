"""Two-board LPV divider comparison, unqualified component proxy."""
import hashlib, json, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ENET=ROOT/'eda工程文件/导出/当前交付/FOC_G431_P1.enet'
EXE=ROOT/'仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
OUT=Path(__file__).with_suffix('.json')
data=json.loads(ENET.read_text(encoding='utf-8-sig'))
parts={x['props']['Designator']:x for x in data['components'].values()}
for r,expected in {'R_BUCK_VALID_BASE':{'LOAD_POWER_VALID','BUCK_RELEASE_BASE'},'R_BUCK_VALID_PD':{'BUCK_RELEASE_BASE','GND_SYS'},'R_LOAD_VALID_BIAS':{'LOAD_EN_BIAS_EXT','SAFE_5V'}}.items():
 assert {x['net'] for x in parts[r]['pinInfoMap'].values()}==expected
assert parts['J_MCU_L']['pinInfoMap']['16']['net']==''
case={'revoke':('DC 4.75','PWL(0 5 1m 5 1.001m 0)','20p'),
      'revoke_1n':('DC 4.75','PWL(0 5 1m 5 1.001m 0)','1n'),
      'revoke_10n':('DC 4.75','PWL(0 5 1m 5 1.001m 0)','10n'),
      'bias_a_lost':('PWL(0 4.75 1m 4.75 1.001m 0)','DC 5','20p'),
      'precharge':('DC 4.75','DC 0','20p')}
res={}
with tempfile.TemporaryDirectory(prefix='foc_lpv_div_') as tmp:
 w=Path(tmp)
 for name,(safe,en,cap) in case.items():
  deck=f'''LPV two receiver plus passive MCU inputs proxy
.model np NPN(IS=1e-14 BF=100 VAF=80 CJE=8p CJC=4p TF=1n TR=100n)
.model opto SW(Ron=300 Roff=1G Vt=2 Vh=0)
Vsafe safe 0 {safe}
Rsource safe optin 1.01k
Venable en 0 {en}
Sopt optin lpv en 0 opto
RrecvA lpv ba 4.747k
RpdA ba 0 99k
QrecvA ca ba 0 np
RrecvB lpv bb 4.747k
RpdB bb 0 99k
QrecvB cb bb 0 np
Rca ca 0 100k
Rcb cb 0 100k
RtopA lpv pina 101.35k
RbotA pina 0 217.03k
RtopB lpv pinb 101.35k
RbotB pinb 0 217.03k
Cina pina 0 {cap}
Cinb pinb 0 {cap}
.control
set noaskquit
set wr_singlescale
tran 0.2u 2m
wrdata wave.txt v(lpv) v(pina) v(pinb)
quit
.endc
.end
'''
  (w/'test.cir').write_text(deck,encoding='ascii'); (w/'wave.txt').unlink(missing_ok=True)
  p=subprocess.run([str(EXE),'-b','test.cir'],cwd=w,capture_output=True,text=True,timeout=30)
  if p.returncode or not (w/'wave.txt').exists(): raise RuntimeError(p.stdout+p.stderr)
  rows=[list(map(float,x.split())) for x in (w/'wave.txt').read_text().splitlines()]
  res[name]={str(t):min(rows,key=lambda a:abs(a[0]-t)) for t in (.0009,.0019)}
  if name.startswith('revoke'):
   hits=[r[0] for r in rows if r[0]>=.001001 and r[2]<=1.08]
   res[name]['fall_to_1p08v_delay_us']=(min(hits)-.001001)*1e6 if hits else None
assert res['revoke']['0.0009'][2]>2.024
assert res['revoke']['0.0019'][2]<.3
assert res['bias_a_lost']['0.0019'][2]<.3
assert res['precharge']['0.0019'][2]<.3
top_min,top_max,bot_min,bot_max=98650,101350,217030,222970
vhigh_min=3.12*bot_min/(top_max+bot_min)-0.0000001*(top_max*bot_min/(top_max+bot_min))
vpin_max=5.25*bot_max/(top_min+bot_max)
OUT.write_text(json.dumps({'status':'UNQUALIFIED_CONDITIONAL_COMPARISON','enet_sha256':hashlib.sha256(ENET.read_bytes()).hexdigest(),'columns':['time_s','LPV_V','PC10_A_V','PC10_B_V'],'dc_bounds':{'assumed_lpv_min_v':3.12,'vhigh_min_after_100nA_sink_v':vhigh_min,'assumed_vdd_max_v':3.6,'design_vih_v':.49*3.6+.26,'absolute_max_assumed_lpv_5p25_v':vpin_max,'vdd0_abs_max_v':4},'limitations':'Fixed 300 ohm opto proxy, generic NPN, ideal supply, no BMS LED CTR, no actual GPIO or transients. LPV>=3.12 V is a conditional input, not a demonstrated bound. 100 nA input leakage applies stated powered ST conditions, not power off. BC receiver collector proxy not Buck. Upper resistor open pulls input low; lower resistor open can exceed VDD=0 absolute maximum. No CPU or braking validation.','result':res},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Five LPV-divider proxy cases; conditional, NOT qualified')
