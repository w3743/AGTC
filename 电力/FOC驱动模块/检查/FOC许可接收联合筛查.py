"""FOC LPV receiver/hold candidate, proxy circuit only; no device qualification."""
import hashlib, json, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENET = ROOT / 'eda工程文件/导出/当前交付/FOC_G431_P1.enet'
EXE = ROOT / '仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
OUT = Path(__file__).with_suffix('.json')
data = json.loads(ENET.read_text(encoding='utf-8-sig'))
parts = {v['props']['Designator']: v for v in data['components'].values()}
for ref, nets in {'Q_BUCK_CLAMP':['BUCK_CLAMP_BASE','GND_SYS','BUCK_RT_SD'], 'Q_BUCK_RELEASE':['BUCK_RELEASE_BASE','GND_SYS','BUCK_CLAMP_BASE'], 'R_BUCK_VALID_BASE':['BUCK_RELEASE_BASE','LOAD_POWER_VALID'], 'R_LOAD_VALID_BIAS':['LOAD_EN_BIAS_EXT','SAFE_5V']}.items():
    found = [parts[ref]['pinInfoMap'][str(i)]['net'] for i in range(1,len(nets)+1)]
    if found != nets: raise ValueError((ref,found))
assert not parts['J_MCU_L']['pinInfoMap']['9']['net'] and not parts['J_MCU_L']['pinInfoMap']['16']['net']

board = '''
Rcl{b} bus cl{b} 99k
Rv{b} lpv base{b} 4.747k
Rpd{b} base{b} 0 99k
Qr{b} cl{b} base{b} 0 np
Qc{b} rt{b} cl{b} 0 np
Rrt{b} bus rt{b} 277.2k
Brt{b} rt{b} 0 I={{max(0,(V(rt{b})-2.2)/100)}}
Vgpio{b} gpio{b} 0 {gpio}
Rh{b} gpio{b} hb{b} 4.747k
Rhp{b} hb{b} 0 99k
Qh{b} cl{b} hb{b} 0 np
Rn{b} lpv nb{b} 47.47k
Qn{b} in{b} nb{b} 0 np
Rp{b} vcc{b} in{b} 9.9k
Cpin{b} in{b} 0 20p
'''
cases = {
 'normal_lpv_revoked': ('DC 4.75','PWL(0 5 1m 5 1.001m 0)','VccA vcca 0 DC 3.3','PWL(0 0 0.4m 0 0.401m 2.4)','PWL(0 0 0.4m 0 0.401m 2.4)',''),
 'a_bias_lost': ('PWL(0 4.75 1m 4.75 1.001m 0)','DC 5','VccA vcca 0 DC 3.3','PWL(0 0 0.4m 0 0.401m 2.4)','PWL(0 0 0.4m 0 0.401m 2.4)',''),
 'a_mcu_unpowered': ('DC 4.75','DC 5','VccA vcca 0 DC 0','DC 0','PWL(0 0 0.4m 0 0.401m 2.4)',''),
 'a_mcu_floating': ('DC 4.75','DC 5','Roff vcca 0 10Meg\nCoff vcca 0 1u IC=0','DC 0','PWL(0 0 0.4m 0 0.401m 2.4)',''),
 'cold_precharge': ('DC 4.75','DC 0','VccA vcca 0 DC 3.3','DC 0','DC 0',''),
 'gpio_a_stuck_high_precharge': ('DC 4.75','DC 0','VccA vcca 0 DC 3.3','DC 2.4','DC 0',''),
 'only_b_hold_on_lpv_loss': ('DC 4.75','PWL(0 5 1m 5 1.001m 0)','VccA vcca 0 DC 3.3','DC 0','DC 2.4',''),
 'notice_a_stuck_low': ('DC 4.75','PWL(0 5 1m 5 1.001m 0)','VccA vcca 0 DC 3.3','DC 2.4','DC 2.4','Rshort ina 0 1'),
}
results={}
with tempfile.TemporaryDirectory(prefix='foc_lpv_') as td:
    work=Path(td)
    for name,(safe,opto,vcca,gpioa,gpiob,fault) in cases.items():
        deck=f'''FOC shared LPV and candidate notification screening
.model np NPN(IS=1e-14 BF=100 VAF=80 CJE=8p CJC=4p TF=1n TR=100n)
.model sw SW(Ron=300 Roff=1G Vt=2 Vh=0)
Vbus bus 0 22.5
Vsafe safe 0 {safe}
Rb safe optin 1.01k
Vopto en 0 {opto}
Sop optin lpv en 0 sw
{vcca}
VccB vccb 0 3.3
{fault}
'''+''.join(board.format(b=b,gpio=gpioa if b=='a' else gpiob) for b in 'ab')+f'''
.control
set noaskquit
set wr_singlescale
tran 0.2u 2m {'uic' if name == 'a_mcu_floating' else ''}
wrdata wave.txt v(lpv) v(rta) v(rtb) v(ina) v(inb) v(vcca)
quit
.endc
.end
'''
        (work/'test.cir').write_text(deck,encoding='ascii')
        (work/'wave.txt').unlink(missing_ok=True)
        p=subprocess.run([str(EXE),'-b','test.cir'],cwd=work,capture_output=True,text=True,timeout=30)
        if p.returncode or not (work/'wave.txt').exists(): raise RuntimeError(p.stdout+p.stderr)
        rows=[list(map(float,line.split())) for line in (work/'wave.txt').read_text().splitlines()]
        def at(t): return min(rows,key=lambda r:abs(r[0]-t))
        results[name]={'before_0_9ms':at(.0009),'after_1_9ms':at(.0019)}
assert results['normal_lpv_revoked']['after_1_9ms'][1] < .1
assert results['normal_lpv_revoked']['after_1_9ms'][2] > 1.05
assert results['normal_lpv_revoked']['after_1_9ms'][4] > 3
assert results['a_bias_lost']['after_1_9ms'][1] < .1
assert results['a_bias_lost']['after_1_9ms'][5] > 3
assert results['cold_precharge']['after_1_9ms'][2] < .3 and results['cold_precharge']['after_1_9ms'][3] < .3
assert results['gpio_a_stuck_high_precharge']['after_1_9ms'][2] > 1.05 and results['gpio_a_stuck_high_precharge']['after_1_9ms'][3] < .3
assert results['only_b_hold_on_lpv_loss']['after_1_9ms'][2] < .3 and results['only_b_hold_on_lpv_loss']['after_1_9ms'][3] > 1.05
assert results['notice_a_stuck_low']['after_1_9ms'][4] < .3 and results['notice_a_stuck_low']['after_1_9ms'][5] > 3
OUT.write_text(json.dumps({'status':'PROXY_SCREEN_ONLY_HOLD','source_enet_sha256':hashlib.sha256(ENET.read_bytes()).hexdigest(),'columns':['time_s','LPV_V','RT_A_V','RT_B_V','PC10_A_V','PC10_B_V','VCC_A_V'],'assumptions':'300 ohm optocoupler switch proxy; BC847 proxy BF=100; 20 pF GPIO; fixed 22.5 V bus; ideal GPIO; simplified RT bias. Floating rail has 1 uF and 10 Mohm, initially zero with UIC. Not manufacturer worst case or Buck power model. No wire capacitance or actual propagation bound.','result':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('8 proxy scenarios computed; negative controls demonstrate failure; qualification remains HOLD')
