"""Ideal three-phase SPWM DC-link ripple sensitivity, not a capacitor rating test."""
from pathlib import Path
import json,math

def rms(m,phi,n=360,peak=120):
    acc=0.
    for k in range(n):
        theta=math.tau*k/n
        a=(theta,theta-math.tau/3,theta+math.tau/3)
        duty=[.5+.5*m*math.sin(x) for x in a]
        current=[peak*math.sin(x-phi) for x in a]
        average=sum(d*i for d,i in zip(duty,current))
        edges=sorted((0.,1.,*duty))
        acc+=sum((hi-lo)*(sum(i for d,i in zip(duty,current) if d>(lo+hi)/2)-average)**2 for lo,hi in zip(edges,edges[1:]))
    return math.sqrt(acc/n)

best=(0.,0.,0.)
for im in range(21):
    for ip in range(31):
        m,phi=im/20,ip*math.pi/30
        x=rms(m,phi,120)
        if x>best[0]: best=(x,m,phi)
r,m,phi=best
out={'status':'IDEALIZED_SCREEN_NOT_DESIGN_APPROVAL','model':'Common-carrier sinusoidal PWM, balanced sinusoidal phase currents, constant within each PWM period; ideal DC source supplies instantaneous PWM-cycle mean; all switching ripple assigned to local capacitor bank. Analytic carrier intervals and numerical electrical-angle average. No wire impedance, MLCC current sharing, deadtime, phase-current ripple, 60 C thermal derating, or transients.', 'phase_current_peak_A':120,'modulation_index_scan':[0,1,0.05],'phase_angle_scan_deg':[0,180,6], 'theta_samples':120,'maximum_scan':{'rms_A':r,'modulation_index':m,'phase_angle_deg':phi*180/math.pi,'check_360_samples_A':rms(m,phi)},'example':{'bus_V':33.3333333333,'ideal_input_power_W':1.5*(m/2)*33.3333333333*120*math.cos(phi),'two_equal_capacitors_each_rms_A':r/2,'three_equal_capacitors_each_rms_A':r/3},'PEH200_rating_reference':{'10kHz_40C_A_each':25.6,'10kHz_50C_forced_air_and_chassis_A_each':34.7,'url':'https://content.kemet.com/datasheets/KEM_A4034_PEH200.pdf','caution':'Ratings at stated conditions; waveform is multi-frequency and 60 C environment is not 40 C. Equality of current sharing is assumed.'},'conclusion':'Two 25.6 A entries total 51.2 A versus ideal model 55.13 A at one operating point. This comparison signals inadequate evidence, not a failure or safe RMS rating; do not directly equate 20 kHz broadband RMS with a 10 kHz rating.'}
p=Path(__file__).with_suffix('.json');p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
