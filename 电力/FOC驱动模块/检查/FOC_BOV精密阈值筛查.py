"""DC tolerance comparison only; dynamic qualification and sourcing remain open."""
import json
from pathlib import Path
OUT=Path(__file__).with_suffix('.json')
e=.001+25e-6*60
lb=10000*(1-e); hb=10000*(1+e)
cases=[]
for top in (1_040_000,1_050_000,1_070_000,1_069_800):
 lo=top*(1-e); hi=top*(1+e)
 # 15 nA at INB is specified at VIN=0.1 V, not 0.4 V: sensitivity only.
 up_lo=.396*(1+lo/hb)-15e-9*lo
 up_hi=.404*(1+hi/lb)+15e-9*hi
 down_lo=.387*(1+lo/hb)-15e-9*lo
 down_hi=.400*(1+hi/lb)+15e-9*hi
 energy=.5*.05286*(up_lo**2-42**2)
 # Measured 10mV input overdrive does NOT imply a specified maximum delay.
 overdrive_hi=(hi/lb+1)*.010
 cases.append({'rt_ohm':top,'up_nominal_v':.4*(1+top/10000),
  'up_low_v':up_lo,'up_high_v':up_hi,'down_low_v':down_lo,'down_high_v':down_hi,
  'conditional_margin_vs_42v_v':up_lo-42,'to_44p4_v_v':44.4-up_hi,
  '10mv_input_overdrive_bus_v':overdrive_hi,'up_high_plus_10mv_overdrive_v':up_hi+overdrive_hi,
  'conditional_energy_42v_to_up_low_j':energy,
  'conditional_at_495w_ms':energy/495*1000,'conditional_at_708w_ms':energy/708*1000,
  'conditional_at_915w_ms':energy/915*1000})
central=json.loads((Path(__file__).resolve().parents[2]/'稳压模块/检查/候选_预充与超容隔离_R1/PB7退役_TLV6700过驱窗口比较.json').read_text(encoding='utf-8'))['preferred_255k_plus_258k_over_5110']
precise_e=.001+25e-6*60; cheap_e=.01+100e-6*60
rt_hi=510000*(1+precise_e)+2700*(1+cheap_e); rt_lo=510000*(1-precise_e)+2700*(1-cheap_e)
rb_hi=5110*(1+precise_e); rb_lo=5110*(1-precise_e)
central_new={'rt_nominal_ohm':512700,'rt_min_ohm':rt_lo,'rt_max_ohm':rt_hi,'rb_ohm':5110,'rise_static_latest_v':.404*(1+rt_hi/rb_lo)+25e-9*rt_hi+.04,'fall_static_earliest_v':.387*(1+rt_lo/rb_hi)-25e-9*rt_lo-.04}
central_new['rise_static_earliest_v']=.396*(1+rt_lo/rb_hi)-25e-9*rt_lo-.04
central_new['fall_static_latest_v']=.400*(1+rt_hi/rb_lo)+25e-9*rt_hi+.04
central_new['rise_overdrive10_earliest_v']=central_new['rise_static_earliest_v']+.010*(1+rt_lo/rb_hi)
central_new['fall_overdrive10_latest_v']=central_new['fall_static_latest_v']-.010*(1+rt_hi/rb_lo)
central_new['rise_overdrive10_latest_v']=central_new['rise_static_latest_v']+.010*(1+rt_hi/rb_lo)
central_new['fall_overdrive10_earliest_v']=central_new['fall_static_earliest_v']-.010*(1+rt_lo/rb_hi)
foc=cases[-1]; cv=central_new['rise_overdrive10_latest_v']; windows=[]
assert 42.53 < foc['up_low_v'] < 42.54 and 43.85 < foc['up_high_v'] < 43.87
assert 42.20 < cv < 42.24
v0=15/3.6; peak_zero_delay=30*v0**3/(2*3); peak_100ms=30*v0**3/(2*(3-v0*.1)); scenario={'mass_kg':30,'speed_kmh':15,'distance_m':3,'kinetic_energy_j':.5*30*v0**2,'peak_mechanical_w_ideal_no_delay':peak_zero_delay,'peak_mechanical_w_ideal_100ms_delay':peak_100ms,'peak_with_30w_injection_w':peak_100ms+30,'notes':'constant deceleration after delay; unit conversion efficiency; no rotational inertia or road effects; 30 W is sensitivity only'}
for delta in (0,.1,.2,.3,.4):
 end=foc['up_low_v']-delta
 energy=max(0,.5*.05286*(end**2-cv**2))
 windows.append({'local_minus_central_v':delta,'central_to_foc_static_margin_v':foc['up_low_v']-central_new['rise_static_latest_v']-delta,'post_10mv_overdrive_margin_v':end-cv,'energy_j':energy,'budget_ms_at_450p03w':energy/450.03*1000,'budget_ms_at_915w':energy/915*1000})
assert windows[1]['budget_ms_at_450p03w'] > windows[2]['budget_ms_at_450p03w'] > 0
assert windows[-1]['post_10mv_overdrive_margin_v'] < 0 and windows[-1]['energy_j'] == 0
OUT.write_text(json.dumps({'central_513k_5p11k':central,'central_reuse_512p7k_5p11k':central_new,'vehicle_sensitivity':scenario,'dynamic_windows':windows,'status':'STATIC_SENSITIVITY_ONLY_NOT_ADOPTED','assumptions':
 'TLV6700 positive threshold 0.396..0.404 V and negative 0.387..0.400 V; both resistors 0.1% and 25ppm/C, independent 60 K temperature rise; +/-15nA input bias sensitivity despite TI test at VIN=0.1V; no leakage/contamination/VDD/ageing or dynamics. 42V central dynamic and 44.4V TVS breakdown are comparison assumptions, not safe clamps or guarantees.',
 'cases':cases},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for c in cases:print(c['rt_ohm'],round(c['up_low_v'],3),round(c['up_high_v'],3),round(c['down_low_v'],3),round(c['down_high_v'],3))
