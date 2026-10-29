"""制动能量流平均模型；理想变换器/开关，不代表开关级或闭环稳定性仿真。"""
from pathlib import Path
import csv,json,math
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'仿真/结果'
def run(sc_enabled,dt=2e-6):
 mass=30.; v0=15/3.6; decel=3.9624; stop=v0/decel
 cbus=.0054; cap=350.; r=.82; eta=.95; bus=37.8; cell=2.; dump=False
 t=0.; peak=bus; emin=bus; esc=edump=egen=0.; toggles=0; rows=[]; i=0
 while t<stop:
  step=min(dt,stop-t); speed=max(0.,v0-decel*t); regen=mass*decel*speed
  if not dump and bus>=40.32:dump=True;toggles+=1
  elif dump and bus<=38.63:dump=False;toggles+=1
  capv=8*cell; charge=min(20.,regen*eta/capv) if sc_enabled and cell<2.20 else 0.
  psc=charge*capv/eta; pdump=bus*bus/r if dump else 0.
  previous=bus; bus=math.sqrt(bus*bus+2*step*(regen-psc-pdump)/cbus)
  cell+=step*(charge-cell/120)/cap
  peak=max(peak,bus);emin=min(emin,bus);egen+=regen*step;esc+=psc*step;edump+=pdump*step
  if i%max(1,round(.001/dt))==0:rows.append([t,previous,cell,regen,charge,int(dump),pdump])
  t+=step;i+=1
 residual=egen-esc-edump-.5*cbus*(bus*bus-37.8**2)
 return {'enabled':sc_enabled,'stop_s':stop,'bus_min_V':emin,'bus_max_V':peak,'bus_final_V':bus,'cell_final_V':cell,
         'regen_J':egen,'converter_input_J':esc,'dump_J':edump,'bus_energy_delta_J':.5*cbus*(bus**2-37.8**2),
         'energy_residual_J':residual,'dump_switch_transitions':toggles},rows
results=[];convergence=[]
for enabled in (True,False):
 result,rows=run(enabled);assert abs(result['energy_residual_J'])<.1,result
 coarse,_=run(enabled,20e-6)
 convergence.append({'SC_enabled':enabled,'bus_peak_20us_V':coarse['bus_max_V'],
                     'bus_peak_2us_V':result['bus_max_V'],'difference_V':abs(coarse['bus_max_V']-result['bus_max_V'])})
 p=OUT/('回生平均_超容可用.csv' if enabled else '回生平均_超容不可用.csv')
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['t_s','bus_V','cell_V','regen_W','SC_charge_A','dump_on','dump_W']);w.writerows(rows)
 results.append(result)
out={'status':'IDEALISED_AVERAGED_MODEL_NOT_SWITCHING_OR_HARDWARE_VALIDATION','dt_s':2e-6,'timestep_comparison':convergence,
     'assumptions':['30kg, 15km/h, 3.9624m/s²匀减速；假设全部机械功率即时注入母线','电池充放电均为0、母线只计5.4mF',
       '超容初始各2V、350F、无ESR；转换器理想、效率95%、充电上限20A且不可超出当时回生功率',
       '0.82Ω理想开关；40.32V开启、38.63V关闭，阈值无误差及延迟','忽略电感、PWM、线束、驱动和固件'],
     'cases':results,'limitations':'不能验证MOS栅极、PWM纹波、过压峰值、实际制动距离、长下坡或热能力。'}
(OUT/'回生母线平均模型.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(results,ensure_ascii=False))
