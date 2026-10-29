"""当前8S超容初充的平均电流模型；不模拟PWM/驱动或实际控制软件。"""
import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def run(caps,dt=0.001):
    v=[0.0]*8;t=0.0;current=20.0
    while max(v)<2.0 and t<100:
        v=[x+(current-x/120)/c*dt for x,c in zip(v,caps)]
        t+=dt
    return {'capacitance_F':caps,'stop_time_s':t,'cell_voltages_V':v,
            'total_voltage_V':sum(v),'maximum_cell_V':max(v),'current_A':current}
cases=[run([c]*8) for c in (280,350,420)]
cases.append(run([280,420,350,350,350,350,350,350]))
assert all(0 < x['stop_time_s'] < 100 for x in cases)
assert abs(cases[1]['stop_time_s'] - (-120*350*math.log(1-2/(20*120))))<0.002
result={'status':'AVERAGED_MODEL_COMPLETED_NOT_SWITCHING_SIMULATION','model':'8节串联，每节并联120Ω；理想20A恒流，最高理想电容电压2.00V立即停止；欧拉步长1ms',
        'cases':cases,'not_modelled':['ESR压升','电感及PWM','驱动和隔离器件动态','测量误差和采样延迟','故障与制动','实际固件'],
        'scope':'仅检查假设的恒流充电与容量差；不是实际电路运行验证。'}
p=ROOT/'仿真/结果/当前方案动态筛查.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(cases,ensure_ascii=False))
