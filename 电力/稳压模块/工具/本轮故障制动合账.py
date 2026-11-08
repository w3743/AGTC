"""条件算术筛查；不代表实测或制动安全许可。"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / '检查' / '本轮磁盘读回.enet'
OUT = ROOT / '检查' / '本轮故障制动合账.json'
rlo, rhi = 0.82 * 0.95, 0.82 * 1.05
voltages = (22.5, 37.8, 39.209, 40.3, 40.81, 42.0)
rows = [{'V': v, 'current_A_min': v/rhi, 'current_A_max': v/rlo,
         'power_W_min': v*v/rhi, 'power_W_max': v*v/rlo} for v in voltages]
v0 = 37.8
def buffer(cap, limit, power):
    e = 0.5 * cap * (limit**2 - v0**2)
    return {'C_F':cap, 'v_start_V':v0, 'v_comparison_V':limit,
            'energy_J':e, 'net_input_W':power, 'ideal_time_s':e/power}

m, speed, accel = 30, 15/3.6, 3.9624
t = speed/accel
scenarios=[]
for deg in (0,10,20):
    slope = 9.80665*math.sin(math.radians(deg))
    energy = 0.5*m*speed**2 + m*slope*speed*t/2
    scenarios.append({'grade_degrees':deg,'stop_s':t,'distance_m':speed*t/2,
                      'mechanical_initial_W':m*(accel+slope)*speed,
                      'mechanical_energy_J':energy})
result = {
 'status':'screening_only_not_guarantee',
 'source_netlist_sha256':hashlib.sha256(NET.read_bytes()).hexdigest(),
 'handoff_sha256': {name:hashlib.sha256((ROOT.parent/name/'设计细节.md').read_bytes()).hexdigest()
                    for name in ('BMS模块', 'FOC驱动模块')},
 'resistor':{'ohm_nominal':0.82,'ohm_min':rlo,'ohm_max':rhi,
             'assumptions':'constant resistance, ideal switch, no wiring, no thermal change; no real power or pulse rating known',
             'static_full_on':rows,
             'example_300W_only':{'allowed_duty_40p3V_average_only':300/(40.3**2/rlo),
                 'minimum_period_s_for_260p4167J_average_only':(0.5*m*speed**2)/300}},
 'capacitor_comparison':[
      buffer(.024,40.3,500),buffer(.024,39.209,500),
      buffer(.05286,40.3,495.3),buffer(.05286,39.209,495.3),
      buffer(.0054156,40.3,500)],
 'mechanical_comparison':{'assumptions':'30kg, 15km/h, constant deceleration 3.9624m/s2; gravitational slope added, no inertia, losses or rolling resistance; not user stopping requirement',
                          'scenarios':scenarios},
 'sc_initial_charge':{'series_cells':8,'F_per_cell':350,'equivalent_F':43.75,
                      'target_cell_V':2,'sc_current_A':20,'ideal_time_s':35,
                      'ideal_energy_J':5600,'terminal_power_W':320},
 'required_formulas':{
   'node_balance':'P_regen + P_other_sources = P_battery_net + P_sc_bus + P_dump + P_other_load + dE_bus/dt + losses',
   'battery_limit':'0 <= I_battery_net <= min(16 A, qualified fresh CCL), shared once',
   'dc_link_energy':'E = integral V dQ; for constant effective C, ΔE=0.5*C*(Vlimit^2-Vstart^2)',
   'sc_energy':'E_sc=0.5*sum(C_i*(V_i_end^2-V_i_start^2)); stop at highest qualified cell limit',
   'resistor_energy':'E_R = integral i_R(t)^2 * R(T,t) dt',
   'precharge':'C_eff(V,T,t)*dV/dt=(Vpack-V)/Rpre-I_REG(V,t)-I_other(V,t)',
   'repeat':'P_avg= sum(E_pulses)/period <= installed derated continuous rating; thermal transient and individual pulse limits also required'
 },
 'cautions':['The given voltages are comparison points, not guaranteed safe maximums.',
             'The circuit has no verified brake resistor pulse, temperature, lifetime or continuous rating.',
             'FOC end-to-end BOV response and central 12 V failure timing are not bounded.',
             'Capacitance tolerances are initial values and exclude temperature, aging, DC bias, wiring and module input capacitance.',
             'No local dump is credited after a FOC feeder disconnect; gate shutdown alone does not dissipate energy.']
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(str(OUT))
