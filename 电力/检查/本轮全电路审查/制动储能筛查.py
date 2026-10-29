"""制动优先用途的能量/功率筛查；只输出本目录，不代表电气或实物验收。"""
from pathlib import Path
import hashlib
import json
import math

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
mechanical_path = ROOT / '机械/参数基线.json'
mechanical = json.loads(mechanical_path.read_text(encoding='utf-8-sig'))
documented = {x['id']: x for x in mechanical['groups']['documented']}
mass = float(documented['vehicle_mass_nominal']['value'])
speed_kmh = float(documented['nominal_top_speed']['value'])
speed = speed_kmh/3.6
ek = 0.5*mass*speed**2
g = 9.80665

result = {
 'date': '2026-10-29',
 'user_requirements': {
   'mechanical_brake': False,
   'supercap_priority': '吸收电制动回生；启动补能次要，无余电允许慢加速',
   'initial_charge_target': '几十秒量级；不是必须满充才准车辆启动',
   'independent_regulator_required': True,
   'new_selected_battery_design': '9S5P EVE30P；现行原生仍是旧9S4P，130A为增强候选',
   'dual_motor_peak_input_each_W': 1800,
   'peak_duration': '未冻结',
 },
 'input': {
   'mechanical_path': str(mechanical_path.relative_to(ROOT)),
   'mechanical_sha256': hashlib.sha256(mechanical_path.read_bytes()).hexdigest(),
   'mass_kg': mass, 'speed_kmh': speed_kmh,
   'mechanical_status': '文档基线，非实称/测速，不含未知载荷/旋转等效惯量',
   'cell_count': 8, 'cell_nominal_F': 350,
   'cell_EOL_sensitivity_F': 280,
   'voltage_points_only_not_frozen_thresholds_V_per_cell': [2.2, 2.45],
 },
 'braking': {
   'translational_energy_J': ek,
   'constant_deceleration_level_ground_cases': [
     {'stop_time_s': t, 'decel_m_per_s2': speed/t, 'stop_distance_m': 0.5*speed*t,
      'mean_mechanical_power_W': ek/t, 'initial_mechanical_power_W': 2*ek/t}
     for t in [0.5, 1, 2, 3]],
   'rotating_mass_sensitivity': [
     {'total_energy_to_translation_ratio': k, 'energy_J': ek*k}
     for k in [1, 1.25, 1.5, 2]],
   'downhill_constant_speed_cases': [
     {'slope_deg': slope, 'gravitational_power_W': mass*g*speed*math.sin(math.radians(slope)),
      'gravitational_energy_per_vertical_m_J': mass*g}
     for slope in [5, 10, 20, 30]],
   'boundary': '忽略滚阻/风阻，按能量全部需电气处理作筛查；不证明轮地附着、制动力矩、低速或故障停车。'},
 'capacitor_cases': [],
}
for series, parallel in [(8, 1), (4, 2)]:
 for cf in [350, 280]:
    ceq = cf*parallel/series
    vl, vh = 2.2*series, 2.45*series
    reserve = 0.5*ceq*(vh**2-vl**2)
    charge = ceq*vl
    result['capacitor_cases'].append({
      'series': series, 'parallel': parallel, 'cell_F': cf, 'equivalent_F': ceq,
      'selection_status': '当前8S1P基线比较' if series == 8 else '未选用的串并对比，不是替代方案实施',
      'low_bank_V': vl, 'high_bank_V': vh,
      'ideal_energy_at_low_J': 0.5*ceq*vl**2,
      'ideal_energy_at_high_J': 0.5*ceq*vh**2,
      'ideal_reserve_low_to_high_J': reserve,
      'translation_energy_equivalent_stops': reserve/ek,
      'bank_V_after_one_translation_stop_from_low': math.sqrt(vl**2+2*ek/ceq),
      'bank_V_after_2x_translation_energy_from_low': math.sqrt(vl**2+4*ek/ceq),
      'maximum_initial_voltage_for_translation_stop_V': math.sqrt(vh**2-2*ek/ceq),
      'initial_charge_comparison': [
        {'time_s_comparison_only': t, 'ideal_constant_bank_charge_A': charge/t,
         'ideal_per_parallel_cell_charge_A': charge/t/parallel,
         'end_cap_power_W': charge/t*vl, 'mean_cap_power_W': 0.5*charge/t*vl}
        for t in [30, 60]],
      'cap_side_current_for_reference_power': [
         {'power_W_comparison_only': p, 'current_A_ignoring_losses': p/vl}
         for p in [500, 1000]],
      'notes': '没有ESR、能量转换损耗、末段降流、监测无效区、滤波/截止延迟；不构成保证容量/功率或停车能力。'
    })

(OUT/'制动储能筛查.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(f'm={mass:g}kg v={speed_kmh:g}km/h E_translation={ek:.3f}J')
for c in result['capacitor_cases']:
    print(f"{c['series']}S{c['parallel']}P {c['cell_F']}F/cell reserve={c['ideal_reserve_low_to_high_J']:.3f}J "
          f"I_30s={c['initial_charge_comparison'][0]['ideal_constant_bank_charge_A']:.3f}A "
          f"I_60s={c['initial_charge_comparison'][1]['ideal_constant_bank_charge_A']:.3f}A")
