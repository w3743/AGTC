"""独立核算AFE改造的电压、失电与门控边界；不代表硬件/固件验证。"""
from pathlib import Path
import hashlib
import itertools
import json
import runpy

MODULE = Path(__file__).resolve().parents[2]
CONFIG = MODULE/'原理图准备/AFE启动配置.json'
config = json.loads(CONFIG.read_text(encoding='utf-8'))
params = config['software_contract']
registers = {int(r['address'],16):int(r['value'],16) for r in config['registers']}
cap_model_module = runpy.run_path(str(MODULE/'仿真/源码/增容筛选.py'))
derive_supercapacitor_model = cap_model_module['derive_supercapacitor_model']
simulate = cap_model_module['simulate']
verify_simulation_api = cap_model_module['verify_simulation_api']
verify_cancelled_configuration_rejected = cap_model_module['verify_cancelled_configuration_rejected']
supercapacitor = derive_supercapacitor_model(config)
sensitivity = config['energy_sensitivity']
api_validation = verify_simulation_api()
api_validation['cancelled_16_cell_configuration_rejected'] = verify_cancelled_configuration_rejected(config)
checks = []

def check(name, passed, **evidence):
    checks.append(dict(name=name, passed=bool(passed), **evidence))

# Physical channel map is derived independently from the 8S wiring, not copied
# from the bitmask: seven bottom intervals plus VC16-VC15.
physical_channels = list(range(1,8))+[16]
expected_mask = sum(1 << (n-1) for n in physical_channels)
check('eight_real_channels_only', registers[0x9304] == expected_mask,
      expected_mask=hex(expected_mask), channels=physical_channels)
check('no_fake_current_protection', registers[0x9261] == 0x0c)
check('fast_digital_fault_output_mapping', registers[0x9265] in (0x18,0x98)
      and registers[0x9269] == 0xe4)
check('autonomous_output_logic_enabled', registers[0x9308]&8 and
      registers[0x9343]&16 and not registers[0x9308]&1)
check('no_charge_pump_no_internal_balancing', registers[0x9309] == 0 and
      registers[0x9335] == 0x10)
check('healthy_high_logic', all(registers[a] == 0xaa for a in (0x9301,0x9302)))
check('selected_supercapacitor_is_current_8S1P',
      supercapacitor['series_groups'] == 8 and
      supercapacitor['parallel_cells_per_group'] == 1 and
      supercapacitor['cell_count'] == 8 and
      supercapacitor['cell_count'] == supercapacitor['series_groups'] * supercapacitor['parallel_cells_per_group'],
      manufacturer_part=supercapacitor['manufacturer_part'],
      series_groups=supercapacitor['series_groups'],
      parallel_cells_per_group=supercapacitor['parallel_cells_per_group'],
      cell_count=supercapacitor['cell_count'])

# Conservative resistor tolerance + 100 ppm/C x 100 C absolute temperature
# deviation for logic resistors. 3.0V is the specified REG1 minimum, not nominal.
rtol = .01+.01
series_min, series_max = 680*(1-rtol), 680*(1+rtol)
pulldown_min = 10000*(1-rtol)
gate_high_min = 2.4*pulldown_min/(series_max+pulldown_min)
fault_sink_at_vol = (3.6-.4)/series_min
check('drv_enable_high_level', gate_high_min > 2.2,
      node_V_min=gate_high_min, drv_VIH_max_V=2.2,
      assumed_and_VOH_min_V=2.4, and_VDD_min_V=3.0)
check('fault_output_sink', fault_sink_at_vol <= .005,
      current_at_0p4V_A=fault_sink_at_vol, qualified_sink_A=.005)
truth_table = []
for charge_ok, discharge_ok, mcu_ok, total_ov_ok in itertools.product((False,True),repeat=4):
    enabled=charge_ok and discharge_ok and mcu_ok and total_ov_ok
    truth_table.append(dict(charge_ok=charge_ok, discharge_ok=discharge_ok,
                           mcu_ok=mcu_ok,total_ov_ok=total_ov_ok,enabled=enabled))
check('all_four_permissions_required', sum(x['enabled'] for x in truth_table)==1,
      rows=truth_table)

# Output pulldowns ensure the isolated MCU pins stay near ground. Their normal
# load must also be included in ADC transfer functions.
pd = 47000.
ron_max = 4.5
off_v = 2e-6*pd*1.01
def transfer(top,bottom):
    bottom_with_output_load = 1/(1/bottom+1/(pd+ron_max))
    return 1/(1+top/bottom_with_output_load)*pd/(pd+ron_max)
bus_gain = transfer(300000.,10000.)
sc_gain = transfer(68000.,10000.)
imon_gain = transfer(10000.,10000.)
check('mcu_off_analog_pins', off_v < .3,
      voltage_bound_V=off_v, assumed_full_temperature_Ipoff_A=2e-6)
check('tmux_off_signal_range', 80/(1+300000*.999/(10000*1.001)) < 3.6
      and 21.6/(1+68000*.999/(10000*1.001)) < 3.6
      and 6/(1+10000*.999/(10000*1.001)) < 3.6,
      raw_bus_at_80V_V=80/31,raw_sc_at_21p6V_V=21.6/7.8,
      raw_imon_at_drv_absolute_6V_V=3.,
      warning='Absolute maximum is only a containment check, never an operating permission.')
transfer_functions = dict(bus_volts_per_adc_volt=1/bus_gain,
                         sc_total_volts_per_adc_volt=1/sc_gain,
                         csa_volts_per_adc_volt=1/imon_gain,
                         imon_nominal_zero_V=1.65*imon_gain,
                         imon_nominal_V_per_A=.010*imon_gain,
                         imon_adc_12bit_step_A=(3.3/4096)/(.010*imon_gain),
                         imon_filter_approx_tau_s=(10000*10000/(20000))*100e-12)

# REGIN supply at normal lower stack voltage. REG1 serves only the logic gate
# and healthy outputs. 6mA is a design ceiling incl. fault-loaded output, not a
# measured consumption. Soft-start and actual rail behavior remain qualification.
iregin_budget=.006
available=(13.6-1.-5.8-.2)/(1000*1.01)
check('preregulator_static_load_budget', available>iregin_budget,
      available_A=available, allocation_A=iregin_budget,
      min_stack_V=13.6, diode_drop_assumption_V=1.,
      max_prereg_output_V=5.8, transistor_headroom_allocation_V=.2)

group_c_min = float(supercapacitor['eol_capacitance_per_group_F'])
max_i = params['provisional_peak_current_screen_A']
soft_dv = max_i/group_c_min*params['soft_trip_inclusive_age_to_zero_pwm_budget_s']
hard_dv = max_i/group_c_min*params['hard_trip_complete_response_budget_s']
e = params['cell_measurement_error_allocation_V']
soft_uv_bound=params['normal_cell_soft_uv_V']-e-soft_dv
soft_ov_bound=params['normal_cell_soft_ov_V']+e+soft_dv
hard_ov_bound=registers[0x9278]*.0506+params['hard_cov_error_V']+hard_dv+.004
hard_uv_bound=registers[0x9275]*.0506-params['hard_cuv_error_conservative_V']-hard_dv-.004
check('soft_uv_stops_before_operating_floor', soft_uv_bound>=1.7,
      true_voltage_lower_bound_V=soft_uv_bound, floor_V=1.7)
check('soft_ov_within_cell_rating', soft_ov_bound<2.7,
      true_voltage_upper_bound_V=soft_ov_bound,rating_V=2.7)
check('hard_ov_backstop_conditional', hard_ov_bound<2.7,
      voltage_upper_bound_V=hard_ov_bound,response_budget_s=.15,
      note='Response time budget requires fault injection; not a manufacturer timing guarantee.')
check('hard_uv_avoids_reversal_conditional', hard_uv_bound>0,
      voltage_lower_bound_V=hard_uv_bound,
      note='Emergency threshold permits loss of normal voltage window; it does not prove sustained power.')

# Use the current 8S1P equivalent stack capacitance and configured ESR model.
cap = float(supercapacitor['eol_stack_equivalent_capacitance_F'])
idle=params['idle_internal_target_V']
bias=params['idle_target_error_allocation_V']
deviation=params['max_true_group_deviation_from_mean_allocation_V']
energy_rows=[]
series = int(supercapacitor['series_groups'])
ripple_half_peak = float(sensitivity['ripple_bus_voltage_V']) / (
    8 * float(sensitivity['inductor_min_H']) * float(sensitivity['switching_frequency_Hz'])
)
energy_scenarios = [
    ('eol_parameter_sensitivity', float(sensitivity['nominal_efficiency_assumption']),
     float(supercapacitor['eol_stack_series_resistance_ohm']), 0.,
     'PARAMETER_SENSITIVITY_ONLY'),
    ('unqualified_extra_loss_sensitivity', float(sensitivity['unqualified_stress_efficiency_assumption']),
     float(supercapacitor['extra_loss_stack_series_resistance_ohm']),
     float(sensitivity['unqualified_start_voltage_error_V']),
     'UNQUALIFIED_SENSITIVITY_ONLY'),
]
for scenario_name, eta, resistance, start_error, qualification in energy_scenarios:
    for power in sensitivity['illustrative_bus_power_W']:
        for duration in sensitivity['illustrative_durations_s']:
            row=dict(scenario=scenario_name, qualification=qualification,
                     power_W=float(power), duration_s=float(duration),
                     duration_role='one_second_reference_only' if float(duration)==1.0 else 'illustrative_sensitivity_only')
            for direction in ('support','absorb'):
                initial=idle+(-start_error if direction=='support' else start_error)
                coarse=simulate(cap,resistance,initial,float(power),eta,direction,.0001,float(duration))
                fine=simulate(cap,resistance,initial,float(power),eta,direction,.00005,float(duration))
                if coarse['trajectory_complete'] and fine['trajectory_complete']:
                    difference=max(abs(coarse[k]-fine[k]) for k in ('minimum_terminal_V','maximum_terminal_V'))
                    coarse['terminal_voltage_timestep_difference_V']=difference
                    coarse['timestep_convergence_pass']=difference<.001
                    coarse['refined_energy_balance_residual_J']=fine['energy_balance_residual_J']
                    peak=float(coarse['maximum_average_port_current_A'])+ripple_half_peak
                    if direction=='support':
                        measured_worst=coarse['minimum_terminal_V']/series-deviation-e
                        voltage_ok=measured_worst>=params['normal_cell_soft_uv_V']
                    else:
                        measured_worst=coarse['maximum_terminal_V']/series+deviation+e
                        voltage_ok=measured_worst<=params['normal_cell_soft_ov_V']
                    coarse['estimated_inductor_peak_A']=peak
                    coarse['adverse_measured_cell_V']=measured_worst
                    coarse['conditional_screen_pass']=bool(voltage_ok and peak<=max_i and coarse['timestep_convergence_pass'])
                else:
                    coarse['terminal_voltage_timestep_difference_V']=None
                    coarse['timestep_convergence_pass']=False
                    coarse['estimated_inductor_peak_A']=None
                    coarse['adverse_measured_cell_V']=None
                    coarse['conditional_screen_pass']=False
                    coarse['screen_failure_reason']='constant-power branch has no real solution before requested end'
                row[direction]=coarse
            energy_rows.append(row)
check('energy_sensitivity_grid_is_current_configuration',
      len(energy_rows)==len(energy_scenarios)*len(sensitivity['illustrative_bus_power_W'])*len(sensitivity['illustrative_durations_s'])
      and all(api_validation.values()),
      cell_count=supercapacitor['cell_count'],
      durations_s=sensitivity['illustrative_durations_s'],
      invalid_duration_rejected=api_validation['nonpositive_duration_rejected'])

completed_trajectories=[v for row in energy_rows for d in ('support','absorb')
                        for v in (row[d],) if v['trajectory_complete']]
infeasible_trajectories=[dict(scenario=row['scenario'],power_W=row['power_W'],
                               duration_s=row['duration_s'],direction=d,
                               failure_time_s=row[d].get('failure_time_s'))
                         for row in energy_rows for d in ('support','absorb')
                         if not row[d]['trajectory_complete']]
energy_numerical_ok=bool(completed_trajectories) and all(
    abs(float(x['energy_balance_residual_J']))<.1 and
    abs(float(x['refined_energy_balance_residual_J']))<.1 and
    x['timestep_convergence_pass']
    for x in completed_trajectories)
result=dict(status='CONDITIONAL_DESIGN_CALCULATIONS_ONLY_NO_HARDWARE_RELEASE',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            capacitor_model_source_sha256=hashlib.sha256((MODULE/'仿真/源码/增容筛选.py').read_bytes()).hexdigest(),
            supercapacitor_model=supercapacitor,
            all_checks_pass=all(x['passed'] for x in checks),checks=checks,
            analog_transfer_functions=transfer_functions,
            energy_sensitivity=dict(requirement_status=sensitivity['requirement_status'],
                                    illustrative_bus_power_W=sensitivity['illustrative_bus_power_W'],
                                    illustrative_durations_s=sensitivity['illustrative_durations_s'],
                                    rows=energy_rows),
            numerical_verification=dict(completed_trajectories=len(completed_trajectories),
                                        infeasible_sensitivity_trajectories=infeasible_trajectories,
                                        energy_balance_and_step_convergence_pass=energy_numerical_ok,
                                        simulation_api_validation=api_validation),
            residuals=['Native netlist and actual selected component bindings require independent verification.',
                       'The 25mV measurement, 25mV group imbalance, 50ms/150ms response allocations require hardware tests.',
                       '80A peak screen is not a qualified hard current clamp. Calibrate CSA and validate COMP/TIM1 propagation.',
                       'High/low power MOS short and fuse clearing/energy isolation are separate unresolved fault cases.',
                       'No zero-volt autonomous recovery is implemented. Commissioning charging remains controlled.',
                       'Energy rows are 8S1P power-duration sensitivities only; startup power and duration are not specified and 1s is not a requirement.'])
out=MODULE/'仿真/结果/AFE保护边界验证.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(all_checks_pass=result['all_checks_pass'],checks=len(checks),
                      failed=[x['name'] for x in checks if not x['passed']],
                      energy_sensitivity_rows=len(energy_rows),
                      energy_sensitivity_numerics=energy_numerical_ok,
                      analog_transfer_functions=transfer_functions),ensure_ascii=True))
raise SystemExit(0 if result['all_checks_pass'] and energy_numerical_ok and all(api_validation.values()) else 2)
