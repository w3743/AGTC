"""当前稳压原理级边界：算术筛查，不替代波形、热或实体验证。"""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
live_path = root / '检查' / '本轮活跃工程只读核对.json'
live = json.loads(live_path.read_text(encoding='utf-8'))
pages_path = root / '检查' / '本轮四页活动会话官方检查.json'
pages = json.loads(pages_path.read_text(encoding='utf-8'))
assert live['live_instances'] == 166
assert not live['added'] and not live['removed'] and not live['pin_or_mpn_changes']
assert len(pages['pages']) == 4 and all(p['drc'] == [] for p in pages['pages'])

cell_f = 350.0
cells = 8
cell_target_v = 2.0
charge_a = 20.0
bank_f = cell_f / cells
bank_target_v = cells * cell_target_v
bulk_f = 6 * 900e-6
bus_low_v, bus_high_v = 37.8, 40.3
bus_delta_j = 0.5 * bulk_f * (bus_high_v**2 - bus_low_v**2)
reference_brake_w, reference_time_s = 495.3, 0.03
brake_energy_j = reference_brake_w * reference_time_s
isolation_l_h, current_a = 24e-6, 80.0
inductor_energy_j = 0.5 * isolation_l_h * current_a**2
new_c = 6 * 1000e-6
candidate_ripple_a = 6 * 4.5 * 1.19
ideal_ripple_a = current_a * math.sqrt(0.5 * 0.5)
out = {
    'status': 'SCREENING_ONLY_NOT_QUALIFIED',
    'source': {'live_readonly': str(live_path.relative_to(root)), 'sha256': hashlib.sha256(live_path.read_bytes()).hexdigest(),
               'active_page_drc': live['drc_current_page'], 'netlist_sha256': live['netlist_sha256'],
               'four_page_drc': str(pages_path.relative_to(root)), 'four_page_sha256': hashlib.sha256(pages_path.read_bytes()).hexdigest()},
    'zero_charge': {'assumptions': '8 equal 350F cells; ideal series bank; 20A constant; start 0V; ideal cell threshold 2V',
                    'series_capacitance_F': bank_f, 'target_bank_V': bank_target_v, 'seconds': bank_f * bank_target_v / charge_a,
                    'limitations': 'No ESR, current loop, cell mismatch, switching and gate-driver startup'},
    'regen': {'assumptions': 'local 5.4mF ideal capacitor, no losses, 37.8 to 40.3V; constant 495.3W injection for 30ms',
              'local_capacitor_J': bus_delta_j, 'injected_J': brake_energy_j, 'ideal_no_other_sink_time_ms': 1000*bus_delta_j/reference_brake_w,
              'limitations': 'Not full bus or braking response: FOC bus caps, controller, line, ESR, protection delay and load excluded'},
    'isolation_energy': {'L_uH': 24, 'I_A': current_a, 'stored_J': inductor_energy_j,
                         'limitations': 'No MOS fault waveform, parasitic or shutdown timing; gate signal is not proof of current interruption'},
    'candidate_BFT': {'six_C_nominal_uF': new_c*1e6,'rating_100kHz_105C_A_each':4.5*1.19,
                      'rating_100kHz_105C_A_six_ideal_equal_share':candidate_ripple_a,
                      'rectangular_bridge_AC_A_at_80A_D_0p5':ideal_ripple_a,
                      'limitations': 'Waveform RMS and single-frequency rating are not directly comparable; actual spectrum, current sharing, ESR and temperature conditions missing'},
    'required_proof': {'fast_loss_bidir_isolation':'NOT_PROVEN', 'zero_V_actual_start':'NOT_PROVEN',
                       'regen_full_envelope':'NOT_PROVEN', 'capacitor_ripple_and_ESR':'NOT_PROVEN',
                       'four_page_live_DRC':'PASS_LIVE_SESSION_ONLY', 'reopen_saved_source':'NOT_PROVEN',
                       'physical_testing':'OUT_OF_SCOPE'}
}
path = root / '检查' / '本轮原理核算.json'
path.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
