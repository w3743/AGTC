"""比较稳压保持电容独立充放电分支，使用包络二极管和理想早轨。"""

import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


OUT = Path(__file__).resolve().parent
POWER = OUT.parents[1]
EXE = POWER / 'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
R1 = POWER / '稳压模块/检查/候选_预充与超容隔离_R1'
NETLIST = '''Guard reserve comparison
VPROT prot 0 {early_supply}
VFAULT fault 0 PULSE(0 1 100u 1n 1n 1 2)
RCHARGE prot reserve {charge_r}
CRESERVE reserve 0 {cap_f} IC={initial_v}
BMAIN prot vin I={{max(0,(V(prot)-V(vin)-0.6)/0.1)}}
BRESERVE reserve vin I={{max(0,(V(reserve)-V(vin)-0.6)/0.1)}}
CIN vin 0 100n IC={vin_initial_v}
BLOAD vin 0 I={{min(0.105,max(0,0.105*V(vin)/4.8))}}
SFAULT {fault_node} 0 fault 0 SWFAULT
.model SWFAULT SW(Ron=0.01 Roff=1e12 Vt=0.5 Vh=0.1)
.control
set noaskquit
set wr_singlescale
tran {step} {stop} uic
wrdata result.txt v(prot) v(reserve) v(vin) i(VPROT)
quit
.endc
.end
'''


def run_case(work, early_supply='11.4', charge_r=425.7, cap_f='2.2u', initial_v=10.7,
             fault_node='dummy', step='100n', stop='400u'):
    netlist = NETLIST.format(
        early_supply=early_supply, charge_r=charge_r, cap_f=cap_f, initial_v=initial_v,
        vin_initial_v=10.7895, fault_node=fault_node, step=step, stop=stop,
    )
    (work / 'reserve.cir').write_text(netlist, encoding='ascii')
    wave = work / 'result.txt'
    wave.unlink(missing_ok=True)
    proc = subprocess.run([str(EXE), '-b', 'reserve.cir'], cwd=work, capture_output=True,
                          text=True, errors='replace', timeout=60,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if proc.returncode or 'Error:' in proc.stdout + proc.stderr or not wave.is_file():
        raise RuntimeError(proc.stdout + proc.stderr)
    rows = [tuple(map(float, line.split())) for line in wave.read_text(encoding='ascii').splitlines()]
    if len(rows) < 10:
        raise RuntimeError('保持电源波形不完整')
    return rows


def main():
    with TemporaryDirectory(prefix='agtc_guard_') as temp:
        work = Path(temp)
        lost = run_case(work, early_supply='PWL(0 11.4 100u 11.4 100.001u 0)')
        short = run_case(work, fault_node='reserve')
        common_short = run_case(work, fault_node='vin')
        startup = run_case(work, charge_r=434.3, cap_f='26.4u', initial_v=0, step='10u', stop='65m')
        resistor_open = run_case(work, charge_r=1e12, initial_v=0)

    hold_us = (next(row[0] for row in lost if row[0] > 100e-6 and row[3] < 4.8) - 100e-6) * 1e6
    startup_ms = next(row[0] for row in startup if row[2] >= 10.7) * 1e3
    source_files = (EXE, R1 / '保护页_保持与监控.json', R1 / '受限储能隔离_概念候选.svg')
    result = {
        'qualification': 'HOLD',
        'input_sha256': {str(path.relative_to(POWER.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in source_files},
        'scope': '电源拓扑筛查。理想 11.4 V 早轨、固定 0.6 V 加 0.1 Ω 的单向导通包络及 105 mA 压力负载。欠压监管、输出关栅、共点短路导致早轨的实际退化和器件损耗动态待核。',
        'startup_from_zero_ms_to_10p7_v': startup_ms,
        'startup_scope': '26.4 µF、434.3 Ω，无检测输入分压漏载，计二极管分流至105mA负载。',
        'early_rail_short_hold_above_4p8_v_us': hold_us,
        'early_rail_short_scope': '有效保持电容仅 2.2 µF，起始10.7 V条件，425.7 Ω 向短接早轨反向泄放。',
        'reserve_short_steady_ldo_input_v': short[-1][3],
        'reserve_short_steady_prot_current_a_in_model': -short[-1][4],
        'reserve_short_max_added_source_current_a_at_13p2_v': 13.2 / 425.7,
        'reserve_short_max_charge_resistor_power_w_at_13p2_v': 13.2**2 / 425.7,
        'charge_resistor_open_reserve_v_in_model': resistor_open[-1][2],
        'common_ldo_input_short_v_in_idealized_model': common_short[-1][3],
        'ldo_thermal_screen': {
            'datasheet': 'https://www.ti.com/lit/ds/symlink/tlv760.pdf',
            'specified_max_output_a': 0.1,
            'ambient_c': 60, 'operating_junction_limit_c': 125, 'datasheet_theta_ja_c_per_w': 275.2,
            'steady_output_limit_a_ignoring_ground_current_at_13p2_in': (125 - 60) / (275.2 * (13.2 - 3.3)),
            'scope': '105mA 为电源前端压力假设。实际 TPSI/逻辑脉冲和平均负载、稳压及散热待验证。',
        },
        'remaining': '欠压检测必须在电容未就绪、充电开路及电容短路时硬禁能；保持通路的二极管和公共 LDO 输入节点短路仍须分别处理。',
    }
    output = OUT / '保持电容短路隔离复核.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(output.name)


if __name__ == '__main__':
    main()
