"""Check pre-capture part contracts, harness pins, and fresh simulation evidence."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
POWER = HERE.parents[1]
CONTRACTS = {
    '稳压': '稳压模块/检查/闭合候选_P5Rev2/稳压P5Rev2施工合同.json',
    'BMS': 'BMS模块/检查/闭合候选_9S5P/BMS_9S5P施工合同.json',
    'FOC': 'FOC驱动模块/检查/闭合候选_164Rev2/FOC_164Rev2施工合同.json',
    '水泵': '水泵驱动模块/检查/闭合候选_隔离CAN/水泵闭合施工合同.json',
}
FIELDS = ('mpn', 'manufacturer', 'lcsc', 'footprint')
BOARD_INSTANCES = {'BMS': 'BMS', '稳压': '稳压', 'FOC-A': 'FOC', 'FOC-B': 'FOC', '水泵': '水泵'}
CAN_CONNECTOR = ('B3B-XH-A(LF)(SN)', 'C144394', 'CONN-TH_3P-P2.50_A2501WV2-3P')
CAN_CABLE_PINS = {'GND_CAN': '1', 'CANH': '2', 'CANL': '3'}
SIGNAL_CONNECTORS = {
    'BMS': {
        'J_LOAD_POWER_VALID': ('LOAD_EN_BIAS_EXT', 'LOAD_POWER_VALID'),
        'J_LOAD_POWER_VALID_B': ('LOAD_EN_BIAS_EXT_B', 'LOAD_POWER_VALID_B'),
        'J_LPV_REG': ('LOAD_VALID_REG_LED_A', 'LOAD_VALID_LED_K'),
        'J_EARLY_READY_IN': ('EARLY_GUARD_READY_N', 'B0'),
    },
    '稳压': {
        'J_LPV_IN': ('BMS_LPV_LED_A', 'BMS_LPV_LED_K'),
        'J_EARLY_READY': ('EARLY_GUARD_READY_N', 'B0_READY_RETURN'),
        'J_NTC_DUMP': ('NTC_DUMP', 'GND_SYS'),
        'J_NTC_MAIN': ('NTC_MAIN', 'GND_SYS'),
        'J_NTC_SC': ('NTC_SC', 'GND_SYS'),
        'J_5V_EXT': ('5V_EXT', 'GND_SYS'),
    },
    'FOC': {
        'J_LOAD_POWER_VALID': ('LOAD_EN_BIAS_EXT', 'LOAD_POWER_VALID'),
        'J_TEMP_MOS': ('TEMP_MOS_NODE', 'GND_SYS'),
        'J_TEMP_MOTOR': ('TEMP_MOTOR_NODE', 'GND_SYS'),
    },
    '水泵': {},
}
CORE_CONNECTORS = {
    'BMS': ('J_CORE_P1', 'J_CORE_P2'), '稳压': ('J_CORE_P1', 'J_CORE_P2'),
    'FOC': ('J_MCU_L', 'J_MCU_R'), '水泵': ('J_MCU_A', 'J_MCU_B'),
}
GREEN_CONNECTORS = {
    'BMS': {'J_R_PRE': ('QPRE_SW', 'PACK+')},
    '水泵': {'J_BUS': ('BUS_M', 'GND_SYS'), 'J_MOTOR': ('BUS_M', 'MOTOR_B')},
}
STUD_NETS = {
    'BMS': {'J_BATT_POS': 'BATT+_FUSED', 'J_BATT_NEG': 'B0', 'J_PACK_POS': 'PACK+', 'J_PACK_NEG': 'PACK-'},
    '稳压': {
        'J_BUS_GND': 'GND_SYS', 'J_BUS_POS': 'BUS_EXT',
        'J_DUMP_BUS': 'BUS_EXT', 'J_DUMP_SW': 'DUMP_SW',
        'J_FUSE_ISO': 'SC_ISO_STORAGE', 'J_FUSE_SC': 'SC_PLUS',
        'J_L_SW': 'SW', 'J_L_SW_B': 'SW_B', 'J_L_SC': 'SC_FUSED',
    },
    'FOC': {
        'J_BUS_P': 'BUS', 'J_BUS_N': 'GND_SYS', 'J_MOTOR_U': 'PH_U',
        'J_MOTOR_V': 'PH_V', 'J_MOTOR_W': 'PH_W',
    },
}
CELL_TAP_PINS = {'1': 'B0', **{str(tap + 1): f'TAP_B{tap}' for tap in range(1, 10)}}
SC_TAP_PINS = dict(zip('123456789', ('SC_MINUS', *(f'T{i}' for i in range(1, 8)), 'SC_PLUS')))
HALL_SUPPLY_NET = '5V_HALL_PROT'
SIGNAL_PART = ('A2501WV-2P', 'C48579979', 'CONN-TH_2P-P2.50_A2501WV-2P')
GREEN_PART = ('2EDGVC-5.08-02P-14-100A(H)', 'C581328', 'CONN-TH_P5.08_2EDGVC-5.08-02P-14-100A-H')
CORE_PART = ('JXT-PM2.54-2X12P-A9', 'C54803369', 'HDR-TH_24P-P2.54-V-F-R2-C12-S2.54-2')
FOC_CONFIG_REFS = ('JP_HA', 'JP_HB', 'JP_HC')
FROZEN_PASSIVES = {
    'C30926': ('0603B104K500NT', 'C0603'),
    'C89258': ('0805B104K101NT', 'C0805'),
    'C21190': ('0603WAF1001T5E', 'R0603'),
    'C25804': ('0603WAF1002T5E', 'R0603'),
    'C25803': ('0603WAF1003T5E', 'R0603'),
    'C23162': ('0603WAF4701T5E', 'R0603'),
}
MOSFET_ROLES = {str(pin): ('G' if pin == 1 else 'D' if pin == 9 else 'S') for pin in range(1, 10)}
CAN_ROLES = dict(zip(map(str, range(1, 17)),
                     'RXD NC NC TXD GNDP1 VDDP VDDL GND1 GND2 VISOIN VISOOUT GNDP2 NC CANL CANH NC'.split()))
OPTO_ROLES = dict(zip('1234', 'A K E C'.split()))
CAN_LOGIC_NETS = {
    'BMS': ('CAN_RX', 'CAN_TX', 'B0', 'VOUT_5V', 'CORE_3V3'),
    '稳压': ('CAN_RX', 'CAN_TX', 'GND_SYS', '5V_EXT', '3V3_CORE'),
    'FOC': ('FDCAN1_RX', 'FDCAN1_TX', 'GND_SYS', '5V_BUCK', 'MCU_3V3'),
    '水泵': ('CAN_RX', 'CAN_TX', 'GND_SYS', '+5V', '+3V3'),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def native_components(path):
    return {row['props']['Designator']: row for row in read(path)['components'].values()}


def native_pins(row):
    return {pin: value['net'] or 'NC' for pin, value in row['pinInfoMap'].items()}


def declared_refs(node):
    if isinstance(node, list):
        for child in node:
            yield from declared_refs(child)
    elif isinstance(node, dict) and node.get('placement') != 'off-board':
        if 'ref' in node and any(key in node for key in ('mpn', 'new_mpn', 'pins')):
            yield node['ref']
        for key, child in node.items():
            if key == 'pin_updates':
                yield from (label.partition('.')[0] for label in child)
            elif key != 'pins':
                yield from declared_refs(child)


def check_components(board, contract):
    findings = []
    rows = contract.get('drawing_components', [])
    by_ref = {}
    for item in rows:
        if item.get('placement') == 'off-board' or item.get('on_board') is False:
            continue
        ref = item.get('ref', '')
        if not re.fullmatch(r'[A-Za-z][\w-]*', ref) or ref in by_ref:
            findings.append(f'{ref!r}：位号为空、格式非法或重复')
            continue
        by_ref[ref] = item
        retained_jumper = board == 'FOC' and ref in FOC_CONFIG_REFS
        generic_hardware = ref in STUD_NETS.get(board, {}) or retained_jumper
        for field in FIELDS:
            if (generic_hardware and field in ('manufacturer', 'lcsc')) or (retained_jumper and field == 'mpn'):
                continue
            value = item.get(field)
            if (not isinstance(value, str) or not value.strip()
                    or re.search(r'UNVERIFIED|SYMBOL_ONLY|待核|待定|UNKNOWN|�', value, re.I)):
                findings.append(f'{ref}.{field}：缺少已核实的字段')
        code = item.get('lcsc')
        if not generic_hardware and (not isinstance(code, str) or not re.fullmatch(r'C[1-9]\d*', code)
                                     or re.fullmatch(r'C99000\d{5}', code)):
            findings.append(f'{ref}.lcsc：须为精确 C 码')
        nets = item.get('pins')
        if not isinstance(nets, dict) or not nets or any(
            not re.fullmatch(r'[1-9]\d*|EP|PAD', pin)
            or not isinstance(net, str) or not net.strip()
            or re.search(r'->|→|待核|待定|UNVERIFIED', net, re.I) for pin, net in nets.items()
        ):
            findings.append(f'{ref}.pins：须提供逐针网名或 NC')
    expected = set(declared_refs(contract.get('changes', [])))
    expected.update(declared_refs(contract.get('in_place_upgrades_for_final_capture', [])))
    expected.update(declared_refs(contract.get('p0_phase_a', {})))
    expected.update(contract.get('passive_freeze', {}).get('refs', {}))
    scope = contract.get('drawing_scope', {})
    for key in ('new_replaced_or_rewired', 'inherited_changed_or_added', 'closure_changed_or_added'):
        expected.update(scope.get(key, []))
    if board == 'BMS':
        baseline = POWER.parent / scope['baseline']
        published = POWER.parent / scope['published_baseline']
        previous = native_components(published)
        if scope['baseline_sha256'] != digest(baseline):
            findings.append('279 器件预充候选网表哈希不一致')
        if scope['published_baseline_sha256'] != digest(published):
            findings.append('224 器件发布网表哈希不一致')
        for ref, row in native_components(baseline).items():
            if row['props'].get('Convert to PCB') != 'yes':
                continue
            old = previous.get(ref)
            if (old is None or native_pins(row) != native_pins(old)
                    or row['props'].get('Manufacturer Part') != old['props'].get('Manufacturer Part')):
                expected.add(ref)
        expected.update(('J_R_PRE', 'J_LPV_REG', 'Q_BREG', 'C_VC1', 'J_CELL_SERVICE'))
        if any(item.get('ref') == 'J_CELL_SERVICE' for item in contract.get('fabricated_interfaces', [])):
            findings.append('J_CELL_SERVICE：旧焊线孔和新插座重复定义')
        removed = set(scope.get('removed_from_frozen_enet', []))
        if removed != {f'R_PSU_MINLOAD{index}' for index in range(1, 6)} or removed & by_ref.keys():
            findings.append('PSU5_EXT：五颗旧预载须全部在施工删除清单中')
        expected.difference_update(removed)
        delta = contract['additional_delta_on_top_of_precharge_contract']
        if len(native_components(baseline)) + delta['added'] - delta['removed'] != delta['net_total_components']:
            findings.append('BMS：目标原理图对象数与底稿及增减数不一致')
    if board == '稳压':
        expected.update(('Q_MAIN_H', 'Q_MAIN_L', 'Q_DUMP', 'R_DRV_GH_SER', 'R_DRV_GL_SER'))
        expected.update(f'C_BUS_BULK{index}' for index in range(1, 7))
        if digest(POWER.parent / contract['baseline']['enet']) != contract['baseline']['enet_sha256']:
            findings.append('234 器件 P5 候选网表哈希不一致')
        delta = contract['net_delta_vs_p5']
        if contract['baseline']['component_count'] + delta['added'] - delta['removed'] != contract['target_component_count']:
            findings.append('稳压：目标原理图对象数与 P5 底稿及增减数不一致')
        for ref, gate in [('R_DRV_GH_SER', 'H'), ('R_DRV_GL_SER', 'L')]:
            item = by_ref.get(ref, {})
            if (item.get('mpn') != '25121WF220KT4E' or item.get('lcsc') != 'C620867'
                    or item.get('footprint') != 'R2512'
                    or item.get('pins') != {'1': f'DRV_G{gate}', '2': f'GATE_{gate}'}):
                findings.append(f'{ref}：2.2 Ω / 1 W / 2512 料号或针网错误')
    if board == 'FOC':
        published_path = POWER.parent / contract['drawing_component_baseline']['published_pin_map']
        baseline_path = POWER.parent / contract['baseline']['candidate_164_enet']
        previous = {item['ref']: item for item in read(published_path)['components']}
        for ref, row in native_components(baseline_path).items():
            if row['props'].get('Convert to PCB') != 'yes':
                continue
            old = previous.get(ref)
            if (old is None or native_pins(row) != {pin: net or 'NC' for pin, net in old['pins'].items()}
                    or row['props'].get('Manufacturer Part') != old['manufacturerId']):
                expected.add(ref)
        expected.update(part['ref'] for part in contract['localization_v1']['main_mos']['reference_pins'])
    if board == '水泵':
        local = contract['localization_v1']
        expected.update(item['ref'] if isinstance(item, dict) else item for item in local['add'])
        expected.update(item['ref'] if isinstance(item, dict) else item for item in local['reuse_pins'])
    missing = sorted(expected - by_ref.keys())
    if missing or not by_ref:
        findings.append(f'施工覆盖不足：{missing}')
    count = contract.get('target_component_count')
    if board == 'BMS':
        count = contract['additional_delta_on_top_of_precharge_contract']['net_total_components']
    core_count = len(contract.get('changes', contract.get('in_place_upgrades_for_final_capture', [])))
    return by_ref, {
        'status': 'FAIL' if findings else 'PASS',
        'target_schematic_count': count,
        'core_change_groups': core_count,
        'drawing_entries': len(by_ref),
        'required_entries': len(expected),
        'findings': findings,
    }


def check_native_mos_mapping(components, regulator, foc):
    failures = []
    for board, contract, baseline, references, old_drain, old_source in (
        ('稳压', regulator, 'enet', ('Q_MAIN_H', 'Q_MAIN_L', 'Q_DUMP'), '2', '3'),
        # The Infineon EasyEDA symbol joins physical source pads 2–8 and
        # uses its logical pin 3 for the TOLL exposed drain pad.
        ('FOC', foc, 'candidate_164_enet', tuple(f'Q{i}' for i in range(1, 7)), '3', '2'),
    ):
        originals = native_components(POWER.parent / contract['baseline'][baseline])
        for ref in references:
            old = native_pins(originals[ref])
            new = components[board].get(ref, {}).get('pins', {})
            # Phase A low-side source now returns through its own Kelvin shunt.
            source = 'PHASE_A_LS' if board == '稳压' and ref == 'Q_MAIN_L' else old.get(old_source)
            if (new.get('1'), new.get('9'), new.get('2')) != (old.get('1'), old.get(old_drain), source):
                failures.append(f'{board}.{ref}：替换后的栅、漏、源目标网与冻结 ENET 不一致')
    return failures


def check_pin_roles(components):
    failures = []
    checked = []
    for board, items in components.items():
        for ref, item in items.items():
            part = item.get('mpn')
            pins = item.get('pins', {})
            label = f'{board}.{ref}'
            roles = None
            valid = True
            if part == 'HYG022N10NS1TA':
                roles = MOSFET_ROLES
                valid = (item['lcsc'] == 'C2925050' and item['footprint'].startswith('TOLL')
                         and len({pins.get(str(pin)) for pin in range(2, 9)}) == 1
                         and len({pins.get('1'), pins.get('2'), pins.get('9')}) == 3)
                if item.get('old_pin_roles') == {'1': 'G', '2': 'D', '3': 'S'}:
                    old = item['old_pins']
                    source = 'PHASE_A_LS' if board == '稳压' and ref == 'Q_MAIN_L' else old['3']
                    valid &= (pins.get('1'), pins.get('2'), pins.get('9')) == (old['1'], source, old['2'])
            elif part == 'BCX56-16':
                roles = dict(zip('123', 'B C E'.split()))
                valid = (item['lcsc'] == 'C5183147'
                         and pins == {'1': 'BREG', '2': 'BREG_COLLECTOR', '3': 'REGIN'}
                         and pins == item.get('old_pins')
                         and item.get('exposed_pad') == {
                             'physical_tab': 'C', 'shares_pad': '2', 'net': pins['2'],
                             'if_separately_numbered': '4',
                         })
            elif part == 'CA-IS2062A':
                roles = CAN_ROLES
                rx, tx, primary_ground, power_rail, logic_rail = CAN_LOGIC_NETS[board]
                valid = (item['lcsc'] == 'C22438481' and item['footprint'].startswith('LGA-16')
                         and all(pins.get(pin) == 'NC' for pin in ('2', '3', '13', '16'))
                         and pins.get('9') == pins.get('12') == 'GND_CAN'
                         and pins.get('5') == pins.get('8') == primary_ground
                         and pins.get('10') == pins.get('11') == 'CAN_ISO_5V'
                         and (pins.get('1'), pins.get('4'), pins.get('6'), pins.get('7'))
                         == (rx, tx, power_rail, logic_rail))
            elif part == 'LTV-817S-TA1-C':
                roles = OPTO_ROLES
                valid = item['lcsc'] == 'C109227' and len(set(pins.values())) == 4 and 'NC' not in pins.values()
            elif part == 'LKMI2001J102MF':
                roles = {'1': '+', '2': '-'}
                valid = (item['lcsc'] == 'C442962'
                         and pins == {'1': 'BUS_EXT', '2': 'GND_SYS'}
                         and item.get('footprint_geometry_mm') == {
                             'can_diameter': 16, 'can_height': 20, 'pitch': 7.5, 'lead_diameter': .8,
                         })
            if roles:
                checked.append(label)
                declared_roles = item.get('pin_roles')
                if not valid or set(roles) != set(pins) or declared_roles != roles:
                    failures.append(f'{label}：关键器件封装、功能针号或目标网不匹配')
    required = {
        '稳压.Q_MAIN_H', '稳压.Q_MAIN_L', '稳压.Q_MAIN_BH', '稳压.Q_MAIN_BL', '稳压.Q_DUMP', '稳压.U_CAN1',
        '稳压.U_LPV_OPTO', '稳压.U_EARLY_READY_OPTO', 'BMS.Q_BREG', 'BMS.U_CAN',
        'BMS.Q_CHG4', 'BMS.Q_DSG4', 'BMS.U_LOAD_POWER_VALID',
        'BMS.U_LOAD_POWER_VALID_B', 'FOC.U5', '水泵.U_CAN',
        *(f'稳压.C_BUS_BULK{index}' for index in range(1, 7)),
        *(f'FOC.Q{index}' for index in range(1, 7)),
    }
    failures.extend(f'{ref}：缺少关键器件检查' for ref in sorted(required - set(checked)))
    return {'status': 'FAIL' if failures else 'PASS', 'checked': checked, 'findings': failures}


def check_harness(components, bms_contract):
    bms, reg, foc = (components[board] for board in ('BMS', '稳压', 'FOC'))
    errors = []
    can = {}
    for instance, board in BOARD_INSTANCES.items():
        items = components[board]
        transceivers = [item for item in items.values() if item.get('mpn') == 'CA-IS2062A']
        tvs = [item for item in items.values() if item.get('mpn') == 'PESD2CANFD24V-TR']
        connector = items.get('J_CAN', {})
        cable_pins = CAN_CABLE_PINS
        if len(transceivers) != 1 or len(tvs) != 1:
            errors.append(f'{instance}：缺少唯一的隔离 CAN 收发器及 TVS')
            continue
        device = transceivers[0]['pins']
        can_h, can_l = ('CANH_CONN', 'CANL_CONN') if board == 'FOC' else ('CANH', 'CANL')
        if (tuple(connector.get(key) for key in ('mpn', 'lcsc', 'footprint')) != CAN_CONNECTOR
                or connector.get('pins') != {'1': 'GND_CAN', '2': can_h, '3': can_l}
                or (device.get('9'), device.get('12'), device.get('15'), device.get('14'))
                != ('GND_CAN', 'GND_CAN', can_h, can_l)
                or tvs[0]['pins'] != {'1': device.get('15'), '2': device.get('14'), '3': 'GND_CAN'}):
            errors.append(f'{instance}：CAN 连接器、收发器或 TVS 针网不一致')
        primary = {device.get(pin) for pin in ('1', '4', '5', '6', '7', '8')}
        for ref, item in items.items():
            nets = set(item.get('pins', {}).values())
            if item is not transceivers[0] and 'GND_CAN' in nets and (nets & primary):
                errors.append(f'{instance}.{ref}：GND_CAN 与逻辑侧针网同时出现')
        can[instance] = dict(cable_pins)

    def nets(rows, ref):
        return rows.get(ref, {}).get('pins', {})

    lpv = {}
    for side, suffix in [('FOC-A', ''), ('FOC-B', '_B')]:
        bms_ref = f'J_LOAD_POWER_VALID{suffix}'
        opto = nets(bms, f'U_LOAD_POWER_VALID{suffix}')
        if (nets(bms, bms_ref) != {'1': f'LOAD_EN_BIAS_EXT{suffix}', '2': f'LOAD_POWER_VALID{suffix}'}
                or opto.get('4') != f'LOAD_EN_BIAS_EXT{suffix}'
                or opto.get('3') != f'LOAD_POWER_VALID{suffix}'
                or nets(foc, 'J_LOAD_POWER_VALID') != {'1': 'LOAD_EN_BIAS_EXT', '2': 'LOAD_POWER_VALID'}
                or nets(foc, 'R_LOAD_VALID_BIAS') != {'1': 'LOAD_EN_BIAS_EXT', '2': 'SAFE_5V'}):
            errors.append(f'{side}：LPV 独立偏置或信号针位不匹配')
        lpv[side] = [bms_ref, 'J_LOAD_POWER_VALID', {'1': '1', '2': '2'}]
    reg_lpv = nets(reg, 'U_LPV_OPTO')
    if (nets(bms, 'J_LPV_REG') != {'1': 'LOAD_VALID_REG_LED_A', '2': 'LOAD_VALID_LED_K'}
            or nets(reg, 'J_LPV_IN') != {'1': 'BMS_LPV_LED_A', '2': 'BMS_LPV_LED_K'}
            or (reg_lpv.get('1'), reg_lpv.get('2')) != ('BMS_LPV_LED_A', 'BMS_LPV_LED_K')):
        errors.append('稳压↔BMS：LPV LED 电流环极性或针位错误')
    early = nets(reg, 'U_EARLY_READY_OPTO')
    if (nets(reg, 'J_EARLY_READY') != {'1': 'EARLY_GUARD_READY_N', '2': 'B0_READY_RETURN'}
            or (early.get('4'), early.get('3')) != ('EARLY_GUARD_READY_N', 'B0_READY_RETURN')
            or nets(bms, 'J_EARLY_READY_IN') != {'1': 'EARLY_GUARD_READY_N', '2': 'B0'}
            or nets(bms, 'R_EARLY_READY_PU') != {'1': 'CORE_3V3', '2': 'EARLY_GUARD_READY_N'}
            or nets(bms, 'J_CORE_P2').get('12') != 'EARLY_GUARD_READY_N'):
        errors.append('稳压↔BMS：EARLY_READY 集电极/发射极、接收端或 J_CORE_P2.12 针位错误')
    early_change = next(change for change in bms_contract['changes'] if change['id'] == 'CHG-BMS-05')
    core = early_change['coreboard_mapping']
    core_audit = read(POWER.parent / core['source'])
    gpio = next(row['mcu_gpio'] for row in core_audit['pin_map']['P2'] if row['pin'] == 12)
    if (core['coreboard_header'], core['pin'], core['mcu_gpio']) != ('P2', 12, gpio) or gpio != 'PC4':
        errors.append('BMS EARLY_READY：WeAct 官方 P2.12 对应 PC4')
    return {
        'status': 'FAIL' if errors else 'PASS', 'can_signal_to_connector_pin': can,
        'can_harness_pin_to_pin': dict(zip('123', '123')),
        'lpv_foc': lpv, 'lpv_regulator': ['J_LPV_REG', 'J_LPV_IN', {'1': '1', '2': '2'}],
        'early_ready': ['J_EARLY_READY', 'J_EARLY_READY_IN', {'1': '1', '2': '2'}, {'J_CORE_P2.12': gpio}],
        'findings': errors,
    }


def check_connector_standard(components, contracts, procurement):
    errors = []
    quantity = Counter()
    old_skus = set()
    new_catalog_skus = set()
    new_mechanical_specs = set()
    configuration = {}
    pos24 = {str(pin) for pin in range(1, 25)}
    frozen_pump = native_components(POWER.parent / contracts['水泵']['baseline']['published_enet'])
    pump_bus = frozen_pump['J_BUS']
    pump_props = pump_bus['props']
    retained = {'水泵': {'J_BUS': {
        'mpn': pump_props['Manufacturer Part'], 'lcsc': pump_props['Supplier Part'],
        'footprint': pump_props['FootprintName'], 'pins': native_pins(pump_bus),
    }}}

    def check(ref, family, part, pins=None):
        row = items.get(ref, {})
        quantity[family] += 1
        checked.add(ref)
        identity = tuple(row.get(key) for key in ('mpn', 'lcsc', 'footprint'))
        actual = row.get('pins', {})
        if identity != part or (pins is not None and actual != pins):
            errors.append(f'{instance}.{ref}：{family} 料号、封装或针网不一致')
        if all(isinstance(field, str) for field in identity[:2]):
            new_catalog_skus.add(identity[:2])

    for instance, board in BOARD_INSTANCES.items():
        items = {**retained.get(board, {}), **components[board]}
        before = contracts[board].get('connector_standard', {}).get('before', {})
        checked = set()
        check('J_CAN', 'CAN_XH_3P', CAN_CONNECTOR)
        for ref, pins in SIGNAL_CONNECTORS[board].items():
            check(ref, 'signal_XH_2P', SIGNAL_PART, dict(zip('12', pins)))
        for ref, pins in GREEN_CONNECTORS.get(board, {}).items():
            check(ref, 'power_5p08_2P', GREEN_PART, dict(zip('12', pins)))
        for ref in CORE_CONNECTORS[board]:
            check(ref, 'core_2x12P', CORE_PART)
            if set(items.get(ref, {}).get('pins', {})) != pos24:
                errors.append(f'{instance}.{ref}：排母须有 24 针')

        if board == 'BMS':
            check('J_CELL_SERVICE', 'cell_XH_10P',
                  ('B10B-XH-A(LF)(SN)', 'C144400', 'CONN-TH_10P-P2.50_B10B-XH-A'), CELL_TAP_PINS)
        if board == '稳压':
            check('J_SC_TAPS', 'supercap_XH_9P',
                  ('B9B-XH-A(LF)(SN)', 'C157966', 'CONN-TH_B9B-XH-A-LF-SN'), SC_TAP_PINS)
            configuration[instance] = ['J_CAN_TERM']
            term = items.get('J_CAN_TERM', {})
            if (term.get('lcsc'), term.get('footprint'), term.get('pins')) != (
                    'C41430884', 'HDR-TH_2541WV-2X01P', {'1': 'CANH', '2': 'TERM_IN'}):
                errors.append('稳压.J_CAN_TERM：保留原 2.54 mm 排针及针序')
        if board == 'FOC':
            check('J_HALL', 'hall_PH_5P',
                  ('B5B-PH-K-S(LF)(SN)', 'C157993', 'CONN-TH_B5B-PH-K-S-LF-SN'),
                  dict(zip('12345', (HALL_SUPPLY_NET, 'HALL_A_RAW', 'HALL_B_RAW', 'HALL_C_RAW', 'GND_SYS'))))
            configuration[instance] = list(FOC_CONFIG_REFS)
            for ref in FOC_CONFIG_REFS:
                jumper = items.get(ref, {})
                hall = f'HALL_{ref[-1]}'
                if (jumper.get('lcsc'), jumper.get('footprint'), jumper.get('pins')) != (
                        None, 'HDR-TH_3P-P2.54-V-M-A', {'1': f'{hall}_DIRECT', '2': hall, '3': f'{hall}_5V_DIV'}):
                    errors.append(f'{instance}.{ref}：保留原 2.54 mm 排针及针序')

        thread, clearance, diameter, current = ('M4', 4.5, 12, 100) if board == 'FOC' else ('M5', 5.5, 14, 150)
        for ref, net in STUD_NETS.get(board, {}).items():
            footprint = f'STUD-{thread}-TH_D{clearance}_PAD{diameter}'
            check(ref, f'power_{thread}', (f'{thread} PCB铜接线柱', None, footprint), {'1': net})
            spec = items.get(ref, {}).get('mechanical_spec', {})
            if (spec.get('thread'), spec.get('clearance_hole_mm'), spec.get('pad_diameter_mm'),
                    spec.get('lug'), spec.get('target_current_a'), spec.get('qualification')) != (
                    thread, clearance, diameter, f'OT-{thread[1:]}', current, 'pending'):
                errors.append(f'{instance}.{ref}：铜柱装配规格或待验证标记不一致')
            if items.get(ref, {}).get('manufacturer') is not None:
                errors.append(f'{instance}.{ref}：待定机械装配须保留供应商品牌为空')
            new_mechanical_specs.add(thread)

        for ref in sorted(checked - before.keys()):
            errors.append(f'{instance}.{ref}：缺少统一前的器件身份')
        old_skus.update(before[ref] for ref in checked & before.keys() if before[ref])

    expected = {'CAN_XH_3P': 5, 'signal_XH_2P': 16, 'power_5p08_2P': 3, 'core_2x12P': 10,
                'cell_XH_10P': 1, 'supercap_XH_9P': 1, 'hall_PH_5P': 2,
                'power_M5': 13, 'power_M4': 10}
    if quantity != expected:
        errors.append(f'全车接口数量不匹配：预期 {expected}，实际 {dict(quantity)}')
    pump_adapter = contracts['水泵']['controller_pin_verification'].get('harness_map_to_bms', {})
    if any(pump_adapter.get(f'pump_J_CAN.{pin}') != f'{signal} -> BMS J_CAN.{pin}'
           for signal, pin in CAN_CABLE_PINS.items()):
        errors.append('水泵↔BMS：线束合同须使用统一直通针序')
    standard = procurement['connector_standard']
    taps = procurement['further_cost_down']['item_3a_bms_tap_connector']
    stock = standard['signal_and_sampling']
    power = standard['medium_power_5p08']
    studs = standard['high_current']
    catalog_parts = [stock['can'], stock['signal_2p'], stock['supercap_9p'], stock['hall_5p'],
                     taps['parts'][0], power['board'], standard['core_sockets']]
    if {(row.get('mpn'), row.get('lcsc')) for row in catalog_parts} != new_catalog_skus:
        errors.append('降本采购：商品与施工合同的板端 SKU 不一致')
    if any(row.get('quantity_vehicle') != quantity[family] for row, family in zip(
            catalog_parts, ('CAN_XH_3P', 'signal_XH_2P', 'supercap_XH_9P', 'hall_PH_5P',
                            'cell_XH_10P', 'power_5p08_2P', 'core_2x12P'))):
        errors.append('降本采购：商品的整车数量与施工合同不一致')
    if stock['can'].get('pins') != {pin: signal for signal, pin in CAN_CABLE_PINS.items()}:
        errors.append('降本采购：CAN 针序须与五板合同一致')
    if (taps.get('board_ref'), taps.get('harness_ref'), taps.get('pins')) != (
            'J_CELL_SERVICE', 'J_CELL_BMS', CELL_TAP_PINS) or [
                (row.get('mpn'), row.get('lcsc'), row.get('quantity_vehicle')) for row in taps['parts'][1:]
            ] != [('XHP-10', 'C144408', 1), ('SXH-001T-P0.6', 'C140573', 10)]:
        errors.append('降本采购：BMS 9S 板端、线端或针序不一致')
    cell_contract = contracts['BMS']['cell_service_cost_down_item_3a']
    cell_cable = cell_contract['wiring']
    housing = cell_contract['candidate'].get('harness_housing', {})
    contacts = cell_contract['candidate'].get('harness_contacts', {})
    if (cell_cable.get('harness_interface'), cell_cable.get('pin_order'), housing.get('lcsc'),
            contacts.get('lcsc'), contacts.get('quantity')) != (
            'J_CELL_BMS', CELL_TAP_PINS, 'C144408', 'C140573', 10):
        errors.append('BMS.J_CELL_BMS：线端配套 C 码、数量或端序错误')
    if (power['mate'].get('lcsc'), power['mate'].get('quantity_vehicle')) != ('C581323', quantity['power_5p08_2P']):
        errors.append('降本采购：5.08 mm 线端 C 码或数量不一致')
    if (studs['m5'].get('quantity_vehicle'), studs['m4'].get('quantity_vehicle')) != (
            quantity['power_M5'], quantity['power_M4']) or any(studs[key].get('lcsc') is not None
                                                               for key in ('m5', 'm4')):
        errors.append('降本采购：M4/M5 结构件数量或 C 码字段不一致')
    maintenance = procurement['offboard']['maintenance']
    if (maintenance.get('female_lcsc'), maintenance.get('male_lcsc')) != ('C19268052', 'C19268051'):
        errors.append('降本采购：AMASS AS150U 维护插头配对 C 码不一致')
    return {
        'status': 'FAIL' if errors else 'PASS', 'findings': errors, 'board_ports': dict(quantity),
        'configuration_jumpers': configuration,
        'sku_scope': '本轮列出的板端接口，排除保留的配置跳线和板外线端。双 FOC 按两板计量，料号去重。无采购型号的裸孔或接口边界不计入旧 SKU。',
        'before_catalog_sku_count': len(old_skus), 'after_catalog_sku_count': len(new_catalog_skus),
        'before_catalog_skus': sorted(old_skus), 'after_catalog_skus': [
            {'mpn': mpn, 'lcsc': code} for mpn, code in sorted(new_catalog_skus)
        ],
        'after_mechanical_specs_without_lcsc': sorted(new_mechanical_specs),
    }


def check_pump_pinmux(contract, drawing):
    expected = {
        'PUMP_PWM': ('J_MCU_A', '10', 'PA8'),
        'PUMP_BRAKE': ('J_MCU_A', '20', 'PB5'),
        'PUMP_I_SENSE': ('J_MCU_B', '11', 'PA7'),
        'ADC_BUS': ('J_MCU_B', '17', 'PA1'),
        'ADC_NTC': ('J_MCU_B', '18', 'PA2'),
    }
    pin_verification = contract['controller_pin_verification']
    plan = pin_verification['signals']
    coreboard = read(POWER.parent / pin_verification['reference_coreboard_pin_audit'])['核心板']
    header_pads = {
        connector: {str(row['脚号']): row['核心板网络'] for row in coreboard[key]}
        for connector, key in [('J_MCU_A', 'P1到J_MCU_A'), ('J_MCU_B', 'P2到J_MCU_B')]
    }
    errors = []
    for signal, (connector, pin, pad) in expected.items():
        item = plan.get(signal, {})
        assigned = drawing.get(connector, {}).get('pins', {}).get(pin)
        if (assigned != signal or item.get('mcu_pad') != pad or item.get('board_connector') != f'{connector}.{pin}'
                or header_pads[connector].get(pin) != pad):
            errors.append(f'{signal}：核心板针位或 MCU PAD 和合同不一致')
    pads = [item.get('mcu_pad') for item in plan.values()]
    if None in pads or len(set(pads)) != len(pads):
        errors.append('STM32G431：存在空缺或重复 MCU PAD')
    pwm = plan.get('PUMP_PWM', {})
    sense = plan.get('PUMP_I_SENSE', {})
    if (pwm.get('peripheral') != 'TIM1_CH1' or pwm.get('alternate_function') != 'AF6'
            or not all(part in sense.get('opamp_path', '') for part in ('OPAMP1', 'ADC1_IN13'))
            or not all(part in sense.get('comparator_path', '') for part in ('COMP2_INP=PA7', 'DAC3_CH2'))
            or 'BKCMP2E' not in sense.get('hardware_trip', '')):
        errors.append('STM32G431：TIM1 PWM 或 COMP2/ADC 内部路由缺失')
    shunt = drawing.get('R_PUMP_SHUNT', {})
    if (shunt.get('mpn') != 'HoJLR2512-2W-20mR-1%-75ppm' or shunt.get('lcsc') != 'C2924538'
            or shunt.get('footprint') != 'R2512'
            or shunt.get('pins') != {'1': 'PUMP_SHUNT', '2': 'GND_SYS'}):
        errors.append('R_PUMP_SHUNT：20 mΩ / 2 W 分流选型或针网错误')
    driver = drawing.get('U_PWM', {})
    if (driver.get('mpn') != 'NSG1416' or driver.get('lcsc') != 'C53436462'
            or driver.get('pins') != {'1': 'NC', '2': '+5V', '3': 'PUMP_PWM',
                                      '4': 'GND_SYS', '5': 'PWM_GATE_DRV'}):
        errors.append('U_PWM：须使用非反相栅驱，保持 PA8 低时关断')
    return {'status': 'FAIL' if errors else 'PASS', 'findings': errors,
            'scope': '核对数字针网和所选 MCU 内部复用，实物核心板方向与保护实测另行确认。'}


def check_foc_cap_bank(contract):
    bank = contract['localization_v1']['offboard_bulk']
    sides = bank.get('harness_per_side', {})
    local_refs = ['CBUS1', 'CBUS2', 'CBUS3']
    errors = []
    if (bank['mpn'] != 'SW31R103MNNAS07S2' or bank['lcsc'] != 'C42400396'
            or bank['per_side'] != 3 or bank['vehicle_quantity'] != 6
            or bank['pins'] != {'1': 'BUS', '2': 'GND_SYS'}):
        errors.append('FOC 母线电容：型号、数量或正负极合同错误')
    if any(sides.get(side, {}).get('references') != local_refs for side in ('FOC-A', 'FOC-B')):
        errors.append('FOC 母线电容：左右线束须分别有 3 只')
    return {'status': 'FAIL' if errors else 'PASS', 'findings': errors,
            'harness': sides}


def check_capture_changes(components, contracts):
    errors = []

    def check(board, ref, expected, part=None, code=None, frozen=None):
        row = components[board].get(ref)
        if row is None and frozen is not None and ref in frozen:
            data = frozen[ref]
            row = {'pins': native_pins(data), 'mpn': data['props'].get('Manufacturer Part')}
        if (not row or any(row['pins'].get(pin) != net for pin, net in expected.items())
                or (part and row.get('mpn') != part) or (code and row.get('lcsc') != code)):
            errors.append(f'{board}.{ref}：料号或目标针网不匹配')

    check('BMS', 'PSU5_EXT', dict(zip(map(str, range(1, 10)),
          'B0 TAP_B9 PSU5_EN B0 PSU5_FB PSU5_VCC PSU5_BST PSU5_SW B0'.split())),
          'SCT2A23ASTER', 'C2901894')
    check('BMS', 'R_PSU5_EN_SW', {'1': 'CTRL_MODULE', '2': 'PSU5_EN'})
    check('BMS', 'C_PSU5_EN', {'1': 'PSU5_EN', '2': 'B0'})
    check('BMS', 'L_PSU5', {'1': 'PSU5_SW', '2': 'VOUT_5V'})
    check('BMS', 'D_TVS', {'1': 'PACK+', '2': 'PACK-'}, 'SMCJ45A', 'C2994055')
    check('BMS', 'C_CAN_ISOIN_1U', {'1': 'CAN_ISO_5V', '2': 'GND_CAN'}, 'CL10A105KB8NNNC', 'C15849')
    bms = contracts['BMS']
    can = bms['can_decoupling']
    if can['visoin'] != ['C_CAN_ISOIN_1U'] or set(can['visoin']) & set(can['visoout']):
        errors.append('BMS.VISOIN：须在 pin10/.12 间使用与 VISOOUT 不同的实体电容')
    holder = bms['localization_v1']['offboard']['tap_fuse_holder']['selected_assembly']
    if (holder.get('rated_dc_v', 0) < 58 or holder.get('configured_channels') != 9
            or holder.get('independent_inputs') is not True or holder.get('common_bus') is not False
            or not holder.get('source')):
        errors.append('BMS：源端九路独立保险装配须有 58 VDC 原厂额定证据')
    bms_original = native_components(POWER.parent / bms['drawing_scope']['baseline'])
    check('BMS', 'RSHUNT', {'1': 'B0', '2': 'PACK-'}, frozen=bms_original)
    check('BMS', 'SW_RUN_STORE', {'4': 'B0', '5': 'CTRL_MODULE'}, frozen=bms_original)
    bypasses = [ref for ref, item in components['BMS'].items()
                if ref != 'RSHUNT' and {'B0', 'PACK-'} <= set(item['pins'].values())]
    if bypasses:
        errors.append(f'BMS：发现施工项直接跨 B0/PACK−：{bypasses}')

    regulator = contracts['稳压']
    reg_original = native_components(POWER.parent / regulator['baseline']['enet'])
    core_path = next(change for change in bms['changes'] if change['id'] == 'CHG-BMS-05')['coreboard_mapping']['source']
    core = read(POWER.parent / core_path)['pin_map']
    af = {
        'P1': {'5': ('PB12', 'DRV_FAULT_N'), '6': ('PB13', 'PWM_L'),
               '10': ('PA8', 'PWM_H'), '11': ('PA9', 'PA9_TIM1_CH2'), '12': ('PA10', 'DRV_FAULT_N'),
               '16': ('PC10', 'PWM_B_L'), '21': ('PB6', 'PWM_B_H'), '22': ('PB7', 'HW_READY_CORE')},
        'P2': {'5': ('PB10', 'MUX_SEL3'), '7': ('PB2', 'ISC_TOTAL_FB'), '9': ('PB0', 'ISC_TOTAL_BIAS'),
               '10': ('PB1', 'IMON'), '11': ('PA7', 'IMON'), '13': ('PA5', 'BUS_ADC'),
               '14': ('PA6', 'DRV_FAULT_N'), '16': ('PA4', 'IMON_B'), '17': ('PA1', 'IMON_A'),
               '18': ('PA2', 'DRV_FAULT_N'), '20': ('PA0', 'IMON')},
    }
    for header, routes in af.items():
        header_gpio = {str(item['pin']): item['mcu_gpio'] for item in core[header]}
        if any(header_gpio.get(pin) != gpio for pin, (gpio, _) in routes.items()):
            errors.append(f'稳压.{header}：核心板 GPIO 分配与官方底稿不一致')
        check('稳压', f'J_CORE_{header}', {pin: net for pin, (_, net) in routes.items()})
    for ref, sw, source, feedback in (('U1', 'SW', 'PHASE_A_LS', 'DRV_SO'),
                                      ('U_DRV_B', 'SW_B', 'PHASE_B_LS', 'DRV_B_SO')):
        check('稳压', ref, {'4': 'DRV_FAULT_N', '7': feedback, '13': 'GND_SYS', '14': source,
                           '15': source, '18': sw, '20': 'BUS_EXT'}, 'DRV8161DGSR', 'C43985204')
    check('稳压', 'R_SC_SHUNT', {'1': 'GND_SYS', '2': 'SC_MINUS'}, frozen=reg_original)
    check('稳压', 'U_SC_GATE_CMD', {'2': 'DRV_FAULT_N', '4': 'SC_ISO_EN'}, frozen=reg_original)
    check('稳压', 'U_SC_ISO_GD', {'1': 'SC_ISO_EN'}, frozen=reg_original)
    check('稳压', 'U_SIGNAL_ISO', {'2': 'BUS_ADC_SOURCE', '3': 'BUS_ADC', '5': 'IMON_A_SOURCE',
                                  '6': 'IMON_A', '8': 'IMON_B', '9': 'IMON_B_SOURCE', '10': '3V3_CORE'},
          'TMUX1511PWR')
    for ref, endpoints in {
        'R_LS_SHUNT_A': ('GND_SYS', 'PHASE_A_LS'), 'R_LS_SHUNT_B': ('GND_SYS', 'PHASE_B_LS'),
        'R_ISC_FEEDBACK': ('IMON', 'ISC_TOTAL_FB'), 'R_ISC_INPUT': ('SC_MINUS', 'ISC_TOTAL_FB'),
        'R_ISC_BIAS_TOP': ('3V3_CORE', 'ISC_TOTAL_BIAS'), 'R_ISC_BIAS_BOT': ('ISC_TOTAL_BIAS', 'GND_SYS'),
    }.items():
        check('稳压', ref, dict(zip('12', endpoints)))
    for ref, anode, cathode in (('D_SC_LO', 'GND_SYS', 'SC_FUSED'),
                              ('D_SC_HI', 'SC_FUSED', 'BUS_EXT'), ('D_DUMP_FW', 'DUMP_SW', 'BUS_EXT')):
        check('稳压', ref, {'1': anode, '2': cathode, '3': anode}, 'MBR30100CT', 'C779296')

    check('FOC', 'DTVS1', {'1': 'BUS', '2': 'GND_SYS'}, 'SMCJ40A', 'C840777')
    foc_original = native_components(POWER.parent / contracts['FOC']['baseline']['candidate_164_enet'])
    check('FOC', 'U1', {}, 'DRV8353RSRGZR', frozen=foc_original)
    for ref, rail in (('U2', 'SAFE_5V'), ('U6', 'DUMP_5V')):
        check('FOC', ref, {'1': rail}, 'TPS7A4001DGNR', frozen=foc_original)
    check('水泵', 'D_BUS', {'1': 'BUS_M', '2': 'GND_SYS'}, 'SMCJ40A', 'C840777')
    check('水泵', 'D_5V', {'1': 'XL_SW', '2': 'GND_SYS'}, 'SS310-SMC', 'C2848691')
    return {'status': 'FAIL' if errors else 'PASS', 'findings': errors}


def check_final_selection(components, contracts, procurement):
    errors = []
    frozen = procurement['final_selection_freeze']
    counts = Counter()
    for instance, board in BOARD_INSTANCES.items():
        by_ref = components[board]
        records = contracts[board].get('passive_freeze', {}).get('refs', {})
        for ref, change in records.items():
            row = by_ref.get(ref, {})
            code = change['selected_lcsc']
            if (code not in FROZEN_PASSIVES or row.get('lcsc') != code
                    or (row.get('mpn'), row.get('footprint')) != FROZEN_PASSIVES[code]
                    or row.get('pins') != change['pins_before']):
                errors.append(f'{instance}.{ref}：阻容料号、封装或针网与选型冻结不一致')
            counts[code] += 1
        if not records:
            errors.append(f'{instance}：缺少本板普通阻容冻结范围')
    if counts != {row['lcsc']: row['quantity_vehicle'] for row in frozen['passives']['catalog']}:
        errors.append('普通阻容采购数量与五板逐位号用量不同')

    reg_contract = contracts['稳压']
    reg_native = native_components(POWER.parent / reg_contract['baseline']['enet'])
    for ref in ('C_BUS_FILM1', 'C_BUS_FILM2', 'C_BUS_FILM3', 'D_BUS_TVS'):
        a = components['稳压'].get(ref, {})
        selection = ('SMCJ43A', 'C2994052') if ref == 'D_BUS_TVS' else ('C312E225K9SC000', 'C5121905')
        if (a.get('mpn'), a.get('lcsc')) != selection or a.get('pins') != native_pins(reg_native[ref]):
            errors.append(f'稳压.{ref}：冻结料号、C 码或针网错误')
        if ref != 'D_BUS_TVS' and '22.5' not in a.get('footprint', '') and '22点5' not in a.get('footprint', ''):
            errors.append(f'稳压.{ref}：法拉薄膜须采用 22.5 mm 脚距')
    aux = reg_native['U_AUX1']
    aux_pins = {'1': 'B0_AUX_EXTERNAL', '2': 'AUX_OR_POS', '3': '12V_PROT',
                '4': 'NC', '5': 'GND_SYS', '6': 'NC'}
    if (aux['props'].get('Manufacturer Part') != 'URB4812YMD-10WR3' or native_pins(aux) != aux_pins
            or any(row['props'].get('Manufacturer Part') == 'URB4805YMD-10WR3' for row in reg_native.values())):
        errors.append('稳压：U_AUX1 隔离针网或旧外置 5 V 清理异常')
    if 'U_AUX1' in components['稳压'] and (
            components['稳压']['U_AUX1']['mpn'], components['稳压']['U_AUX1']['pins']) != (
            'URB4812YMD-10WR3', aux_pins):
        errors.append('稳压：施工增量不得改变保留的 U_AUX1 隔离针网')

    foc_native = native_components(POWER.parent / contracts['FOC']['baseline']['candidate_164_enet'])
    foc = components['FOC']
    npns = [ref for ref, item in foc_native.items() if 'BC847C' in (item['props'].get('Manufacturer Part') or '')
            and item['props'].get('Add into BOM') == 'yes']
    for ref in npns:
        a = foc.get(ref, {})
        if (a.get('mpn'), a.get('lcsc'), a.get('pins'), a.get('pin_roles')) != (
                'MMBT5551', 'C408396', native_pins(foc_native[ref]), dict(zip('123', ('B', 'E', 'C')))):
            errors.append(f'FOC.{ref}：MMBT5551 料号或 B/E/C 针网错误')
    dio = foc.get('U1_BUCK_D', {})
    diode = frozen['foc_buck_diode']
    if (dio.get('mpn'), dio.get('lcsc'), dio.get('pins')) != (
            diode['mpn'], diode['lcsc'], native_pins(foc_native['U1_BUCK_D'])) or diode['vrm_v'] < 100:
        errors.append('FOC.U1_BUCK_D：100 V 续流整流管选型或针网不匹配')
    fuse_rows = [row for row in contracts['FOC']['drawing_components'] if row.get('ref') == 'F_MAIN']
    fuse = fuse_rows[0] if len(fuse_rows) == 1 else {}
    fuse_order = frozen['foc_main_fuse']
    if (fuse.get('mpn'), fuse.get('voltage_rating_vdc'), fuse.get('current_rating_a'),
            fuse.get('interrupting_rating_a'), fuse.get('pins'), fuse.get('placement')) != (
            '0998100.UX-2M8', 70, 100, 2500, native_pins(foc_native['F_MAIN']), 'off-board'):
        errors.append('FOC.F_MAIN：100 A／70 VDC MEGA 保险的料号、分断或板外针网错误')
    if (fuse_order.get('mpn'), fuse_order.get('quantity_vehicle'), fuse_order.get('rated_dc_v'),
            fuse_order.get('interrupting_a')) != ('0998100.UX-2M8', 2, 70, 2500):
        errors.append('双 FOC 保险的采购料号、数量或额定值错误')

    bms_sensors = contracts['BMS']['temperature_sensors']
    bms_pack = bms_sensors['pack_offboard']
    sensor_buy = frozen['temperature_sensors']
    sensor_mpn = 'MF52A103F3950(A1)'
    bms_native = native_components(POWER.parent / contracts['BMS']['drawing_scope']['baseline'])
    if (bms_pack.get('quantity_per_bms'), bms_pack['selected'].get('mpn'),
            bms_pack['selected'].get('supplier_part')) != (3, sensor_mpn, 'C13879'):
        errors.append('BMS 电芯包须采购三颗时恒 10 kΩ / B3950')
    if {row['designator'] for row in bms_pack['refs']} != {'NTC_CELL1', 'NTC_CELL2', 'NTC_CELL3'}:
        errors.append('BMS 电芯包探头位号须完整覆盖 NTC_CELL1–3')
    for record in bms_pack['refs']:
        ref = record['designator']
        signal_pins = {pin: net.split(' / ')[0] for pin, net in record['pin_map'].items()}
        if ref not in bms_native or signal_pins != native_pins(bms_native[ref]):
            errors.append(f'BMS.{ref}：板外探头针网偏离底稿')
    bq = bms_sensors['bq76952']
    if bq.get('selected_pullup_ohm') != 18000 or 'custom' not in bq.get('selected_temperature_model', ''):
        errors.append('BMS：B3950 热敏须有 18 kΩ 上拉的 BQ76952 自定义模型合同')
    reg_sensors = reg_contract['temperature_sensors']
    pump_sensors = contracts['水泵']['temperature_sensors']
    bms_local = bms_sensors['board_onboard']
    if ({a['ref'] for a in reg_sensors} != {'TH_DUMP', 'TH_MAIN', 'TH_SC'} or
            any(a['mpn'] != sensor_mpn or a['lcsc'] != 'C13879' for a in reg_sensors + pump_sensors)
            or {a['ref'] for a in bms_local} != {'NTC_BOARD', 'NTC_MOS'}
            or any(a['mpn'] != sensor_mpn or a['supplier_part'] != 'C13879' for a in bms_local)
            or [(a['ref'], a['pins']) for a in pump_sensors] != [('NTC_BOARD', {'1': 'ADC_NTC', '2': 'GND_SYS'})]
            or bms_pack['quantity_per_bms'] + len(bms_local) + len(reg_sensors) + len(pump_sensors) != 9
            or (sensor_buy['mpn'], sensor_buy['lcsc'], sensor_buy['quantity_vehicle']) != (sensor_mpn, 'C13879', 9)):
        errors.append('整车九颗已列料的测温 NTC 或稳压三处探头选型错误')

    old = {row['mpn'] for row in frozen['stop_purchasing']}
    if not {'R76IR4220SE40J', 'URB4805YMD-10WR3', 'DT04-12PA', 'DT06-12SA-P004',
            'Eaton CBBF-100', '103AT-2', 'DFLS1100-7'} <= old:
        errors.append('停购清单缺少本轮指定的退役型号')
    stopped_mpns = {row['mpn'] for row in frozen['stop_purchasing'] if not row.get('lcsc')}
    stopped_codes = {row['lcsc'] for row in frozen['stop_purchasing'] if row.get('lcsc')} | {'C2846678', 'C22548'}
    for row in procurement['p0_p1_capture_procurement']['sourcing']:
        if row['mpn'] in stopped_mpns or row.get('lcsc') in stopped_codes:
            errors.append(f"当前采购行仍引用退役料 {row['mpn']}")
    return {'status': 'FAIL' if errors else 'PASS', 'findings': errors,
            'passive_quantity_by_code': dict(counts), 'foc_npn_quantity': 2 * len(npns)}


def check_harness_short_circuit_protection(components, contracts, procurement):
    errors = []
    summary = procurement['harness_short_circuit_protection']
    reg, foc = contracts['稳压'], contracts['FOC']
    supercap = reg['supercap_tap_fuses']
    fuse = supercap['part']
    reg_rows = components['稳压']
    raw_nodes = reg_rows.get('J_SC_TAPS', {}).get('pins', {})
    fuse_refs = {f'F_SC_TAP{i}' for i in range(1, 9)}
    branches = {row['ref']: row for row in supercap['fuses']}
    expected = {
        f'F_SC_TAP{i}': {
            'ref': f'F_SC_TAP{i}', 'source': f'SC{i}.1', 'source_connector_pin': str(i + 1),
            'pins': {'1': SC_TAP_PINS[str(i + 1)], '2': f'SC_TAP{i}_PROT'},
        }
        for i in range(1, 9)
    }
    if (len(supercap['fuses']) != 8 or branches != expected or raw_nodes != SC_TAP_PINS
            or supercap.get('placement') != 'off-board' or supercap.get('unfused_return') != 'SC_MINUS'
            or supercap.get('bypass_allowed') is not False or fuse_refs & reg_rows.keys()):
        errors.append('稳压超容抽头：只允许八个正极支路源端串熔，板内针序、负端直通与分域须一致')
    expected_far_end = {'1': 'SC_MINUS', **{str(i + 1): f'SC_TAP{i}_PROT' for i in range(1, 9)}}
    if supercap.get('harness_far_end') != expected_far_end or reg['target_component_count'] != 268:
        errors.append('稳压超容抽头：受保护线束远端针序或板内对象数错误')
    reg_native = native_components(POWER.parent / reg['baseline']['enet'])
    reg_pins = {ref: native_pins(row) for ref, row in reg_native.items()}
    reg_pins.update({ref: row['pins'] for ref, row in reg_rows.items()})
    if any(reg_pins.get(f'SC{i}', {}).get('1') != SC_TAP_PINS[str(i + 1)] for i in range(1, 9)):
        errors.append('稳压超容抽头：源端与冻结的八节超容正极不一致')
    protected_taps = set(expected_far_end.values()) - {'SC_MINUS'}
    if any(protected_taps.intersection(pin_map.values()) for pin_map in reg_pins.values()):
        errors.append('稳压超容抽头：受保护端网名只允许在线束侧出现')
    fuse_purchase = summary['supercap_fuses']
    if (not 2 <= fuse['current_a'] <= 3 or fuse['rated_dc_v'] < 32
            or fuse['dc_interrupt_a'] <= 0 or not fuse['source']
            or any(fuse_purchase.get(key) != fuse[key] for key in
                   ('mpn', 'lcsc', 'current_a', 'rated_dc_v', 'dc_interrupt_a'))
            or fuse_purchase['quantity_vehicle'] != 8):
        errors.append('稳压超容抽头：2–3 A、直流分断额定或采购项错误')
    budget = summary['budget']
    if (budget.get('limit_cny_exclusive') != 15 or budget.get('status') not in ('HOLD', 'PASS')
            or (budget['status'] == 'HOLD' and fuse_purchase.get('order_release') is not False)):
        errors.append('稳压线束采购：须明确 ¥15 上限和预算未闭合时的暂停状态')

    base = native_components(POWER.parent / foc['baseline']['candidate_164_enet'])
    foc_rows = components['FOC']
    ptc = foc_rows.get('F_HALL_5V', {})
    hall = foc['hall_supply_protection']
    if (len(base) != 164 or foc['target_component_count'] != 165
            or foc['delta_vs_164_candidate']['added'] != 1
            or foc['delta_vs_164_candidate']['removed'] != 0
            or set(foc_rows.keys() - base.keys()) != {'F_HALL_5V'}):
        errors.append('FOC 每板只新增 F_HALL_5V，施工对象数必须为 165')
    pins = {ref: native_pins(row) for ref, row in base.items()}
    pins.update({ref: row['pins'] for ref, row in foc_rows.items()})
    uses = {(ref, pin) for ref, pin_map in pins.items()
            for pin, net in pin_map.items() if net == HALL_SUPPLY_NET}
    old_hall = native_pins(base['J_HALL'])
    if (old_hall.get('1') != '5V_BUCK'
            or pins.get('J_HALL') != {**old_hall, '1': HALL_SUPPLY_NET}
            or foc['harness_pin_map'].get('J_HALL') != pins['J_HALL']
            or ptc.get('pins') != {'1': '5V_BUCK', '2': HALL_SUPPLY_NET}
            or uses != {('F_HALL_5V', '2'), ('J_HALL', '1')}):
        errors.append('FOC 霍尔：J_HALL.1 须只经 PTC 取电，三路信号和回流须保留原针网')
    selected = summary['hall_ptc']
    if (tuple(ptc.get(k) for k in ('mpn', 'lcsc', 'manufacturer', 'footprint')) !=
            tuple(selected.get(k) for k in ('mpn', 'lcsc', 'manufacturer', 'footprint'))
            or ptc.get('footprint') not in ('F0603', 'F0805')
            or hall['ref'] != 'F_HALL_5V' or hall['quantity_per_board'] != 1
            or selected['quantity_vehicle'] != 2 or hall['hold_current_a_25c'] not in (0.10, 0.15)
            or not hall['hold_current_a_25c'] < hall['trip_current_a_25c'] <= 0.30
            or hall['hold_current_a_at_60c'] < hall['hall_load_budget_a'] or hall['rated_dc_v'] < 6):
        errors.append('FOC 霍尔：PTC 规格或采购对照错误')

    return {
        'status': 'FAIL' if errors else 'PASS',
        'findings': errors,
        'supercap_offboard_quantity': len(branches),
        'hall_ptc_quantity_vehicle': selected['quantity_vehicle'],
        'foc_target_schematic_count': foc['target_component_count'],
        'physical_qualification': summary['physical_qualification'],
        'budget_status': budget['status'],
    }


def check_simulation(contract_hashes):
    simulation = read(HERE / '降本后典型故障联合仿真.json')
    conclusions = simulation['conclusions']
    expected = {str(number): 'PASS' for number in range(1, 7)}
    errors = []
    if simulation.get('contract_sha256') != contract_hashes:
        errors.append('仿真报告未绑定当前四份合同')
    if simulation.get('script_sha256') != digest(HERE / '降本后典型故障联合仿真.py'):
        errors.append('仿真报告未绑定当前仿真源码')
    if conclusions != expected:
        errors.append('6 个场景必须全部 PASS')
    return {'status': 'FAIL' if errors else 'PASS', 'scenarios': conclusions,
            'pass_count': sum(outcome == 'PASS' for outcome in conclusions.values()),
            'findings': errors}


def main():
    components = {}
    boards = {}
    hashes = {}
    contracts = {}
    for board, rel_path in CONTRACTS.items():
        path = POWER / rel_path
        contracts[board] = read(path)
        components[board], boards[board] = check_components(board, contracts[board])
        hashes[rel_path] = digest(path)
    pin_roles = check_pin_roles(components)
    pin_roles['findings'].extend(check_native_mos_mapping(components, contracts['稳压'], contracts['FOC']))
    pin_roles['status'] = 'FAIL' if pin_roles['findings'] else 'PASS'
    harness = check_harness(components, contracts['BMS'])
    procurement = read(HERE / '降本采购与SMT建议.json')
    connectors = check_connector_standard(components, contracts, procurement)
    final_selection = check_final_selection(components, contracts, procurement)
    short_protection = check_harness_short_circuit_protection(components, contracts, procurement)
    pump_pinmux = check_pump_pinmux(contracts['水泵'], components['水泵'])
    foc_cap_bank = check_foc_cap_bank(contracts['FOC'])
    simulation = check_simulation(hashes)
    capture = check_capture_changes(components, contracts)
    passed = all(section['status'] == 'PASS'
                 for section in [*boards.values(), pin_roles, harness, connectors, final_selection, short_protection,
                                 pump_pinmux, foc_cap_bank,
                                 simulation, capture])
    report = {'status': 'PASS' if passed else 'FAIL', 'contract_sha256': hashes,
              'boards': boards, 'critical_pin_roles': pin_roles, 'harness': harness,
              'connector_standard': connectors, 'final_selection_freeze': final_selection,
              'harness_short_circuit_protection': short_protection, 'pump_pinmux': pump_pinmux,
              'foc_cap_bank': foc_cap_bank, 'simulation_gate': simulation, 'remaining_capture_constraints': capture}
    (HERE / '画图前合同完整性校验.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for board, section in [*boards.items(), ('关键针序', pin_roles), ('跨板接口', harness),
                           ('全车接插件', connectors), ('选型冻结', final_selection), ('线束短路施工合同', short_protection),
                           ('水泵针位', pump_pinmux),
                           ('FOC 电容线束', foc_cap_bank), ('联合仿真', simulation), ('待闭合条件', capture)]:
        print(f"{board} {section['status']}")
        for finding in section['findings']:
            print(f'  {finding}')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
