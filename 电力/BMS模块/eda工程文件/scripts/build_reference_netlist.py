"""Rebuild the BMS schematic from the immutable ENET and drawing contract."""

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(PROJECT.parents[1] / '检查/联合仿真/code_as_eda'))

from reference import apply_drawing_rows, load_baseline, read_json, require, write_reference

CONTRACT = PROJECT.parent / '检查/闭合候选_9S5P/BMS_9S5P施工合同.json'


def build():
    contract = read_json(CONTRACT)
    baseline_path = ROOT / contract['drawing_scope']['baseline']
    parts = load_baseline(baseline_path)
    require(len(parts) == 279, 'The precharge ENET must have 279 schematic objects')

    removed = contract['drawing_scope']['removed_from_frozen_enet']
    require(removed == [f'R_PSU_MINLOAD{i}' for i in range(1, 6)], 'Unexpected removals')
    for ref in removed:
        del parts[ref]

    # R_PRE is the off-board resistor in the older ENET.  The final drawing
    # explicitly transfers its board boundary to J_R_PRE via replaces_ref.
    # The physical resistor remains in the separate off-board procurement.
    rows = contract['drawing_components']
    replacement = next(row for row in rows if row['ref'] == 'J_R_PRE')
    require(replacement.get('replaces_ref') == 'R_PRE', 'R_PRE boundary provenance changed')
    old_pins = parts['R_PRE']['pins']
    require(old_pins == replacement['pins'], 'R_PRE and J_R_PRE terminals differ')
    del parts['R_PRE']
    apply_drawing_rows(parts, rows)

    # The published ENET marked the old service port as an unpopulated
    # schematic placeholder.  It is now the populated ten-pin header.
    parts['J_CELL_SERVICE']['placement'] = 'board'
    parts['J_CELL_SERVICE']['in_bom'] = True

    # The temperature and tap-fuse replacements are off-board items.  Their
    # connection nets come from the original schematic or contract pin map.
    sensor = contract['temperature_sensors']['pack_offboard']['selected']
    for ref in ('NTC_CELL1', 'NTC_CELL2', 'NTC_CELL3'):
        parts[ref].update(mpn=sensor['mpn'], lcsc=sensor['supplier_part'],
                          manufacturer=sensor['manufacturer'], value='10k NTC B3950 off-board')

    tap = contract['localization_v1']['tap_fuse']
    for ref, pins in tap['refs_and_nets'].items():
        parts[ref].update(mpn=tap['mpn'], manufacturer=tap['manufacturer'],
                          lcsc=tap.get('lcsc'), pins=pins, footprint='OFFBOARD-MINI-BLADE',
                          value='5A 58VDC MINI off-board')
    main_fuse = contract['localization_v1']['offboard']['f1']
    parts['F1'].update(mpn=main_fuse['mpn'], manufacturer=main_fuse['manufacturer'],
                       lcsc=main_fuse.get('lcsc'), footprint='OFFBOARD-ANL',
                       value='150A 58VDC ANL off-board')
    # CELL_B9 and CELL_B9_RAW name the same battery-positive source.  The
    # former is used by the baseline F1, the latter by the final tap contract.
    require(parts['F1']['pins']['1'] == 'CELL_B9', 'Unexpected main-fuse source net')
    parts['F1']['pins']['1'] = 'CELL_B9_RAW'

    require(len(parts) == 301, f'Unexpected schematic object count: {len(parts)}')
    require(parts['J_CAN']['pins'] == {'1': 'GND_CAN', '2': 'CANH', '3': 'CANL'},
            'CAN pinout differs from final contract')
    require(parts['J_CORE_P2']['pins']['12'] == 'EARLY_GUARD_READY_N', 'PC4 mapping differs')
    write_reference(PROJECT, 301, CONTRACT, baseline_path, parts, notes=[
        '301 schematic objects include the documented external switch, fuse and probe symbols.',
        'J_R_PRE replaces the old off-board R_PRE symbol at the PCB boundary; the resistor is an external purchase.',
        'F_CELL_B1..B9, F1 and NTC_CELL1..3 use current off-board contract selections.',
        'CELL_B9 on the old F1 and CELL_B9_RAW on F_CELL_B9 are the same source node; use CELL_B9_RAW consistently.',
    ])


if __name__ == '__main__':
    build()
