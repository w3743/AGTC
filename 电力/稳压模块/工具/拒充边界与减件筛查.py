"""Electrical necessary conditions, not hardware qualification."""
import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parents[1]
source=root/'检查/本轮磁盘读回.enet'
r0=.82; tol=.05; vmin=22.5; pflat=495.3
rmin=r0*(1-tol); rmax=r0*(1+tol)
foc_earliest=39.209
threshold_high=.404; threshold_low=.387
# Even granting zero resistor tolerance/input bias/ground difference, independent
# TI published threshold extremes are required for guaranteed ordering.
fall_max_for_order=foc_earliest*threshold_low/threshold_high
voh=2.6; vb=1.0
ibase=(voh-vb)/10100-vb/9900
data={
 'source_netlist_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'guarantee_scope':'conditional arithmetic only; no thermal, failure, end-to-end timing, or actual motor guarantee',
 'ideal_dump_22p5V':{'r_min_ohm':rmin,'r_max_ohm':rmax,
   'min_full_on_W_no_extra_impedance':vmin*vmin/rmax,
   'flat_example_initial_W':pflat,'initial_margin_W':vmin*vmin/rmax-pflat,
   'total_series_ohm_max_for_495p3W':vmin*vmin/pflat,
   'series_ohm_allowance_after_initial_r_max':vmin*vmin/pflat-rmax,
   'incl_temperature_example':{'TCR_per_C_assumed':.0002,'delta_C_assumed':225,
      'R_hot_ohm':rmax*(1+.0002*225),'series_ohm_allowance':vmin*vmin/pflat-rmax*(1+.0002*225)},
   'slope_10deg_initial_W_example':708.1633626896754,
   'slope_10deg_shortfall_W_no_other_sinks':708.1633626896754-vmin*vmin/rmax},
 'comparator_static':{'foc_earliest_paper_V':foc_earliest,'ti_tlv6700_vit_positive_max_V':threshold_high,
  'ti_tlv6700_vit_negative_min_V':threshold_low,
  'max_guaranteed_release_if_earliest_foc_precedes_V':fall_max_for_order,
  'bus_full_V_comparison':37.8,'conditional_conclusion':'with independent published extrema and an ideal single divider, cannot guarantee both central rise before 39.209 V and release above 37.8 V; does not include dynamic delay, R tolerances or offsets'},
 'candidate_remove_R_DUMP_IN_SER':{'PB7_lowest_high_V':voh,'assumed_Vbe_V':vb,'base_series_max_ohm':10100,
  'base_pulldown_min_ohm':9900,'derived_ib_A':ibase,
  'status':'NOT_QUALIFIED: MDD Vce(sat) 0.15 V and Vbe 1 V at Ic=10mA, Ib=1mA, 25 C do not establish Vce at this base current; base series short changes MCU fault-current limiting',
  'component_change_if_qualified':-1,'existing_mpn_change_if_qualified':0},
 'scenarios':{'battery_refuses':'P_batt=0; P_sc + P_dump must cover regen minus actual loads and transient storage',
 'supercap_refuses':'P_sc=0; share fresh BMS net CCL once, then P_dump covers deficit',
 'both_refuse':'P_batt=P_sc=0; P_regen-P_load <= qualified P_dump plus time-limited storage',
 '12V_absent':'no guaranteed central dump or SC converter drive; FOC BOV requested stop and remaining bus paths only',
 'one_foc_feeder_open':'evaluate each electrical island; disconnected FOC cannot use remote dump'},
 'sources':['https://www.ti.com/lit/ds/symlink/tlv6700.pdf',
  'https://www.st.com/resource/en/datasheet/stm32g431cb.pdf',
  'https://www.mouser.com/pdfDocs/ENG_DS_1773309-8_B.pdf']
}
out=root/'检查/拒充边界与减件筛查.json';out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(out)
