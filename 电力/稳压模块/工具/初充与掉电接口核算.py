"""原理阶段的电流包络、充电时间与掉电接口核算；不是开关级 SPICE 或固件。"""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def cycle(v, bus, inductance, resistance, diode, low=15.0, high=25.0):
    """固定瞬时电容开路电压的 RL 充电/续流周期，解析积分。"""
    a = (bus - v) / resistance
    if a <= high:
        raise ValueError("无法达到电流上沿")
    ton = inductance / resistance * math.log((a-low)/(a-high))
    ion = a*ton - inductance/resistance*(high-low)
    i2on = a*a*ton - 2*a*inductance/resistance*(high-low)
    i2on += inductance/(2*resistance)*((a-low)**2-(a-high)**2)
    b = (v+diode)/resistance
    toff = inductance/resistance*math.log((high+b)/(low+b))
    ioff = inductance/resistance*(high-low)-b*toff
    i2off = inductance/(2*resistance)*((high+b)**2-(low+b)**2)
    i2off -= 2*b*inductance/resistance*(high-low)
    i2off += b*b*toff
    period = ton+toff
    return dict(ton_s=ton, toff_s=toff, frequency_Hz=1/period,
                average_A=(ion+ioff)/period, rms_A=math.sqrt((i2on+i2off)/period),
                diode_loss_W=diode*ioff/period)


def profile(cell_c, cell_esr, bus, inductance):
    # 低压段不驱动同步低管，利用体二极管阻止负电流；2V 后允许同步续流。
    # R 的 12mΩ 是 MOS/隔离/电感/分流/连接工程预算，不冒充全温保证。
    resistance = 8*cell_esr+0.012
    total_s = cap_heat = diode_heat = 0.0
    samples = []
    dv = 0.02
    for i in range(880):
        v = (i+0.5)*dv
        result = cycle(v,bus,inductance,resistance,1.3 if v<2 else 0)
        dt = (cell_c/8)*dv/result["average_A"]
        total_s += dt
        cap_heat += result["rms_A"]**2*cell_esr*dt
        diode_heat += result["diode_loss_W"]*dt
        if i in [0,49,99,100,199,399,699,879]:
            samples.append(dict(cap_OCV_V=v, **result))
    return dict(cell_F=cell_c,cell_ESR_ohm=cell_esr,bus_V=bus,L_H=inductance,
                time_to_17p6_s=total_s,per_cell_ESR_heat_J=cap_heat,
                low_side_diode_heat_J=diode_heat,samples=samples)


def main():
    cases = [profile(c,r,b,l) for c,r in [(280,.0056),(350,.0028),(420,.0028)]
             for b in [22.5,37.8,42.0] for l in [12.8e-6,24e-6]]
    assert all(30 < x["time_to_17p6_s"] < 48 for x in cases)
    # 四个主机侧节点以旧 10k 下拉吸收开关的全温失电漏电。
    off_v = 2e-6*10000*1.001
    assert off_v < .3
    ron = 4.5
    low_parallel = 1/(1/10000+1/(10000+ron))
    source_v_per_bus = low_parallel/(300000+low_parallel)
    mcu_v_per_bus = source_v_per_bus*10000/(10000+ron)
    bus_scale = 1/mcu_v_per_bus
    bootstrap = dict(low_pulses=150,input_low_pulse_us=2,period_us=100,
                     sequence_ms=15,bootstrap_nominal_uF=1,
                     required_effective_capacitance_uF_min=.5,
                     gate_charge_budget_nC=500,gate_charge_is_design_budget_not_MPN_max=True,
                     required_initial_bootstrap_V_min=9.7,
                     estimated_after_gate_pulse_V_min=9.7-.5/.5,
                     cap_voltage_before_arm_V_max=19.6,
                     conservative_negative_pulse_A=19.6/12.8e-6*2.08e-6,
                     reverse_current_decay_us_at_22p5V=19.6/(22.5+1.3-19.6)*2.08)
    adc = dict(new_refs=["U_SIGNAL_ISO","C_SIGNAL_ISO"],
               reused_refs=["R_T1_LO","R_T2_LO","R_T3_LO","R_T4_LO"],
               mpn="TMUX1511PWR",supplier_part="C2866750",
               supply="3V3_CORE",select="all four SEL=WD_OK; not the slow AFE scan heartbeats",
               maximum_off_leakage_uA=2,maximum_MCU_side_DC_V_with_10k=off_v,
               maximum_Ron_ohm=4.5,bus_scale_nominal_with_Ron0=61,
               bus_scale_with_Ron4p5=bus_scale,
               bus_ADC_step_V_at_3p3V_12bit=3.3/4096*61,
               procurement_CNY=None,
               normal_voltage_bounds="all S/D 0..3.6V even when VDD<1.5V; not a bus-voltage switch")
    bom = ROOT/"原理图准备/BOM.csv"
    result = dict(date="2026-10-29",status="ANALYTICAL_DESIGN_ENVELOPE_NOT_SPICE_OR_PHYSICAL_PASS",
                  source_bom_sha256=hashlib.sha256(bom.read_bytes()).hexdigest(),
                  current_band_A=[15,25],profiles=cases,
                  calculated_time_range_s=[min(x["time_to_17p6_s"] for x in cases),max(x["time_to_17p6_s"] for x in cases)],
                  bootstrap=bootstrap,power_off_interfaces=adc,
                  scan_tail_at_50A=dict(age_s=.1454,cell_C_min=280,
                     rise_V=50*.1454/280,latest_OV_V=2.4512+.020+.00244+50*.1454/280),
                  assumptions=["当前源未修改；本模型不包含布局寄生、闭环补偿或未保证的电感偏置最小值。",
                   "终点是均衡串OCV17.6V；单体端电压限制可能提前停充/降流。若要求弛豫后每节2.2V，还需尾段时间，46.28s不是最大充满时间。",
                   "15/25A 为实际电流模型带宽；要求后续校准与延迟预算，不把 DRV 标称增益当零误差。",
                   "低压异步段按1.3V体二极管压降作热预算；中高压采用同步续流，须在正电流内换态。",
                   "2us 为输入候选下限；不得在 Break 保护上设置自动恢复以实现逐周期控制。",
                   "50A 是正常充电/采样失效尾窗电流预算，硬短路另行快速隔离。",
                   "60°C环境仍须按单体65°C上限降流；本模型时间不承诺全温最大秒数。"],
                  checks=dict(profile_time_envelope="PASS",off_DC_voltage="PASS",native_schematic="NOT_CHANGED",physical_test="NOT_PERFORMED"))
    out=ROOT/"检查/初充与掉电接口核算.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(output=str(out),time_range_s=result["calculated_time_range_s"],off_pin_mV=off_v*1000,
                         bus_scale=bus_scale,OV_latest_V=result["scan_tail_at_50A"]["latest_OV_V"]),ensure_ascii=False))


if __name__ == "__main__":
    main()
