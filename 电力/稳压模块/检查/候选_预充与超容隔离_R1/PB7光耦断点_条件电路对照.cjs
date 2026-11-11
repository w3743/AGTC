// Piecewise nodal DC comparisons with explicit, non-guaranteed device assumptions.
const fs=require('fs'),path=require('path'),assert=require('assert');
const v12min=11.4,v12max=12.6,rmin=9900,rmax=10100;
const cases=[
 {name:'LPV off, software transistor forced on; opto dark at 25 C datasheet point',off_v:v12min-rmax*100e-9,hardware_ov:false,assumption:'100 nA max only IF=0, VCE=20 V, 25 C; no full-temperature or actual-bias guarantee'},
 {name:'LPV off, hardware overvoltage active',off_v:.2,hardware_ov:true,assumption:'hard-path transistor VCE=0.2 V assumed; opto not in this path; actual UCC action also requires valid early rail and GUARD_OK high'},
 {name:'LPV on, PB7 high, both series transistors assumed saturated',off_v:.4,collector_ma:(v12max-.4)/rmin*1000,assumption:'0.2 V each assumed, not a guaranteed operating point; needed current exceeds 1 mA optocoupler VCEsat test'},
 {name:'negative control: LPV cutoff transistor bypassed, PB7 high',off_v:.2,hardware_ov:false,assumption:'Q_DUMP_SW VCE=0.2 V assumed; represents original fault topology'}
];
const led=(4.75-1.65-1.4-.1)/515.1*1000;
const result={description:'Condition-based DC KCL: non-linear optocoupler and gate-driver timing not simulated.',cases,limits:{current_at_1V_mA:(v12max-1)/rmin*1000,ucc_input_high_v:2.4,ucc_input_low_v:1,serial_led_if_mA:led,serial_led_assumptions:'hypothetical internal LED series: 4.75V, 515.1Ω, Vf 1.65V + 1.4V, MOS 0.1V; Vf values specified at different larger test currents, not guaranteed at actual operating current',bms_vce_sat_test_if_mA:5,bms_vce_sat_test_ic_mA:1,steady_optocoupler_vce_sat_test_if_mA:20,steady_optocoupler_vce_sat_test_ic_mA:1},warnings:['BMS LED path is not exposed; insertion requires BMS redesign; adding load on shared FOC output also violates unverified interface.', 'No full-temperature/low-VCE CTR, leakage, fault timing or LED current guarantee; current comparisons are not qualification.', 'LPV high with local 5V lost remains unblocked; no complete solution.', 'Independent early hard rail and comparator validity, plus retention through regenerative tail, are assumptions.']};
assert(cases[0].off_v>2.4);assert(cases[1].off_v<1);assert(cases[2].collector_ma>1);assert(led<5);fs.writeFileSync(path.join(__dirname,'PB7光耦断点_条件电路对照.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));

