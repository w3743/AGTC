// R1 software dump gating: bounded, abstract checks, not a transistor/SPICE qualification.
const fs=require('fs');
const path=require('path');
const assert=require('assert');
const rows=[];
for(const guard of [0,1]) for(const core of [0,1]) for(const formal5 of [0,1]) for(const lpv of [0,1]) for(const pb7 of [0,1]) for(const hw of [0,1]) {
 const sw=guard&&core&&formal5&&lpv&&pb7?1:0;
 rows.push({guard,core,formal5,lpv,pb7,hw,sw,node_request:(sw||hw)?1:0});
}
const rhi=4700*1.01, rlo=22000*0.99, vcore=3, vce=.2;
const vqual=(vcore-vce)*rlo/(rhi+rlo);
// Example of an unspecified intermediate supply output, deliberately not a prediction.
const vy=1.5, vbe=.55, rb=10100, rpd=9900;
const ib=Math.max(0,(vy-vbe)/rb-vbe/rpd), ic=ib*35;
const result={type:'proposed full-permit ideal logic and explicit assumed counterexample; not transient simulation',inputs:'proposed full qualification contract; resistor example refers to R1 latest official netlist',checks:{logic_rows:rows.length,software_blocked_without_any_permit:rows.every(x=>x.sw===0||x.guard&&x.core&&x.formal5&&x.lpv&&x.pb7),hard_request_node_independent_of_lpv:rows.every(x=>!x.hw||x.node_request===1),optocoupler_high_ideal_conservative_resistors_v:vqual,logic_high_threshold_v:2,low_supply_example:{supply_v:1.5,assumed_undefined_and_output_v:vy,assumed_vbe_v:vbe,assumed_gain:35,base_current_ua:ib*1e6,collector_current_ma:ic*1e3,required_collector_current_ma:(12.6-1)/9900*1e3,interpretation:'Plausibility calculation only; output and BJT behavior not guaranteed; no claim actual failure.'}},limits:['The 64 cases evaluate only the desired DUMP_OFF request expression, not actual netlist or power flow; R1 UCC IN+=GUARD_OK and a low GUARD blocks actuation.', 'Guard 0 to valid, supervisor POR, reset assertion, analog switch power-up and partial rails are unmodeled.', 'Optocoupler VCE=0.2 V is an assumed operating point, not a CTR guarantee.', 'No BJT temperature/saturation, gate charge, wiring, or measured waveforms.']};
assert.equal(rows.length,64);
assert.ok(result.checks.software_blocked_without_any_permit);
assert.ok(result.checks.hard_request_node_independent_of_lpv);
assert.ok(Number.isFinite(vqual) && Number.isFinite(ib) && Number.isFinite(ic));
fs.writeFileSync(path.join(__dirname,'PB7许可边界复算.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));




