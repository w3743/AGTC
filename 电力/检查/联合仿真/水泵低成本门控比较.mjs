// Sensitivity comparison; no candidate is qualified by this model.
// Existing XL7015 EN gate and branch isolation are NOT implemented.
const R = 8.726, Cdirect = .079783824, pumpCapHigh = .000603744;
const Cisolated = Cdirect - pumpCapHigh + .00000012;
const dt = .0001, tMax = 10, threshold = 18;
const cases = [
  {name:'原直挂_两辅助加载', cap:Cdirect, pumpW:1.875, pumpOffA:0},
  {name:'仅禁泵5V_电容仍直挂', cap:Cdirect, pumpW:0, pumpOffA:.0003738},
  {name:'泵全支路隔离', cap:Cisolated, pumpW:0, pumpOffA:.00006},
];
function run(pack, c) {
  let v=0, firstStart=null, hit=null, atWindow=null;
  for(let i=0;i<=Math.round(tMax/dt);i++) {
    const t=i*dt;
    if(firstStart===null && v>=threshold) firstStart=t;
    if(hit===null && pack-v<=.5) hit=t;
    if(i===27380) atWindow=pack-v;
    const powered = firstStart!==null; // idealized once-started behavior, not actual UVLO
    const load=.002+c.pumpOffA+(powered ? (.75+c.pumpW)/Math.max(v,.001) : 0);
    v+=dt*((pack-v)/R-load)/c.cap;
  }
  return {name:c.name,packV:pack,capacitanceF:c.cap,firstStartS:firstStart,
    deltaAt2p738S:atWindow,timeToPoint5V:hit,deltaAt10S:pack-v};
}
console.log(JSON.stringify({status:'COST_AND_POWER_SENSITIVITY_ONLY_HOLD',
 assumptions:{R,dt,threshold,Vstart:'18 V is a model assumption, not a guaranteed startup threshold',
 regulatorInputW:.75,pumpOnInputW:1.875,backgroundOtherA:.002,
 pumpENoffA:.0003738,fullGateOffA:.00006,
 offCurrentWarning:'373.8uA is a constant sensitivity sum: XL7015 VIN shutdown current 200uA + pump divider 164.8uA + DRV 8uA + TVS 1uA at different specified conditions. Divider current in reality varies with voltage. The independent 2mA background excludes these items. 60uA is a specified-condition switch input-current screen. Neither is a full-slope/full-temperature guarantee; both exclude gate/receiver/additional leakage and reverse current.',
 omitted:'Internal input capacitance, supply startup transients, actual shutdown thresholds, extra bias and harness effects, dynamic/thermal behavior, reverse injection, and 0.5V closing qualification.'},
 results:[22.5,37.8].flatMap(v=>cases.map(c=>run(v,c)))},null,2));
