// Conditional comparison only: does not qualify the physical precharge circuit.
// 0.5 V is a comparison point, not an approved main-switch closing threshold.
const packV = 37.8;
const resistanceOhm = 8.726;
const windowS = 2.738; // Longest edge of the legacy DRVn high-window estimate.
const targetDeltaV = 0.5;
const capFullF = 0.079783824;
// Remove the pump branch's 503.12 uF with 20% tolerance; add the
// proposed upstream 0.1 uF bypass at +20%. The gate is not implemented.
const pumpCapHighF = 0.000603744;
const gateBypassHighF = 0.00000012;
const cases = [
  {name: '现行直挂_空载', capacitanceF: capFullF, loadA: 0},
  {name: '候选泵门控_空载', capacitanceF: capFullF - pumpCapHighF + gateBypassHighF, loadA: 0},
  {name: '候选泵门控_25mA假设', capacitanceF: capFullF - pumpCapHighF + gateBypassHighF, loadA: 0.025},
  // Historical system sensitivity, not a deduplicated post-gating load budget.
  {name: '候选泵门控_118p3mA假设', capacitanceF: capFullF - pumpCapHighF + gateBypassHighF, loadA: 0.1183},
];
function calculate({name, capacitanceF, loadA}) {
  const residualV = resistanceOhm * loadA;
  const tauS = resistanceOhm * capacitanceF;
  const deltaAtWindowV = residualV + (packV - residualV) * Math.exp(-windowS / tauS);
  const tTargetS = targetDeltaV > residualV
    ? tauS * Math.log((packV - residualV) / (targetDeltaV - residualV)) : null;
  return {name, capacitanceF, loadA, residualV, deltaAtWindowV, tTargetS,
    reachesComparisonPointInWindow: tTargetS !== null && tTargetS <= windowS};
}
console.log(JSON.stringify({qualification: 'CONDITIONAL_MODEL_NOT_HARDWARE_ACCEPTANCE',
  assumptions: {packV, resistanceOhm, windowS, targetDeltaV, pumpCapHighF, gateBypassHighF,
    constraints: 'Ideal constant voltage source and capacitance; constant load; 0 V initial BUS; no wiring, dynamic loads or temperature drift. 118.3mA is a historical sensitivity, not a post-gating verified budget.'},
  cases: cases.map(calculate)}, null, 2));
