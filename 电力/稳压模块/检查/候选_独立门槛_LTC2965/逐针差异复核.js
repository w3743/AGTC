// Read-only comparison of the recorded official ENET exports.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const checks = path.resolve(__dirname, '..');
const inputs = [path.join(checks, '当前官方网表.enet'), path.join(__dirname, '候选.enet')];
function read(file) {
  const bytes = fs.readFileSync(file);
  const net = JSON.parse(bytes);
  const components = {};
  for (const c of Object.values(net.components)) {
    const p = c.props || {};
    if (!p.Designator) throw Error('Component missing designator');
    if (components[p.Designator]) throw Error('Duplicate designator: ' + p.Designator);
    const pins = {};
    for (const [number, info] of Object.entries(c.pinInfoMap || {})) pins[number] = info.net || '';
    components[p.Designator] = {mpn: p['Manufacturer Part'] || '', value: p.Value || '', pins};
  }
  return {sha256: crypto.createHash('sha256').update(bytes).digest('hex'), components};
}
const [current, candidate] = inputs.map(read);
const refs = [...new Set([...Object.keys(current.components), ...Object.keys(candidate.components)])].sort();
const differences = refs.filter(ref => JSON.stringify(current.components[ref]) !== JSON.stringify(candidate.components[ref]))
  .map(ref => ({ref, current: current.components[ref] || null, candidate: candidate.components[ref] || null}));
process.stdout.write(JSON.stringify({
  scope: 'Recorded official exports; pin nets, MPN and value only; not a new DRC or simulation.',
  sources: inputs.map((file, i) => ({path: file, sha256: [current, candidate][i].sha256})),
  counts: [Object.keys(current.components).length, Object.keys(candidate.components).length],
  differences
}, null, 2) + '\n');
