// Opens only this copied project in a uniquely identified CLI headless session.
const fs = require('fs'), path = require('path'), cp = require('child_process');
const exe = 'C:/Program Files/lceda-pro/lceda-pro.exe';
const project = path.resolve(__dirname, 'sc-regulator/sc-regulator.eprj3');
function call(...args) {
  const p = cp.spawnSync(exe, args, {encoding:'utf8', timeout:75000, maxBuffer:32*1024*1024});
  if (p.error || p.status !== 0) throw Error(String(p.error || p.stderr));
  const v = JSON.parse(p.stdout.trim());
  if (!v.ok) throw Error(JSON.stringify(v));
  return v.value;
}
(async()=>{
  const prior = new Set(call('session','list').map(x=>x.sessionId));
  cp.spawn(exe,['open','--path',project,'--headless','true'],{detached:true,stdio:'ignore'}).unref();
  let session;
  for(let i=0;i<40;i++) {
    await new Promise(r=>setTimeout(r,500));
    const matches=call('session','list').filter(x=>!prior.has(x.sessionId)&&x.origin==='cli'&&x.status==='alive'&&path.resolve(x.path).toLowerCase()===project.toLowerCase());
    if(matches.length===1){session=matches[0];break;}
  }
  if(!session)throw Error('Could not identify unique owned session');
  try {
    await new Promise(r=>setTimeout(r,6000));
    const body=fs.readFileSync(path.resolve(process.argv[2]),'utf8');
    const result=call('invoke','--session',session.sessionId,'--ext-uuid','eda','--timeout','60000','--code',body);
    fs.writeFileSync(path.resolve(process.argv[3]),JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify({ok:true,output:process.argv[3]}));
  } finally {
    call('session','close','--session',session.sessionId,'--destroy');
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
