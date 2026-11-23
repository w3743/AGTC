const $=id=>document.getElementById(id);
const set=(id,value)=>{$(id).textContent=value};
const names=["地形网络","局部神经元","RNN 记忆"];
const stateNames={running:"训练中",stopping:"正在停止",completed:"训练完成",failed:"训练异常",
  stopped:"已停止",interrupted:"训练中断"};
let state=null;
let mode="baseline";
let busy=false;
const fixed=(n,d=4)=>Number.isFinite(n)?n.toFixed(d):"—";

function chart(){
  const canvas=$("chart"),ctx=canvas.getContext("2d");
  const w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);
  const rows=state?.run?.epochs?.[$("model").value]||[];
  ctx.font="12px Consolas, monospace";
  if(!rows.length){
    ctx.fillStyle="#a4b4bb";
    ctx.fillText("等待首轮训练指标",18,32);
    return;
  }
  const maxEpoch=rows[0].epochs;
  const values=rows.flatMap(r=>[r.val_loss,r.train_loss]).filter(Number.isFinite);
  let lo=Math.min(...values),hi=Math.max(...values);
  const spread=Math.max(hi-lo,.005);
  lo=Math.max(0,lo-spread*.12);
  hi+=spread*.12;
  const x=e=>51+e/Math.max(1,maxEpoch)*(w-76);
  const y=v=>12+(hi-v)/(hi-lo)*(h-37);
  ctx.strokeStyle="#324952";
  ctx.fillStyle="#a4b4bb";
  for(let i=0;i<=3;i++){
    const value=hi-i*(hi-lo)/3;
    ctx.beginPath();ctx.moveTo(51,y(value));ctx.lineTo(w-12,y(value));ctx.stroke();
    ctx.fillText(fixed(value,3),5,y(value)+4);
  }
  ctx.fillText("0",48,h-7);
  ctx.fillText(maxEpoch+" 轮",w-64,h-7);
  for(const [key,colour] of [["train_loss","#8ebfff"],["val_loss","#90e2bc"]]){
    const points=rows.filter(q=>Number.isFinite(q[key]));
    if(!points.length)continue;
    ctx.strokeStyle=colour;ctx.fillStyle=colour;ctx.lineWidth=2;ctx.beginPath();
    points.forEach((r,i)=>i?ctx.lineTo(x(r.epoch),y(r[key])):ctx.moveTo(x(r.epoch),y(r[key])));
    ctx.stroke();ctx.beginPath();
    const end=points.at(-1);ctx.arc(x(end.epoch),y(end[key]),3,0,Math.PI*2);ctx.fill();
  }
}

function sources(){
  const wanted=mode==="preview"&&state?.preview?.phase==="ready"?state.preview.url:state?.baseline_url;
  if(!wanted)return;
  const isPreview=wanted===state?.preview?.url&&mode==="preview";
  for(const [id,route] of [["map","/"],["think","/think"]]){
    if($(id).getAttribute("src")!==wanted+route)$(id).src=wanted+route;
  }
  set("weights",isPreview?"本轮检查点":"基线模型");
  set("liveHint",isPreview?"独立试跑本轮检查点。双击地面设目标，不影响基线仿真。":"双击地图设定目标。两个视图共用同一个基线仿真任务。");
}

function render(){
  const job=state?.run;
  const active=["running","stopping"].includes(job?.state);
  $("inputs").disabled=active||busy;
  $("start").disabled=active||busy;
  $("stop").disabled=job?.state!=="running"||busy;
  $("preview").disabled=busy||job?.state!=="completed"||state.preview?.phase==="starting";
  set("status",stateNames[job?.state]||"等待开始");
  $("status").classList.toggle("error",["failed","interrupted"].includes(job?.state));
  set("backend",job?.backend?.name||"等待设备探测");
  set("stage",job?.phase||"—");
  const camera=job?.data?.camera;
  set("captureReport",camera ? (camera.match_size.join("×")+" 双目匹配 · "+camera.stereo_pairs+" 对双目帧（"+camera.localized_pairs+" 对定位成功） · "+
    (camera.interval_count ? camera.interval_min_ms+"–"+camera.interval_max_ms+" ms 组内帧间隔":"无连续采样")+
    " · "+(job.data.physical_stereo||0)+" 条双目地形特征") : (job&&!active?"本轮无 4K 采样记录":""));
  const total=job?job.config.terrain+job.config.local+job.config.memory:0;
  const done=names.reduce((n,k)=>n+(job?.epochs?.[k]?.at(-1)?.epoch||0),0);
  set("count",job?done+" / "+total:"—");
  const bar=$("progress");
  bar.max=total||1;bar.value=done;
  if(!busy)set("message",job?.message||"设置轮数后点击「开始训练」。");
  const rows=job?.probes?.length?job.probes:[{kind:"NPU",detail:state?.npu_name?"已检测到 "+state.npu_name+"，待训练探测":"启动训练时探测"}];
  $("probes").replaceChildren(...rows.map(p=>{
    const li=document.createElement("li");
    li.textContent=p.kind+" · "+p.detail;
    if(p.ok)li.className="ok";
    return li;
  }));
  $("summaryRows").replaceChildren(...names.map(name=>{
    const tr=document.createElement("tr");
    const recent=job?.epochs?.[name]?.at(-1);
    const selected=job?.report?.selected?.[name];
    const row=selected||recent;
    const epoch=selected?.selected_epoch??recent?.epoch;
    for(const val of [name,epoch===0?"基线":(epoch??"—"),fixed(row?.val_loss),
      Number.isFinite(row?.label_match)?fixed(100*row.label_match,1)+"%":"—"]){
      const td=document.createElement("td");td.textContent=val;tr.append(td);
    }
    return tr;
  }));
  set("output",job?job.dir+"  ·  "+(job.checkpoints.length?job.checkpoints.join(" / "):"等待检查点"):"暂无运行记录");
  set("log",job?.tail?.join("\n")||"暂无日志");
  chart();
  sources();
}

async function send(action,body){
  busy=true;
  render();
  try{
    const resp=await fetch("/api/"+action,{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(body||{})});
    const payload=await resp.json();
    if(!resp.ok)throw new Error(payload.error);
    if(action==="preview"){
      state.preview=payload;
      mode="preview";
      sources();
      $("live").scrollIntoView({behavior:"smooth"});
    }
  }catch(error){set("message",error.message);await new Promise(resolve=>setTimeout(resolve,1800))}
  finally{busy=false;await refreshOnce()}
}

async function refreshOnce(){
  try{
    state=await(await fetch("/api/state")).json();
    render();
  }catch{set("status","连接中断")}
}
async function poll(){
  await refreshOnce();
  setTimeout(poll,750);
}
$("model").onchange=chart;
$("form").onsubmit=event=>{
  event.preventDefault();
  const fields=Object.fromEntries([...new FormData(event.currentTarget)].map(([key,value])=>[key,Number(value)]));
  send("train",fields);
};
$("stop").onclick=()=>send("stop");
$("preview").onclick=()=>send("preview");
$("baseline").onclick=()=>{mode="baseline";sources()};
$("showLive").onclick=()=>$("live").scrollIntoView({behavior:"smooth"});
$("layout").onchange=event=>{
  const v=event.target.value;
  $("panes").classList.toggle("single",v!=="both");
  $("map").hidden=v==="think";
  $("think").hidden=v==="map";
};
poll();
