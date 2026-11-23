const $=id=>document.getElementById(id), canvas=$("scene"), ctx=canvas.getContext("2d");
const thinking=location.pathname==="/think";
const set=(id,v)=>{const element=$(id),text=String(v);if(element.textContent!==text)element.textContent=text};
const f=(v,n=2)=>Number(v).toFixed(n);
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
let scene,state,camera,field,roadView,drag=null,cloud=false,follow=false,configKey="";
let mapLayer={map:[],blocked:[],revision:0}, mapBusy=false, mapRefresh=0, photoRevision=0;
let memoryBusy=false,memoryRefresh=0;
let fromCar,toCar,arrived=0,blendMs=0,drawId=0;
const initial=()=>({yaw:-.22,pitch:.91,zoom:.92,x:0,y:0});
camera=initial();
if(thinking){
  document.body.classList.add("thinking"); document.title="AGTC · 地形决策与风险记忆";
  set("title","神经元实时地形决策"); set("viewTitle","路线与视觉风险");
  set("sideTitle","从双目输入到决策"); $("other").href="/"; set("other","三维地图 ↗");
}

function surface(x,y){
  const h=scene.heights, n=h.length, m=h[0].length;
  const i=clamp((x+scene.x)/(2*scene.x)*(m-1),0,m-1.001), j=clamp((y+scene.y)/(2*scene.y)*(n-1),0,n-1.001);
  const u=Math.floor(i),v=Math.floor(j),a=i-u,b=j-v;
  return (1-b)*((1-a)*h[v][u]+a*h[v][u+1])+b*((1-a)*h[v+1][u]+a*h[v+1][u+1]);
}

function viewFor(w,h,car){
  const {yaw,pitch,zoom}=camera;
  const [x,y]=follow&&car?car:[camera.x,camera.y];
  return {x,y,c:Math.cos(yaw),s:Math.sin(yaw),cp:Math.cos(pitch),sp:Math.sin(pitch),
    scale:Math.min(w/(2*scene.x+3),h/(2*scene.y+1))*zoom};
}

function projector(w,h,{x,y,c,s,cp,sp,scale}){
  return ([u,v,z])=>{
    u-=x;v-=y;
    const r=u*c-v*s, t=u*s+v*c;
    return [w/2+r*scale,h/2-(t*sp+z*cp)*scale,-t*cp+z*sp];
  };
}

function interpolatedCar(now){
  if(!fromCar||!blendMs)return toCar;
  const t=clamp((now-arrived)/blendMs,0,1);
  const pose=toCar.map((v,i)=>fromCar[i]+(v-fromCar[i])*t);
  const norm=Math.hypot(...pose.slice(3));
  for(let i=3;i<7;i++)pose[i]/=norm;
  return pose;
}

function paint(){
  if(!drawId)drawId=requestAnimationFrame(draw);
}

function draw(now){
  drawId=0;
  if(!scene || !state) return;
  const bounds=canvas.getBoundingClientRect(), pd=Math.min(window.devicePixelRatio||1,2);
  if(canvas.width!==Math.round(bounds.width*pd)||canvas.height!==Math.round(bounds.height*pd)){
    canvas.width=Math.round(bounds.width*pd);canvas.height=Math.round(bounds.height*pd);
  }
  const w=bounds.width,h=bounds.height;
  if(!w||!h)return;
  const car=interpolatedCar(now);
  const view=viewFor(w,h,car),P=projector(w,h,view);
  field.draw(w,h,pd,view,P);
  ctx.setTransform(pd,0,0,pd,0,0);
  ctx.clearRect(0,0,w,h);
  const list=[];
  function polygon(world, fill, stroke){
    const uv=world.map(P);
    list.push({uv,fill,stroke,d:uv.reduce((sum,v)=>sum+v[2],0)/uv.length});
  }
  if(car){
    const [px,py,pz,qw,qx,qy,qz]=car;
    const rotate=([x,y,z])=>{
      const a=2*(qy*z-qz*y),b=2*(qz*x-qx*z),c=2*(qx*y-qy*x);
      return [x+qw*a+(qy*c-qz*b)+px,y+qw*b+(qz*a-qx*c)+py,z+qw*c+(qx*b-qy*a)+pz];
    };
    const cuboid=(x0,x1,y0,y1,z0,z1,col)=>{
      const vertices=[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]].map(rotate);
      for(const ids of [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[3,0,4,7]])
        polygon(ids.map(i=>vertices[i]),col,"#283d3f");
    };
    cuboid(-.4,.4,-.15,.15,-.075,.075,"#d8ead8");
    cuboid(-.31,.46,-.24,-.17,-.56,-.48,"#a8b0a4");
    cuboid(-.31,.46,.17,.24,-.56,-.48,"#a8b0a4");
    const cameraZ=state.config.camera_height-scene.carZ;
    const roof=state.config.height-scene.carZ;
    cuboid(-.06,.05,-.043,.043,.075,roof,"#a8bbad");
    cuboid(.28,.40,-.063,.063,cameraZ-.025,cameraZ+.025,"#ffd787");
    const angle=state.camera_yaw*Math.PI/180;
    const ray=[rotate([.34,0,cameraZ]),rotate([.34+1.45*Math.cos(angle),1.45*Math.sin(angle),cameraZ])].map(P);
    list.push({uv:ray,fill:"#ffd782",stroke:"#ffd782",ray:true,d:(ray[0][2]+ray[1][2])/2});
  }
  list.sort((a,b)=>a.d-b.d);
  for(const p of list){
    ctx.beginPath();ctx.moveTo(...p.uv[0].slice(0,2));
    for(let i=1;i<p.uv.length;i++)ctx.lineTo(...p.uv[i].slice(0,2));
    if(!p.ray){ctx.closePath();ctx.fillStyle=p.fill;ctx.fill()}
    ctx.strokeStyle=p.stroke;ctx.lineWidth=p.ray?2.2:.6;ctx.stroke();
  }
  if(cloud||thinking){
    for(const [x,y,z,risk] of mapLayer.map){
      const [u,v]=P([x,y,z+.06]);
      ctx.fillStyle=risk>.6?"#ff9d867f":"#8fedc88c";
      ctx.fillRect(u-1.7,v-1.7,3.4,3.4);
    }
  }
  if(roadView&&(roadView.showReference||state.recording!=null)){
    const points=roadView.memory.reference;
    const stride=Math.max(1,Math.ceil(points.length/1600));
    for(let i=0;i<points.length;i+=stride){
      const [x,y,z,rock]=points[i], [u,v]=P([x,y,z+.05]);
      ctx.fillStyle=rock?"#ffc87990":"#71d5ef78";
      ctx.fillRect(u-1.4,v-1.4,2.8,2.8);
    }
  }
  for(const [x,y] of mapLayer.blocked){
    const [u,v]=P([x,y,surface(x,y)+.03]);
    ctx.fillStyle="#f49b7d5e";ctx.fillRect(u-2.5,v-2.5,5,5);
  }
  function line(points,col,dash){
    if(points.length<2)return;
    const sticks=[];
    for(let i=0;i<points.length-1;i++){
      const a=points[i],b=points[i+1],count=Math.max(1,Math.ceil(Math.hypot(b[0]-a[0],b[1]-a[1])/.3));
      for(let j=0;j<count;j++)sticks.push([a[0]+(b[0]-a[0])*j/count,a[1]+(b[1]-a[1])*j/count]);
    }
    sticks.push(points.at(-1));
    ctx.beginPath();ctx.setLineDash(dash);ctx.strokeStyle=col;ctx.lineWidth=2.6;
    sticks.forEach(([x,y],i)=>{const [u,v]=P([x,y,surface(x,y)+.09]);i?ctx.lineTo(u,v):ctx.moveTo(u,v)});
    ctx.stroke();ctx.setLineDash([]);
  }
  line(state.route,"#9dcaff",[7,5]);line(state.trace,"#92e5af",[]);
  function flag(x,y,colour,label){
    const z=surface(x,y), [u,v]=P([x,y,z]), [a,b]=P([x,y,z+.85]);
    ctx.beginPath();ctx.moveTo(u,v);ctx.lineTo(a,b);ctx.strokeStyle=colour;ctx.lineWidth=2;ctx.stroke();
    ctx.beginPath();ctx.arc(a,b,4,0,2*Math.PI);ctx.fillStyle=colour;ctx.fill();
    if(label){ctx.font="12px sans-serif";ctx.fillText(label,a+7,b-3)}
  }
  if(state.goal)flag(...state.goal,"#ffd782","目标");
  if(state.look&&thinking)flag(...state.look.point,"#b8eedd","评估点");
  if(car){
    const [x,y]=P([car[0],car[1],car[2]+.25]);
    ctx.fillStyle="#d9f6da";ctx.font="700 12px sans-serif";ctx.fillText("小车",x-13,y-9);
  }
  if(now<arrived+blendMs)paint();
}

async function submit(path,data={}){
  try{
    const res=await fetch("/api/"+path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
    const reply=await res.json();
    set("notice",reply.message);
    return reply;
  }catch{set("notice","仿真未响应")}
}
canvas.addEventListener("contextmenu",e=>e.preventDefault());
canvas.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY,pan:e.button===2||e.shiftKey};canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener("pointerup",()=>{drag=null});
canvas.addEventListener("pointercancel",()=>{drag=null});
canvas.addEventListener("pointermove",e=>{
  if(!drag||!scene)return;
  const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
  if(drag.pan){
    if(follow&&state?.vehicle){
      camera.x=state.vehicle[0];camera.y=state.vehicle[1];
      follow=false;$("follow").setAttribute("aria-pressed","false");
    }
    const {width:w,height:h}=canvas.getBoundingClientRect();
    const {c,s,scale,sp}=viewFor(w,h);
    camera.x+=(-c*dx+s*dy/sp)/scale;camera.y+=(s*dx+c*dy/sp)/scale;
  }else{camera.yaw+=dx*.008;camera.pitch=clamp(camera.pitch+dy*.006,.27,Math.PI/2)}
  drag={...drag,x:e.clientX,y:e.clientY};paint();
});
canvas.addEventListener("wheel",e=>{
  e.preventDefault();camera.zoom=clamp(camera.zoom*Math.exp(-e.deltaY*.0012),.55,12);paint();
},{passive:false});
function overview(c){camera=c;follow=false;$("follow").setAttribute("aria-pressed","false");paint()}
$("top").onclick=()=>overview({...initial(),pitch:Math.PI/2,yaw:0,zoom:1});
$("orbit").onclick=()=>overview(initial());
$("follow").onclick=()=>{follow=!follow;if(follow)camera.zoom=Math.max(camera.zoom,4);$("follow").setAttribute("aria-pressed",follow);paint()};
$("cloud").onclick=()=>{cloud=!cloud;$("cloud").setAttribute("aria-pressed",cloud);paint()};
$("demo").onclick=()=>{if(scene)submit("goal",{x:scene.goal[0],y:scene.goal[1]})};
$("stop").onclick=()=>submit("stop");
$("reset").onclick=()=>submit("reset");
new ResizeObserver(paint).observe(canvas);

function update(s){
  if(!state||s.sim_time!==state.sim_time||s.vehicle.some((v,i)=>v!==state.vehicle[i])){
    const now=performance.now(),next=s.vehicle.slice(),previous=interpolatedCar(now);
    const matching=previous&&Math.hypot(next[0]-previous[0],next[1]-previous[1])<1.5&&s.sim_time>=state.sim_time;
    fromCar=matching?previous:null;
    if(matching&&next.slice(3).reduce((a,v,i)=>a+v*previous[i+3],0)<0){
      for(let i=3;i<7;i++)next[i]*=-1;
    }
    toCar=next;
    blendMs=matching?clamp(now-arrived,35,160):0;
    arrived=now;
  }
  state=s;
  roadView.update(s);
  set("phase",s.phase);set("reward",f(s.reward));set("distance",f(s.distance)+" m · "+f(s.seconds,1)+" s");
  const alert=$("alert");
  alert.hidden=!s.alert;
  const alertMessage=s.alert?"车辆自救失败 · "+s.alert.message:"";
  if(alert.textContent!==alertMessage)alert.textContent=alertMessage;
  const target=s.goal?roadView.local(s.goal):null;
  set("target",target?"目标 "+target.map(x=>f(x,1)).join(", ")+" m":"在二维道路地图设目标");
  const newKey=JSON.stringify(s.config);
  if(newKey!==configKey){
    $("vehicleHeight").value=f(s.config.height*100,0);
    $("cameraHeight").value=f(s.config.camera_height*100,0);
    $("maxSlope").value=s.config.max_slope;
    $("cameraYaw").value=Math.round((s.config.camera_yaw%360+360)%360);
    configKey=newKey;
  }
  $("settingsFields").disabled=!s.can_configure;
  set("settingsMessage",s.can_configure?"小车已静止，可调整参数":"有任务或车辆未停稳，参数已锁定");
  set("vision",(s.stereo_size?s.stereo_size.join("×")+" · ":"")+s.cloud+" 个当前双目测点");
  const visual=s.visual?[...roadView.local(s.visual.slice(0,2)),s.visual[2]]:null;
  set("pose",visual?"定位 "+visual.map(x=>f(x)).join(", ")+" m · "+s.markers+" 个视觉地标":"视觉定位暂不可用");
  set("scan","云台 "+f(s.camera_yaw,0)+"° · "+(s.scan_attempts?"搜索 "+s.scan_attempts+" 帧":"恢复定位 "+s.recoveries+" 次"));
  set("map",s.observed_cells+" 个已观测格");set("riskCells",s.hazard_cells+" 个风险格限制通行");
  const k=s.look;
  set("network",k&&k.known?"通过评分 "+f(k.pass_probability*100,0)+"%":"等待评估点测量");
  set("networkMeta",k&&k.known?"坡角 "+f(Math.atan(k.grade)*180/Math.PI,1)+"° · 预测路面奖励 "+f(k.network_reward):"坡度 · 高差 · 粗糙度 · 观测覆盖 · 障碍");
  const m=s.research;
  set("networkMemory",m&&m.ports?"7×7 局部风险 "+f(m.local_risk*100,0)+"% · RNN 记忆风险 "+f(m.memory_risk*100,0)+"%":"局部神经元与 RNN 等待路线数据");
  set("decision",s.reason);
  set("command","实测 "+f(s.speed)+" m/s · 转向 "+f(s.cmd[1])+" rad/s · 当前步 "+f(s.latency,1)+" ms");
  set("crossed",s.passed?"跨坡 已记录":"跨坡 未记录");set("detour",s.detour?"绕行 已选择":"绕行 未触发");
  set("safety",s.alert?"自救失败":s.phase==="自救后退"?"正在沿原路自救":s.flipped?"翻车":
    s.collision?"碰撞":s.backtrack_count?"自救成功 "+s.backtrack_count+" 次":"未检出碰撞、翻车或卡住");
  paint();
}
$("settingsForm").addEventListener("submit",e=>{
  e.preventDefault();
  submit("config",{height:Number($("vehicleHeight").value)/100,camera_height:Number($("cameraHeight").value)/100,
    max_slope:Number($("maxSlope").value),camera_yaw:Number($("cameraYaw").value)});
});
async function run(){
  try{
    if(!scene){
      const geometry=await(await fetch("/api/scene")).json();
      field=new FieldLayer($("field"),geometry);
      roadView=new RoadMemoryView(geometry,submit,paint);
      scene=geometry;
      set("fieldSize",scene.x*2+" × "+scene.y*2+" m");
      set("demo",scene.kind==="farm"?"田间巡行演示":"跨坡和绕行演示");
    }
    const data=await(await fetch("/api/live")).json();
    const now=performance.now();
    if(data.vision_revision!==mapLayer.revision&&!mapBusy&&now>=mapRefresh){
      mapBusy=true;
      mapRefresh=now+180;
      fetch("/api/map").then(r=>r.json()).then(v=>{mapLayer=v;paint()}).catch(()=>{}).finally(()=>{mapBusy=false});
    }
    if(!thinking&&photoRevision!==data.preview_revision){
      photoRevision=data.preview_revision;
      for(const id of ["left","right"])$(id).src="/api/camera/"+id+".jpg?v="+photoRevision;
    }
    update(data);
    if(data.memory_revision!==roadView.memory.revision&&!memoryBusy&&now>=memoryRefresh){
      memoryBusy=true;
      memoryRefresh=now+180;
      fetch("/api/memory").then(r=>r.json()).then(memory=>{roadView.update(state,memory);paint()})
        .catch(()=>{}).finally(()=>{memoryBusy=false});
    }
  }catch{set("phase","仿真连接中断")}
  setTimeout(run,state?.can_configure?220:45);
}
run();
