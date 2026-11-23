"use strict";

class RoadMemoryView{
  constructor(scene,send,onReference){
    this.scene=scene;
    this.send=send;
    this.onReference=onReference;
    this.elements=id=>document.getElementById(id);
    const $=this.elements;
    this.canvas=$("roadMap");
    this.ctx=this.canvas.getContext("2d");
    this.base=document.createElement("canvas");
    this.memory={revision:-1,origin:null,cells:[],corridors:[],ridges:[],reference:[]};
    this.full=true;
    this.settingOrigin=false;
    this.showReference=false;
    this.keys=new Set();
    this.pointers=new Map();
    this.posting=false;
    this.pending=false;
    this.cellKey="";
    $("roadChoice").replaceChildren(new Option("当前通道",""),
      ...(scene.aisles||[]).map(n=>new Option("通道 "+n,String(n))));

    $("roadFit").onclick=()=>{
      this.full=!this.full;
      $("roadFit").textContent=this.full?"聚焦已记":"全场";
      this.draw();
    };
    $("roadOrigin").onclick=()=>{
      this.settingOrigin=!this.settingOrigin;
      $("roadOrigin").setAttribute("aria-pressed",String(this.settingOrigin));
      this.updateHint();
    };
    $("referenceCloud").onclick=()=>{
      this.showReference=!this.showReference;
      $("referenceCloud").setAttribute("aria-pressed",String(this.showReference));
      this.onReference();
    };
    $("startMemory").onclick=()=>this.send("memory/start",{mode:"manual"});
    $("autoMemory").onclick=()=>this.send("memory/start",{mode:"auto"});
    $("semiMemory").onclick=()=>{
      const choice=$("roadChoice").value;
      this.send("memory/start",{mode:"semi",number:choice?Number(choice):null});
    };
    $("finishMemory").onclick=()=>{
      this.release();
      this.send("memory/finish");
    };
    this.canvas.addEventListener("click",e=>{
      if(!this.settingOrigin)return;
      const p=this.pointAt(e);
      if(!p||Math.abs(p[0])>scene.x||Math.abs(p[1])>scene.y)return;
      this.settingOrigin=false;
      $("roadOrigin").setAttribute("aria-pressed","false");
      this.send("memory/origin",{x:p[0],y:p[1]});
      this.updateHint();
    });
    this.canvas.addEventListener("dblclick",e=>{
      if(this.settingOrigin||this.state?.recording)return;
      const p=this.pointAt(e);
      if(p&&Math.abs(p[0])<scene.x&&Math.abs(p[1])<scene.y)this.send("goal",{x:p[0],y:p[1]});
    });
    this.canvas.addEventListener("pointermove",e=>this.updateHint(this.pointAt(e)));
    this.canvas.addEventListener("pointerleave",()=>this.updateHint());
    new ResizeObserver(()=>this.draw()).observe(this.canvas);
    this.bindControls();
  }

  update(state,memory=this.memory){
    const finished=this.state?.recording==="manual"&&state.recording!=="manual";
    this.state=state;
    if(finished)this.release();
    this.memory=memory;
    const $=this.elements, mode=state.recording;
    $("startMemory").disabled=!!mode||!state.can_configure;
    const disabled=!!mode||!state.can_configure||!this.scene.aisles?.length;
    $("autoMemory").disabled=$("semiMemory").disabled=$("roadChoice").disabled=disabled;
    $("finishMemory").disabled=!mode;
    $("manualControls").hidden=mode!=="manual";
    const pieces=[];
    if(mode){
      pieces.push({manual:"手动建图",auto:"全自动建图",semi:"半自动建图"}[mode]);
      if(state.mapping){
        pieces.push("通道 "+state.mapping.number+(state.mapping.stage==="join"?" · 前往地头":" · 沿通道采集"));
        if(mode==="auto")pieces.push("已扫完 "+state.mapping.complete+"/"+this.scene.aisles.length);
      }
    }else if(memory.result)pieces.push(memory.result);
    if(memory.cells.length)pieces.push(memory.cells.length+" 个已记格");
    if(!pieces.length)pieces.push("地图记忆为空");
    const status=$("roadStatus"),text=pieces.join(" · ");
    if(status.textContent!==text)status.textContent=text;
    this.draw();
  }

  local(point){
    return this.memory.origin?point.map((v,i)=>v-this.memory.origin[i]):point;
  }

  updateHint(point){
    const hint=this.elements("roadHint");
    const text=this.settingOrigin?"单击图内位置设为原点":
      point&&this.memory.origin?"相对原点  X "+this.local(point)[0].toFixed(2)+" m · Y "+this.local(point)[1].toFixed(2)+" m":
        "在二维图双击设目标。黄色田垄禁行，车辆可从地头绕行。";
    if(hint.textContent!==text)hint.textContent=text;
  }

  pointAt(e){
    if(!this.transform)return null;
    const b=this.canvas.getBoundingClientRect();
    const {x,y,w,h,scale}=this.transform;
    return [x+(e.clientX-b.left-this.canvas.clientLeft-w/2)/scale,
      y-(e.clientY-b.top-this.canvas.clientTop-h/2)/scale];
  }

  draw(){
    const {clientWidth:w,clientHeight:h}=this.canvas;
    if(w<38||h<38)return;
    const dpr=Math.min(devicePixelRatio||1,2),width=Math.round(w*dpr),height=Math.round(h*dpr);
    if(this.canvas.width!==width||this.canvas.height!==height){
      this.canvas.width=width;
      this.canvas.height=height;
    }
    const {cells,corridors,ridges,origin}=this.memory;
    const car=this.state?.visual;
    let x0=-this.scene.x,x1=this.scene.x,y0=-this.scene.y,y1=this.scene.y;
    if(!this.full&&cells.length){
      x0=x1=cells[0][0]; y0=y1=cells[0][1];
      for(const [x,y] of cells){
        x0=Math.min(x0,x); x1=Math.max(x1,x);
        y0=Math.min(y0,y); y1=Math.max(y1,y);
      }
    }
    const x=(x0+x1)/2,y=(y0+y1)/2;
    const scale=Math.min((w-34)/Math.max(5,x1-x0+2),(h-34)/Math.max(4,y1-y0+2));
    this.transform={x,y,w,h,scale};
    const screen=([a,b])=>[w/2+(a-x)*scale,h/2-(b-y)*scale];
    const key=[this.memory.revision,this.full,w,h,dpr].join(",");
    if(key!==this.cellKey){
      this.cellKey=key;
      this.base.width=width;this.base.height=height;
      const c=this.base.getContext("2d");
      c.setTransform(dpr,0,0,dpr,0,0);
      const half=this.scene.cell*scale/2;
      const colors={1:"#376d5a",2:"#954b51",3:"#bd8941",4:"#30484c"};
      for(const [a,b,kind] of cells){
        const [u,v]=screen([a,b]);
        c.fillStyle=colors[kind];
        c.fillRect(u-half,v-half,2*half+.5,2*half+.5);
      }
      c.strokeStyle="#44615f";c.lineWidth=1;
      c.strokeRect(...screen([-this.scene.x,this.scene.y]),2*this.scene.x*scale,2*this.scene.y*scale);
      if(origin){
        const [u,v]=screen(origin);
        c.beginPath();c.moveTo(0,v);c.lineTo(w,v);c.moveTo(u,0);c.lineTo(u,h);c.stroke();
        c.fillStyle="#d3e2dd";c.font="11px sans-serif";c.fillText("原点",u+5,v-6);
      }
      for(const lane of corridors){
        const [u,v]=screen([lane.x,lane.y]),label="通道 "+lane.number;
        c.font="600 11px sans-serif";
        c.fillStyle="#112824";c.fillRect(u-3,v-13,c.measureText(label).width+6,15);
        c.fillStyle="#c0edd9";c.fillText(label,u,v-2);
      }
      for(const ridge of ridges){
        const [u,v]=screen([ridge.x,ridge.y]),label="垄 "+ridge.number;
        c.font="600 11px sans-serif";
        const width=c.measureText(label).width;
        c.fillStyle="#503915";c.fillRect(u-width-8,v-11,width+6,14);
        c.fillStyle="#ffe5ad";c.fillText(label,u-width-5,v);
      }
      if(!cells.length){
        c.font="13px sans-serif";c.fillStyle="#93b1ad";
        c.textAlign="center";c.fillText("开始建图后显示已观测的区域",w/2,h/2);
      }
    }
    const ctx=this.ctx;
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,w,h);
    ctx.drawImage(this.base,0,0,w,h);
    if(this.state?.route?.length>1){
      ctx.beginPath();
      this.state.route.forEach((p,i)=>{const [a,b]=screen(p);if(i)ctx.lineTo(a,b);else ctx.moveTo(a,b)});
      ctx.lineWidth=2;ctx.strokeStyle="#8fc7ff";ctx.stroke();
    }
    function marker(p,style){
      if(!p)return;
      const [a,b]=screen(p);
      ctx.fillStyle=style;ctx.beginPath();ctx.arc(a,b,4,0,2*Math.PI);ctx.fill();
    }
    marker(this.state?.goal,"#ffe08b");
    marker(car,"#d9f7ea");
  }

  bindControls(){
    const controls=this.elements("manualControls");
    const translations={ArrowUp:"w",ArrowDown:"s",ArrowLeft:"a",ArrowRight:"d"};
    const key=e=>translations[e.key]||e.key.toLowerCase();
    window.addEventListener("keydown",e=>{
      const code=key(e);
      if(this.state?.recording!=="manual"||e.target.matches?.("input,select,textarea")||!["w","a","s","d"].includes(code))return;
      e.preventDefault();
      if(!this.keys.has(code)){this.keys.add(code);this.drive();}
    });
    window.addEventListener("keyup",e=>{if(this.keys.delete(key(e)))this.drive()});
    window.addEventListener("blur",()=>this.release());
    document.addEventListener("visibilitychange",()=>{if(document.hidden)this.release()});
    for(const b of controls.querySelectorAll("[data-steer]")){
      b.onpointerdown=e=>{
        if(e.button!==0||this.state?.recording!=="manual")return;
        e.preventDefault();
        b.setPointerCapture(e.pointerId);
        this.pointers.set(e.pointerId,b.dataset.steer);
        this.drive();
      };
      const release=e=>{if(this.pointers.delete(e.pointerId))this.drive()};
      b.onpointerup=b.onpointercancel=b.onlostpointercapture=release;
    }
    setInterval(()=>{if(this.keys.size||this.pointers.size)this.drive()},105);
  }

  release(){
    this.keys.clear();
    this.pointers.clear();
    this.drive();
  }

  async drive(){
    if(this.state?.recording!=="manual")return;
    if(this.posting){this.pending=true;return;}
    this.posting=true;
    const held=new Set([...this.keys,...this.pointers.values()]);
    const motion={throttle:Number(held.has("w"))-Number(held.has("s")),
                  steering:Number(held.has("a"))-Number(held.has("d"))};
    try{
      await fetch("/api/manual",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify(motion),signal:AbortSignal.timeout(450)});
    }catch{}finally{
      this.posting=false;
      if(this.pending){this.pending=false;this.drive();}
    }
  }
}
