"use strict";
const fs=require("node:fs");
const path=require("node:path");
const http=require("node:http");
const readline=require("node:readline");
const {spawn}=require("node:child_process");

const ROOT=__dirname;
const SIM=path.dirname(ROOT);
const CACHE=path.join(SIM,".cache");
const RUNS=path.join(ROOT,"runs");
const PYTHON=process.env.AGTC_TRAIN_PYTHON || (process.platform==="win32"?"python":"python3");
const PORT=Number(process.argv[2]||8924);
const PREVIEW_PORT=Number(process.argv[3]||8925);
const ENV={...process.env,TEMP:CACHE,TMP:CACHE,TORCHINDUCTOR_CACHE_DIR:CACHE,
  PYTHONIOENCODING:"utf-8",PYTHONUNBUFFERED:"1"};
const WEIGHTS={terrain:"terrain.pt",research:"research.pt"};
const DEFAULT={terrain:12,local:8,memory:8,threads:4,seed:112121};
const ACTIVE=new Set(["running","stopping"]);
fs.mkdirSync(CACHE,{recursive:true});
fs.mkdirSync(RUNS,{recursive:true});

let child=null;
let previewChild=null;
let preview={phase:"idle",run_id:null};
let run=restore();

function restore(){
  const names=fs.readdirSync(RUNS,{withFileTypes:true}).filter(d=>d.isDirectory()).map(d=>d.name).sort().reverse();
  for(const name of names){
    try{
      const previous=JSON.parse(fs.readFileSync(path.join(RUNS,name,"status.json"),"utf8"));
      if(ACTIVE.has(previous.state)){
        previous.state="interrupted";
        previous.message="工作台进程退出，训练未完整结束";
      }
      return previous;
    }catch{}
  }
  return null;
}

function atomicFile(file,data){
  fs.writeFileSync(file+".partial",data);
  fs.renameSync(file+".partial",file);
}

function persist(job){
  atomicFile(path.join(job.dir,"status.json"),JSON.stringify(job,null,2)+"\n");
}

function log(job,message){
  const line=String(message).slice(0,2000);
  job.tail.push(line);
  if(job.tail.length>50)job.tail.shift();
  fs.appendFileSync(path.join(job.dir,"training.log"),line+"\n");
}

function options(input){
  const cfg={};
  for(const [name,base] of Object.entries(DEFAULT)){
    const max=name==="threads"?16:name==="seed"?2147483647:200;
    const value=input[name]??base;
    if(typeof value!=="number"||!Number.isInteger(value)||value<1||value>max)
      throw new Error(name+" 必须为 1–"+max+" 的整数");
    cfg[name]=value;
  }
  return cfg;
}

function start(input){
  if(child||run&&ACTIVE.has(run.state))throw new Error("已有训练正在运行");
  const config=options(input);
  const id=new Date().toISOString().replace(/[:.]/g,"-");
  const dir=path.join(RUNS,id);
  fs.mkdirSync(dir);
  const job={id,dir,config,started:new Date().toISOString(),state:"running",phase:"设备探测",
    message:"按 NPU → GPU → CPU 验证训练能力",probes:[],backend:null,data:null,epochs:{},tail:[],
    checkpoints:[],report:null};
  run=job;
  persist(job);
  const args=["-B",path.join(ROOT,"worker.py"),...Object.entries(config).flatMap(([k,v])=>["--"+k,String(v)])];
  const proc=spawn(PYTHON,args,{cwd:ROOT,windowsHide:true,env:ENV,stdio:["ignore","pipe","pipe"]});
  child=proc;
  let failure="";
  function fail(err){
    if(failure)return;
    failure=String(err.message||err);
    if(proc.pid)proc.kill();
  }
  readline.createInterface({input:proc.stdout}).on("line",line=>{
    try{
      if(!line.startsWith("@AGTC "))log(job,line);
      else{
        const record=JSON.parse(line.slice(6));
        if(record.event==="checkpoint"){
          const file=WEIGHTS[record.name];
          if(!Object.hasOwn(WEIGHTS,record.name)||typeof record.bytes_b64!=="string")throw new Error("无法识别的检查点");
          const data=Buffer.from(record.bytes_b64,"base64");
          if(data.length<500||data.length>2*1024*1024)throw new Error("检查点大小异常");
          atomicFile(path.join(dir,file),data);
          if(!job.checkpoints.includes(file))job.checkpoints.push(file);
        }else{
          fs.appendFileSync(path.join(dir,"metrics.jsonl"),JSON.stringify(record)+"\n");
          if(record.event==="probe")job.probes.push(record);
          else if(record.event==="backend")job.backend=record;
          else if(record.event==="data")job.data=record;
          else if(record.event==="stage"){
            job.phase=record.phase;
            job.message=record.message;
          }else if(record.event==="fallback"){
            job.epochs={};
            log(job,"设备退回 · "+record.device+" · "+record.detail);
          }else if(record.event==="epoch"){
            (job.epochs[record.model]??=[]).push(record);
            log(job,record.model+" "+record.epoch+"/"+record.epochs+" · 验证损失 "+record.val_loss.toFixed(4));
          }else if(record.event==="complete")job.report=record.report;
        }
      }
      persist(job);
    }catch(err){fail(err)}
  });
  readline.createInterface({input:proc.stderr}).on("line",line=>{try{log(job,line)}catch(err){fail(err)}});
  proc.on("error",fail);
  proc.on("close",code=>{
    child=null;
    job.finished=new Date().toISOString();
    if(job.state==="stopping"){
      job.state="stopped"; job.phase="已停止"; job.message="保留已写出的训练指标，继续训练请开始新一轮";
    }else if(code===0&&job.report&&job.checkpoints.length===2&&!failure){
      job.state="completed"; job.phase="训练完成"; job.message="已保存两个检查点，可在独立仿真中试跑";
    }else{
      job.state="failed"; job.phase="训练失败"; job.message=failure||"训练进程退出。详情见训练日志";
    }
    persist(job);
  });
  return job;
}

function stop(){
  if(!child||run.state!=="running")throw new Error("当前没有可停止的训练");
  run.state="stopping";
  run.phase="正在停止";
  run.message="正在结束训练子进程";
  persist(run);
  child.kill();
}

async function launchPreview(){
  if(!run||run.state!=="completed")throw new Error("需要一轮完整结束的训练");
  if(preview.phase==="starting")throw new Error("正在加载预览");
  if(preview.phase==="ready"&&preview.run_id===run.id)return preview;
  const current=run;
  if(previewChild){
    const old=previewChild;
    await new Promise(resolve=>{old.once("close",resolve);old.kill()});
  }
  preview={phase:"starting",run_id:current.id};
  const args=["-B",path.join(SIM,"offroad_server.py"),"--port",String(PREVIEW_PORT),
    "--terrain-checkpoint",path.join(current.dir,WEIGHTS.terrain),
    "--research-checkpoint",path.join(current.dir,WEIGHTS.research)];
  const proc=spawn(PYTHON,args,{cwd:SIM,windowsHide:true,env:ENV,stdio:["ignore","pipe","pipe"]});
  previewChild=proc;
  const url="http://127.0.0.1:"+PREVIEW_PORT;
  return await new Promise((resolve,reject)=>{
    let ready=false;
    const timeout=setTimeout(()=>{
      proc.kill();
      reject(new Error("预览进程启动超时"));
    },25000);
    function failed(err){
      clearTimeout(timeout);
      if(previewChild===proc){
        preview={phase:"failed",run_id:current.id,message:err.message};
        previewChild=null;
      }
      reject(err);
    }
    proc.on("error",failed);
    proc.on("close",()=>{
      if(!ready)failed(new Error("预览进程退出，请查看本轮 preview.log"));
      else if(previewChild===proc){
        preview={phase:"idle",run_id:null};
        previewChild=null;
      }
    });
    function output(line){
      fs.appendFileSync(path.join(current.dir,"preview.log"),line+"\n");
      if(!ready&&line.startsWith(url+"/")){
        clearTimeout(timeout);
        ready=true;
        preview={phase:"ready",run_id:current.id,url};
        resolve(preview);
      }
    }
    readline.createInterface({input:proc.stdout}).on("line",output);
    readline.createInterface({input:proc.stderr}).on("line",output);
  });
}

async function requestBody(req){
  let data="";
  for await(const chunk of req){
    data+=chunk.toString("utf8");
    if(data.length>4096)throw new Error("请求数据过长");
  }
  const body=data?JSON.parse(data):{};
  if(!body||typeof body!=="object"||Array.isArray(body))throw new Error("参数格式无效");
  return body;
}

const FILES={"/":["workbench.html","text/html"],"/workbench.js":["workbench.js","text/javascript"],
  "/workbench.css":["workbench.css","text/css"]};

function respond(res,code,body,type="application/json"){
  res.writeHead(code,{"Content-Type":type+"; charset=utf-8","Cache-Control":"no-store"});
  res.end(type==="application/json"?JSON.stringify(body):body);
}

const server=http.createServer(async(req,res)=>{
  const target=req.url;
  try{
    if(req.method==="GET"&&Object.hasOwn(FILES,target)){
      const [name,type]=FILES[target];
      respond(res,200,fs.readFileSync(path.join(ROOT,name)),type);
    }else if(req.method==="GET"&&target==="/api/state"){
      respond(res,200,{run,preview,defaults:DEFAULT,npu_name:process.env.AGTC_NPU_NAME||null,
                      baseline_url:"http://127.0.0.1:8922"});
    }else if(req.method==="POST"&&target==="/api/train"){
      respond(res,202,{id:start(await requestBody(req)).id});
    }else if(req.method==="POST"&&target==="/api/stop"){
      stop(); respond(res,202,{ok:true});
    }else if(req.method==="POST"&&target==="/api/preview"){
      respond(res,200,await launchPreview());
    }else{
      respond(res,404,{error:"该页面不存在"});
    }
  }catch(err){
    respond(res,400,{error:err.message});
  }
});
function close(){
  if(child)child.kill();
  if(previewChild)previewChild.kill();
  server.close(()=>process.exit(0));
}
process.on("SIGINT",close);
process.on("SIGTERM",close);
server.listen(PORT,"127.0.0.1",()=>console.log("AGTC 本地训练 http://127.0.0.1:"+PORT+"/"));
