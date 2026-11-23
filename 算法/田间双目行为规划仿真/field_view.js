"use strict";

// The terrain and plants are uploaded once. Orbiting updates only the projection.
function fieldGeometry(scene,stride){
  const vertices=[],heights=scene.heights;
  function triangle(a,b,c,color){
    for(const p of [a,b,c])vertices.push(...p,...color);
  }
  function grid(i,j){
    return [i*2*scene.x/(heights[0].length-1)-scene.x,j*2*scene.y/(heights.length-1)-scene.y,heights[j][i]];
  }
  for(let j=0;j<heights.length-1;j+=stride){
    const v=Math.min(j+stride,heights.length-1);
    for(let i=0;i<heights[0].length-1;i+=stride){
      const u=Math.min(i+stride,heights[0].length-1);
      const a=grid(i,j),b=grid(u,j),c=grid(u,v),d=grid(i,v);
      const z=(a[2]+b[2]+c[2]+d[2])/4;
      const shade=.30*(b[2]-a[2])+.36*(d[2]-a[2])+.055*z;
      const y=(a[1]+d[1])/2,spacing=scene.row_spacing;
      const row=spacing?Math.abs(((y%spacing)+spacing)%spacing-spacing/2):0;
      const seedBed=spacing&&Math.abs(a[0])<19.8&&Math.abs(y)<14.4&&row<.37;
      const soil=scene.kind==="farm"?(seedBed?[.41,.30,.17]:[.33,.24,.145]):[.25,.36,.33];
      const texture=.015*Math.sin(i*11.21+j*3.42);
      const color=soil.map(x=>x+shade+texture);
      triangle(a,b,c,color);
      triangle(a,c,d,color);
    }
  }
  for(const rock of scene.rocks){
    for(const face of rock.faces){
      const [a,b,c]=face.map(i=>rock.vertices[i]);
      const light=.045*(b[2]+c[2]-2*a[2]);
      triangle(a,b,c,[.51+light,.42+light,.33+light]);
    }
  }
  for(const [x,y,z,h,width,weed] of scene.plants||[]){
    const green=weed?[.30,.47,.16]:[.48,.51,.22];
    const grain=weed?[.39,.54,.20]:[.81,.66,.33];
    for(const [dx,dy] of [[width,.025],[.025,width]]){
      const a=[x-.22*dx,y-.22*dy,z],b=[x+.22*dx,y+.22*dy,z];
      const c=[x+.32*dx,y+.32*dy,z+h*.75],d=[x-.32*dx,y-.32*dy,z+h*.73];
      triangle(a,b,c,green);
      triangle(a,c,d,green);
      triangle(c,[x,y,z+h],d,grain);
      triangle(a,d,[x-dx,y-dy,z+h*.47],green);
      triangle(b,[x+dx,y+dy,z+h*.54],c,green);
    }
  }
  return new Float32Array(vertices);
}

class FieldLayer{
  constructor(canvas,scene){
    this.canvas=canvas;
    this.scene=scene;
    this.lastKey="";
    this.gl=canvas.getContext("webgl2",{antialias:true,alpha:false,powerPreference:"high-performance"});
    if(this.gl){
      this.setupGpu();
      canvas.addEventListener("webglcontextlost",e=>e.preventDefault());
      canvas.addEventListener("webglcontextrestored",()=>{
        this.setupGpu();
        this.lastKey="";
        if(this.lastArgs)this.draw(...this.lastArgs);
      });
    }else{
      this.cpu=canvas.getContext("2d",{alpha:false});
      this.vertices=fieldGeometry(scene,4);
      canvas.dataset.renderer="Canvas 2D";
    }
  }
  setupGpu(){
    const gl=this.gl,program=gl.createProgram();
    const vertex=`#version 300 es
      layout(location=0) in vec3 position;
      layout(location=1) in vec3 color;
      uniform vec4 rotation;
      uniform vec2 center;
      uniform vec3 scale;
      out vec3 tint;
      void main(){
        vec2 p=position.xy-center;
        float r=dot(p,vec2(rotation.x,-rotation.y));
        float t=dot(p,vec2(rotation.y,rotation.x));
        gl_Position=vec4(r*scale.x,(t*rotation.z+position.z*rotation.w)*scale.y,
                         (t*rotation.w-position.z*rotation.z)*scale.z,1.0);
        tint=color;
      }`;
    const fragment=`#version 300 es
      precision mediump float;
      in vec3 tint;
      out vec4 pixel;
      void main(){pixel=vec4(tint,1.0);}`;
    for(const [type,source] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]){
      const shader=gl.createShader(type);
      gl.shaderSource(shader,source);
      gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));
      gl.attachShader(program,shader);
      gl.deleteShader(shader);
    }
    gl.linkProgram(program);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
    this.rotation=gl.getUniformLocation(program,"rotation");
    this.center=gl.getUniformLocation(program,"center");
    this.scale=gl.getUniformLocation(program,"scale");
    const vertices=fieldGeometry(this.scene,1);
    const buffer=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    gl.bufferData(gl.ARRAY_BUFFER,vertices,gl.STATIC_DRAW);
    gl.vertexAttribPointer(0,3,gl.FLOAT,false,24,0);
    gl.vertexAttribPointer(1,3,gl.FLOAT,false,24,12);
    gl.enableVertexAttribArray(0);
    gl.enableVertexAttribArray(1);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(.063,.125,.145,1);
    this.count=vertices.length/6;
    const info=gl.getExtension("WEBGL_debug_renderer_info");
    this.canvas.dataset.renderer="WebGL 2";
    if(info)this.canvas.dataset.device=gl.getParameter(info.UNMASKED_RENDERER_WEBGL);
  }
  draw(w,h,pd,view,project){
    this.lastArgs=[w,h,pd,view,project];
    if(this.gl?.isContextLost())return;
    const {c,s,sp,cp,x,y,scale}=view;
    const key=[w,h,pd,c,s,sp,cp,x,y,scale].join(",");
    if(key===this.lastKey)return;
    this.lastKey=key;
    const canvas=this.canvas;
    const width=Math.round(w*pd),height=Math.round(h*pd);
    if(canvas.width!==width||canvas.height!==height){
      canvas.width=width;
      canvas.height=height;
    }
    const gl=this.gl;
    if(gl){
      gl.viewport(0,0,width,height);
      gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
      gl.uniform4f(this.rotation,c,s,sp,cp);
      gl.uniform2f(this.center,x,y);
      gl.uniform3f(this.scale,2*scale/w,2*scale/h,1/(2*Math.hypot(this.scene.x,this.scene.y)+4));
      gl.drawArrays(gl.TRIANGLES,0,this.count);
    }else{
      this.drawCpu(w,h,pd,project);
    }
  }
  drawCpu(w,h,pd,project){
    const ctx=this.cpu,v=this.vertices,triangles=[];
    for(let i=0;i<v.length;i+=18){
      const a=project(v.subarray(i,i+3)),b=project(v.subarray(i+6,i+9)),c=project(v.subarray(i+12,i+15));
      triangles.push({a,b,c,depth:a[2]+b[2]+c[2],color:`rgb(${v[i+3]*255},${v[i+4]*255},${v[i+5]*255})`});
    }
    triangles.sort((a,b)=>a.depth-b.depth);
    ctx.setTransform(pd,0,0,pd,0,0);
    ctx.fillStyle="#102025";
    ctx.fillRect(0,0,w,h);
    for(const {a,b,c,color} of triangles){
      ctx.beginPath();
      ctx.moveTo(a[0],a[1]);
      ctx.lineTo(b[0],b[1]);
      ctx.lineTo(c[0],c[1]);
      ctx.closePath();
      ctx.fillStyle=ctx.strokeStyle=color;
      ctx.lineWidth=.8;
      ctx.fill();
      ctx.stroke();
    }
  }
}
