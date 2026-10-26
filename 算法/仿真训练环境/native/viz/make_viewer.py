"""Build a self-contained HTML viewer from a recorded native rollout JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>AGTC native rollout viewer</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0f1115; color:#e6e6e6; font:13px/1.5 Segoe UI, sans-serif; display:flex; height:100vh; }
  #left { flex:1; display:flex; flex-direction:column; min-width:0; }
  canvas { flex:1; width:100%; height:100%; display:block; }
  #bar { display:flex; gap:10px; align-items:center; padding:8px 12px; background:#161a22; border-top:1px solid #242a36; }
  #bar input[type=range] { flex:1; }
  button { background:#1f2633; color:#e6e6e6; border:1px solid #333c4d; border-radius:6px; padding:4px 12px; cursor:pointer; }
  #right { width:300px; background:#12161d; border-left:1px solid #242a36; padding:14px; box-sizing:border-box; overflow:auto; }
  h2 { font-size:14px; margin:0 0 10px; color:#7fd1ff; font-weight:600; }
  .row { display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px solid #1c222c; }
  .k { color:#8b95a7; }
  .v { font-variant-numeric: tabular-nums; }
  .act { display:flex; gap:6px; margin-top:6px; }
  .act div { flex:1; height:8px; background:#1c222c; border-radius:4px; position:relative; overflow:hidden; }
  .act i { position:absolute; left:50%; top:0; bottom:0; background:#4da3ff; }
  #note { color:#8b95a7; font-size:11px; margin-top:12px; line-height:1.5; }
  #status { font-weight:700; }
</style>
</head>
<body>
<div id="left">
  <canvas id="cv"></canvas>
  <div id="bar">
    <button id="play">暂停</button>
    <input id="slider" type="range" min="0" value="0" step="1">
    <span id="tlabel" class="v"></span>
  </div>
</div>
<div id="right">
  <h2>AGTC native rollout</h2>
  <div class="row"><span class="k">seed</span><span class="v" id="seed"></span></div>
  <div class="row"><span class="k">时间</span><span class="v" id="t"></span></div>
  <div class="row"><span class="k">离目标</span><span class="v" id="dist"></span></div>
  <div class="row"><span class="k">速度 (XY)</span><span class="v" id="speed"></span></div>
  <div class="row"><span class="k">角速度</span><span class="v" id="angspeed"></span></div>
  <div class="row"><span class="k">roll / pitch</span><span class="v" id="rp"></span></div>
  <div class="row"><span class="k">状态</span><span class="v" id="status"></span></div>
  <div style="margin-top:10px;color:#8b95a7">动作（左/右履带、云台偏航/俯仰）</div>
  <div class="act"><div><i id="a0"></i></div><div><i id="a1"></i></div><div><i id="a2"></i></div><div><i id="a3"></i></div></div>
  <div id="note"></div>
</div>
<script>
const DATA = __DATA__;
const frames = DATA.frames;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const slider = document.getElementById('slider');
slider.max = frames.length - 1;
let idx = 0, playing = true, last = performance.now();

const xs = frames.map(f => f.p[0]).concat([DATA.goal[0]]);
const ys = frames.map(f => f.p[1]).concat([DATA.goal[1]]);
const minX = Math.min(...xs), maxX = Math.max(...xs);
const minY = Math.min(...ys), maxY = Math.max(...ys);
const pad = 1.5;
const world = { cx:(minX+maxX)/2, cy:(minY+maxY)/2, w:(maxX-minX)+2*pad, h:(maxY-minY)+2*pad };

function resize() { cv.width = cv.clientWidth * devicePixelRatio; cv.height = cv.clientHeight * devicePixelRatio; }
window.addEventListener('resize', resize); resize();

function worldToScreen(x, y) {
  const s = Math.min(cv.width / world.w, cv.height / world.h);
  return [cv.width/2 + (x - world.cx) * s, cv.height/2 - (y - world.cy) * s, s];
}

function drawGrid() {
  const step = 2;
  ctx.strokeStyle = '#1c222c'; ctx.lineWidth = 1;
  for (let gx = Math.ceil((world.cx-world.w/2)/step)*step; gx <= world.cx+world.w/2; gx += step) {
    const a = worldToScreen(gx, world.cy - world.h/2), b = worldToScreen(gx, world.cy + world.h/2);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  for (let gy = Math.ceil((world.cy-world.h/2)/step)*step; gy <= world.cy+world.h/2; gy += step) {
    const a = worldToScreen(world.cx - world.w/2, gy), b = worldToScreen(world.cx + world.w/2, gy);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
}

function drawVehicle(f) {
  const [sx, sy, s] = worldToScreen(f.p[0], f.p[1]);
  const len = 0.8 * s, wid = 0.47 * s;
  ctx.save(); ctx.translate(sx, sy); ctx.rotate(-f.yaw);
  ctx.fillStyle = f.flipped ? '#ff5f5f' : '#4da3ff';
  ctx.strokeStyle = '#d8e6ff'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.roundRect(-len/2, -wid/2, len, wid, 3); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(len/2, 0); ctx.strokeStyle='#fff'; ctx.stroke();
  ctx.restore();
}

function draw() {
  ctx.clearRect(0,0,cv.width,cv.height);
  drawGrid();
  const goalS = worldToScreen(DATA.goal[0], DATA.goal[1]);
  ctx.strokeStyle = '#3dffa2'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(goalS[0], goalS[1], 0.4*goalS[2], 0, Math.PI*2); ctx.stroke();
  ctx.fillStyle = '#3dffa2'; ctx.font = '12px sans-serif'; ctx.fillText('target', goalS[0]+8, goalS[1]-8);

  ctx.beginPath();
  frames.forEach((f, i) => {
    const p = worldToScreen(f.p[0], f.p[1]);
    if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
  });
  ctx.strokeStyle = 'rgba(77,163,255,.35)'; ctx.lineWidth = 2; ctx.stroke();

  const f = frames[idx];
  drawVehicle(f);
  if (f.flipped) {
    ctx.fillStyle = 'rgba(255,60,60,.12)'; ctx.fillRect(0,0,cv.width,cv.height);
    ctx.fillStyle = '#ff6b6b'; ctx.font = 'bold 22px sans-serif';
    ctx.fillText('FLIPPED', cv.width/2 - 50, 40);
  }
}

function updatePanel() {
  const f = frames[idx];
  document.getElementById('seed').textContent = DATA.seed;
  document.getElementById('t').textContent = f.t.toFixed(2) + ' s';
  document.getElementById('dist').textContent = f.dist.toFixed(2) + ' m';
  document.getElementById('speed').textContent = f.speed.toFixed(3) + ' m/s';
  document.getElementById('angspeed').textContent = f.ang_speed.toFixed(3) + ' rad/s';
  document.getElementById('rp').textContent = (f.roll*57.2958).toFixed(1) + '° / ' + (f.pitch*57.2958).toFixed(1) + '°';
  const st = document.getElementById('status');
  st.textContent = f.flipped ? '翻车 (terminated)' : (f.success ? '成功' : '行驶中');
  st.style.color = f.flipped ? '#ff6b6b' : (f.success ? '#3dffa2' : '#e6e6e6');
  for (let i = 0; i < 4; i++) {
    const bar = document.getElementById('a'+i);
    const v = Math.max(-1, Math.min(1, f.a[i]));
    bar.style.left = v >= 0 ? '50%' : (50 + v*50) + '%';
    bar.style.width = Math.abs(v)*50 + '%';
    bar.style.background = v >= 0 ? '#4da3ff' : '#ff8b4d';
  }
  slider.value = idx;
  document.getElementById('tlabel').textContent = (idx+1) + ' / ' + frames.length;
}

function tick(now) {
  if (playing && now - last > 33) { idx = Math.min(idx + 1, frames.length - 1); last = now; }
  draw(); updatePanel();
  requestAnimationFrame(tick);
}
document.getElementById('play').onclick = () => { playing = !playing; document.getElementById('play').textContent = playing ? '暂停' : '播放'; };
slider.oninput = () => { idx = parseInt(slider.value); };
document.getElementById('note').textContent = DATA.note;
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main() -> int:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    payload = json.loads(src.read_text(encoding="utf-8"))
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    dst.write_text(html, encoding="utf-8")
    print(f"WROTE {dst} frames={len(payload['frames'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
