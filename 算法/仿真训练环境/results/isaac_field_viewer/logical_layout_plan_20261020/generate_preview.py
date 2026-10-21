#!/usr/bin/env python3
"""Build a self-contained static layout preview from the accepted frame snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SNAPSHOT_PATH = (
    ROOT / "results" / "isaac_field_viewer" / "direct_tracks_20261020" / "frame_after.json"
)
OUTPUT_PATH = HERE / "index.html"
MANIFEST_PATH = HERE / "source_manifest.json"


def main() -> None:
    raw = SNAPSHOT_PATH.read_bytes()
    snapshot = json.loads(raw.decode("utf-8"))
    if snapshot.get("sequence") != 149 or len(snapshot.get("nodes", [])) != 57:
        raise SystemExit("Expected the accepted frame-149 snapshot with 57 nodes.")

    sha256 = hashlib.sha256(raw).hexdigest()
    manifest = {
        "source": "../direct_tracks_20261020/frame_after.json",
        "sha256": sha256,
        "bytes": len(raw),
        "sequence": snapshot["sequence"],
        "sim_time": snapshot["sim_time"],
        "output_time": snapshot["output_time"],
        "node_count": len(snapshot["nodes"]),
        "color_name": snapshot["legend"]["name"],
        "color_mode": snapshot["legend"]["mode"],
        "color_minimum": snapshot["legend"]["minimum"],
        "color_zero": snapshot["legend"]["zero"],
        "color_maximum": snapshot["legend"]["maximum"],
        "note": "Images and node metadata are embedded unchanged in index.html.",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    embedded = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    embedded = embedded.replace("</", "<\\/")
    html = HTML.replace("__SOURCE_SHA256__", sha256).replace(
        "__SNAPSHOT_JSON__", embedded
    )
    OUTPUT_PATH.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT_PATH}")
    print(f"source sha256 {sha256}")
    print(f"embedded nodes {len(snapshot['nodes'])}; sequence {snapshot['sequence']}")


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="source-snapshot-sha256" content="__SOURCE_SHA256__">
  <title>布局方案预览 · 记录帧149 · 非实时</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172331;
      --muted: #596a7c;
      --line: #d9e1e9;
      --panel: #ffffff;
      --page: #f2f5f8;
      --teal: #126b73;
      --teal-soft: #e7f4f3;
      --blue-soft: #eef4fb;
      --gold-soft: #fff5dd;
      --shadow: 0 5px 18px rgba(21, 40, 61, .07);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font: 15px/1.5 "Microsoft YaHei UI", "Segoe UI", sans-serif;
    }
    header {
      padding: 22px clamp(16px, 3vw, 42px) 17px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }
    .heading-row {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    h1 { margin: 0; font-size: 24px; line-height: 1.25; letter-spacing: .01em; }
    .stamp {
      border: 1px solid #edc76b;
      background: #fff6df;
      color: #71500b;
      border-radius: 999px;
      padding: 4px 10px;
      font-weight: 700;
      font-size: 13px;
      white-space: nowrap;
    }
    .intro { margin: 9px 0 0; max-width: 1200px; color: var(--muted); }
    .facts {
      display: flex;
      gap: 9px 18px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 14px;
      color: #34475a;
      font-size: 13px;
    }
    .fact {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 26px;
    }
    .legend-bar {
      width: 170px;
      height: 11px;
      border: 1px solid rgba(23, 35, 49, .2);
      border-radius: 999px;
    }
    .legend-labels { display: flex; width: 170px; justify-content: space-between; font-variant-numeric: tabular-nums; }
    main { padding: 16px clamp(12px, 2.5vw, 34px) 34px; }
    .tabs {
      display: flex;
      gap: 7px;
      overflow-x: auto;
      padding: 2px 1px 12px;
      scrollbar-width: thin;
    }
    .tab {
      flex: 0 0 auto;
      border: 1px solid #ccd7e1;
      border-radius: 8px;
      background: #fff;
      padding: 8px 13px;
      color: #293d50;
      font: inherit;
      font-size: 14px;
      font-weight: 650;
      cursor: pointer;
    }
    .tab:hover { border-color: #77aeb1; background: #f8fdfc; }
    .tab[aria-selected="true"] {
      color: #fff;
      background: #145c68;
      border-color: #145c68;
      box-shadow: 0 2px 8px rgba(20, 92, 104, .18);
    }
    .view-heading {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 5px 15px;
      margin: 3px 0 10px;
    }
    h2 { font-size: 18px; margin: 0; }
    .view-count { color: var(--muted); font-size: 13px; }
    .notice {
      margin: 0 0 14px;
      padding: 9px 12px;
      border: 1px solid #d6e4eb;
      border-left: 4px solid #5594a2;
      border-radius: 7px;
      background: #f8fbfd;
      color: #344b5b;
      font-size: 13px;
    }
    .canvas {
      overflow-x: auto;
      padding: 2px 1px 12px;
      scrollbar-width: thin;
    }
    .stages {
      display: grid;
      grid-template-columns: repeat(6, minmax(170px, 1fr));
      gap: 16px;
      min-width: 1120px;
      align-items: stretch;
    }
    .stage {
      position: relative;
      min-width: 0;
      padding: 11px;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: rgba(255,255,255,.68);
    }
    .stage:not(:last-child)::after {
      content: "›";
      position: absolute;
      z-index: 2;
      top: 11px;
      right: -14px;
      color: #76909c;
      font-size: 24px;
      font-weight: 700;
      line-height: 1;
    }
    .stage-head {
      min-height: 48px;
      margin-bottom: 9px;
      padding: 0 0 8px;
      border-bottom: 1px solid #e2e8ee;
    }
    .stage-title { font-size: 14px; font-weight: 750; }
    .stage-subtitle { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .stage-nodes { display: grid; gap: 7px; align-content: start; }
    .node-card {
      display: block;
      width: 100%;
      min-width: 0;
      padding: 8px;
      text-align: left;
      color: var(--ink);
      background: var(--panel);
      border: 1px solid #dbe3ea;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(21, 40, 61, .035);
      font: inherit;
      cursor: pointer;
      transition: border-color .12s ease, box-shadow .12s ease, transform .12s ease;
    }
    .node-card:hover { border-color: #6ba1a5; box-shadow: var(--shadow); transform: translateY(-1px); }
    .node-card.selected { border: 2px solid var(--teal); padding: 7px; box-shadow: 0 0 0 2px rgba(18, 107, 115, .1); }
    .node-card:focus-visible, .tab:focus-visible { outline: 3px solid #96c7c9; outline-offset: 2px; }
    .node-title { display: block; font-size: 13px; line-height: 1.35; font-weight: 700; overflow-wrap: anywhere; }
    .node-meta { display: flex; flex-wrap: wrap; gap: 4px 7px; margin-top: 5px; color: #53687a; font-size: 12px; }
    .node-tag { border-radius: 4px; padding: 1px 5px; background: #f0f4f7; }
    .node-image {
      display: block;
      width: 100%;
      height: 57px;
      margin: 0 0 6px;
      object-fit: contain;
      image-rendering: auto;
      background: #fafbfd;
      border: 1px solid #edf0f3;
      border-radius: 4px;
    }
    .node-card.compact { padding: 7px 8px; }
    .node-card.compact .node-meta { margin-top: 3px; }
    .input-list .node-card { background: #fbfdff; }
    .action-fields {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
      margin-top: 7px;
    }
    .action-field {
      min-width: 0;
      padding: 4px 5px;
      border: 1px solid #dbe7eb;
      border-radius: 5px;
      background: #f2f8f8;
      color: #234850;
      font-size: 12px;
      line-height: 1.3;
      overflow-wrap: anywhere;
    }
    .detail {
      display: grid;
      grid-template-columns: minmax(190px, 280px) minmax(260px, 1fr);
      gap: 14px;
      margin-top: 8px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: #fff;
      box-shadow: var(--shadow);
    }
    .detail-image-wrap {
      min-height: 168px;
      display: grid;
      place-items: center;
      border: 1px solid #e6ebef;
      border-radius: 7px;
      background: #fafbfd;
      padding: 8px;
    }
    .detail-image { display: block; max-width: 100%; max-height: 240px; object-fit: contain; }
    .detail-placeholder { color: #778695; font-size: 13px; }
    .detail-content { min-width: 0; }
    .detail-title-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 10px; }
    .detail-title { font-size: 18px; font-weight: 750; margin: 0; }
    .detail-key { color: #647789; font-size: 12px; overflow-wrap: anywhere; }
    .detail-facts { display: flex; flex-wrap: wrap; gap: 6px; margin: 9px 0; }
    .detail-fact {
      padding: 3px 7px;
      border-radius: 5px;
      background: var(--blue-soft);
      color: #294456;
      font-size: 13px;
    }
    .detail-label { margin: 9px 0 2px; color: #627487; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .035em; }
    .detail-description { margin: 0; font-size: 14px; }
    .parents { display: flex; flex-wrap: wrap; gap: 5px; }
    .parents code {
      border: 1px solid #e1e7ec;
      border-radius: 4px;
      padding: 2px 5px;
      background: #f6f8fa;
      color: #394b5c;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .empty-parents { color: #718294; font-size: 13px; }
    .branch-layout {
      display: grid;
      grid-template-columns: minmax(420px, 2.2fr) repeat(4, minmax(168px, .9fr));
      gap: 14px;
      min-width: 1180px;
      align-items: stretch;
    }
    .branch-stage .stage-nodes { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .branch-stage .node-image { height: 48px; }
    .flow-row {
      display: grid;
      grid-template-columns: 1fr 1.2fr 1fr;
      gap: 14px;
      min-width: 760px;
      align-items: stretch;
    }
    .flow-row.two { grid-template-columns: 1fr 1fr; }
    .state-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
    .gate-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
    .memory-layout {
      display: grid;
      grid-template-columns: minmax(250px, 1fr) minmax(260px, 1fr) minmax(250px, 1fr) minmax(165px, .7fr);
      gap: 14px;
      min-width: 1010px;
      align-items: stretch;
    }
    .critic-banner {
      margin: 0 0 12px;
      padding: 10px 13px;
      border: 1px solid #e6c56e;
      border-left: 5px solid #b17a09;
      border-radius: 8px;
      background: var(--gold-soft);
      color: #624805;
      font-weight: 750;
    }
    .split-layout {
      display: grid;
      grid-template-columns: minmax(500px, 1.7fr) minmax(260px, .8fr);
      gap: 14px;
      min-width: 800px;
      align-items: start;
    }
    .lane {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(255,255,255,.75);
    }
    .lane h3 { margin: 0 0 4px; font-size: 15px; }
    .lane-note { margin: 0 0 9px; color: var(--muted); font-size: 13px; }
    .lane-nodes { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 8px; }
    .lane-nodes.vertical { grid-template-columns: 1fr; }
    .path-label {
      display: flex;
      align-items: center;
      gap: 7px;
      margin: 8px 0;
      color: #758696;
      font-size: 13px;
    }
    .path-label::after { content: ""; height: 1px; flex: 1; background: #dfe6ec; }
    .footer-note { margin: 13px 2px 0; color: #667889; font-size: 12px; }
    @media (max-width: 1050px) {
      header { padding-top: 17px; }
      h1 { font-size: 21px; }
      .stages, .branch-layout, .memory-layout, .flow-row, .flow-row.two, .split-layout {
        display: flex;
        min-width: 0;
        flex-direction: column;
      }
      .stage:not(:last-child)::after {
        content: "⌄";
        top: auto;
        right: 15px;
        bottom: -19px;
        font-size: 19px;
      }
      .stage { margin-bottom: 5px; }
      .stage-head { min-height: 0; }
      .branch-stage .stage-nodes { grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
      .detail { grid-template-columns: 1fr; }
      .detail-image-wrap { min-height: 120px; }
      .detail-image { max-height: 190px; }
    }
    @media (max-width: 560px) {
      .facts { align-items: flex-start; flex-direction: column; gap: 6px; }
      .legend-labels { width: 150px; }
      .legend-bar { width: 150px; }
      .action-fields { grid-template-columns: 1fr; }
      .gate-grid, .state-pair { grid-template-columns: 1fr 1fr; }
      .node-image { height: 48px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="heading-row">
      <h1>布局方案预览</h1>
      <span class="stamp">记录帧149 · 非实时</span>
    </div>
    <p class="intro">基于已验收的真实 snapshot 重排阅读层次。页面只展示固定记录数据，不连接仿真器，也不发送运行或暂停请求。</p>
    <div class="facts">
      <span class="fact">来源帧 <strong>149</strong></span>
      <span class="fact">输入时刻 <strong>7.35 s</strong></span>
      <span class="fact">步后时刻 <strong>7.40 s</strong></span>
      <span class="fact">源图 <strong>57 个真实节点</strong></span>
      <span class="fact">当前视图 <strong id="visible-count">—</strong></span>
      <span class="fact">Turbo 色标 <strong id="legend-range">—</strong></span>
      <span class="fact">
        <span id="legend-bar" class="legend-bar" aria-label="原始 Turbo 色标"></span>
        <span class="legend-labels"><span id="legend-min">−</span><span>0</span><span id="legend-max">+</span></span>
      </span>
    </div>
  </header>
  <main>
    <nav class="tabs" role="tablist" aria-label="网络布局视图">
      <button class="tab" type="button" role="tab" data-view="main" aria-selected="true">主链</button>
      <button class="tab" type="button" role="tab" data-view="lidar" aria-selected="false">雷达</button>
      <button class="tab" type="button" role="tab" data-view="depth" aria-selected="false">深度</button>
      <button class="tab" type="button" role="tab" data-view="detections" aria-selected="false">检测</button>
      <button class="tab" type="button" role="tab" data-view="state" aria-selected="false">状态</button>
      <button class="tab" type="button" role="tab" data-view="memory" aria-selected="false">驾驶记忆</button>
      <button class="tab" type="button" role="tab" data-view="distribution" aria-selected="false">动作分布</button>
      <button class="tab" type="button" role="tab" data-view="critic" aria-selected="false">价值诊断</button>
    </nav>
    <div class="view-heading">
      <h2 id="view-title">主链</h2>
      <span class="view-count" id="view-count"></span>
    </div>
    <p class="notice" id="view-note"></p>
    <section class="canvas" id="canvas" aria-live="polite"></section>
    <section class="detail" aria-label="节点详情">
      <div class="detail-image-wrap" id="detail-image-wrap">
        <span class="detail-placeholder">选择任一节点查看记录帧中的真实伪彩图</span>
      </div>
      <div class="detail-content">
        <div class="detail-title-row">
          <h2 class="detail-title" id="detail-title">节点详情</h2>
          <code class="detail-key" id="detail-key"></code>
        </div>
        <div class="detail-facts">
          <span class="detail-fact" id="detail-shape">shape —</span>
          <span class="detail-fact" id="detail-source">来源 —</span>
        </div>
        <div class="detail-label">说明</div>
        <p class="detail-description" id="detail-description">单击节点卡片只更新此详情区域；当前视图保持不变。</p>
        <div class="detail-label">直接父节点</div>
        <div class="parents" id="detail-parents"><span class="empty-parents">无父节点</span></div>
      </div>
    </section>
    <p class="footer-note">截图、张量和文字均来自原始 frame_after.json；图片不重绘、不重新定标，统一 Turbo 范围沿用源帧。缩略图只改变显示尺寸。SHA-256: <code>__SOURCE_SHA256__</code></p>
  </main>
  <script id="snapshot-data" type="application/json">__SNAPSHOT_JSON__</script>
  <script>
    "use strict";
    const snapshot = JSON.parse(document.getElementById("snapshot-data").textContent);
    const nodes = snapshot.nodes;
    const byKey = new Map(nodes.map((node) => [node.key, node]));
    const actionFields = snapshot.detail && snapshot.detail.key === "action.applied"
      ? snapshot.detail.fields || []
      : [];
    const canvas = document.getElementById("canvas");
    const visibleCount = document.getElementById("visible-count");
    let selectedKey = "action.applied";
    let shownKeys = new Set();

    const fmt = (value) => Number(value).toFixed(4).replace(/\.?0+$/, "");
    const shapeText = (shape) => "[" + (shape || []).join(" × ") + "]";
    const colorStops = snapshot.legend.colors.map((rgb, i) => {
      const pct = Math.round((i / (snapshot.legend.colors.length - 1)) * 100);
      return "rgb(" + rgb.join(",") + ") " + pct + "%";
    });
    document.getElementById("legend-bar").style.background =
      "linear-gradient(90deg, " + colorStops.join(", ") + ")";
    document.getElementById("legend-min").textContent = fmt(snapshot.legend.minimum);
    document.getElementById("legend-max").textContent = "+" + fmt(snapshot.legend.maximum);
    document.getElementById("legend-range").textContent =
      "[" + fmt(snapshot.legend.minimum) + ", +" + fmt(snapshot.legend.maximum) + "]";

    function el(tag, className, text) {
      const item = document.createElement(tag);
      if (className) item.className = className;
      if (text !== undefined) item.textContent = text;
      return item;
    }

    function card(key, options) {
      const node = byKey.get(key);
      if (!node) throw new Error("Snapshot node is missing: " + key);
      shownKeys.add(key);
      const opts = options || {};
      const button = el("button", "node-card" + (opts.compact ? " compact" : ""));
      button.type = "button";
      button.dataset.nodeKey = key;
      button.setAttribute("aria-label", node.title + "，shape " + shapeText(node.shape));
      if (opts.image !== false && node.image) {
        const image = el("img", "node-image");
        image.src = node.image;
        image.alt = node.title + "：源帧伪彩";
        image.decoding = "async";
        button.appendChild(image);
      }
      button.appendChild(el("span", "node-title", opts.label || node.title));
      const meta = el("span", "node-meta");
      meta.appendChild(el("span", "node-tag", "shape " + shapeText(node.shape)));
      button.appendChild(meta);
      if (key === "action.applied" && actionFields.length) {
        const fields = el("span", "action-fields");
        actionFields.forEach((field, index) => {
          fields.appendChild(el("span", "action-field", (index + 1) + " · " + field));
        });
        button.appendChild(fields);
      }
      return button;
    }

    function stage(title, subtitle, keys, options) {
      const opts = options || {};
      const box = el("section", "stage" + (opts.className ? " " + opts.className : ""));
      const head = el("div", "stage-head");
      head.appendChild(el("div", "stage-title", title));
      if (subtitle) head.appendChild(el("div", "stage-subtitle", subtitle));
      box.appendChild(head);
      const list = el("div", "stage-nodes" + (opts.listClass ? " " + opts.listClass : ""));
      keys.forEach((key) => list.appendChild(card(key, opts.cardOptions)));
      box.appendChild(list);
      return box;
    }

    function group(title, subtitle, keys, options) {
      const box = el("section", "stage");
      const head = el("div", "stage-head");
      head.appendChild(el("div", "stage-title", title));
      if (subtitle) head.appendChild(el("div", "stage-subtitle", subtitle));
      box.appendChild(head);
      const list = el("div", "stage-nodes");
      keys.forEach((key) => list.appendChild(card(key, options)));
      box.appendChild(list);
      return box;
    }

    function setHeader(title, note, countText) {
      document.getElementById("view-title").textContent = title;
      document.getElementById("view-note").textContent = note;
      document.getElementById("view-count").textContent = countText || "";
    }

    function renderMain() {
      setHeader(
        "主链",
        "六个阶段先说明整体数据流；输入与编码各保留四个代表节点。单击卡片只更新下方详情，不展开完整图。",
        "按六阶段阅读"
      );
      const flow = el("div", "stages");
      flow.appendChild(stage("① 输入", "4 路小列表", [
        "input.lidar", "input.depth", "input.detections", "input.state"
      ], { className: "input-list", cardOptions: { compact: true, image: false } }));
      flow.appendChild(stage("② 支路编码", "32 / 64 / 32 / 32 维", [
        "features_extractor.laser.7",
        "features_extractor.depth.7",
        "detections.pooled",
        "features_extractor.proprioception.1"
      ], { cardOptions: { compact: true } }));
      flow.appendChild(stage("③ 融合", "四路拼接 · 160 维", [
        "features_extractor"
      ]));
      flow.appendChild(stage("④ 驾驶记忆", "Actor LSTM · 128", [
        "lstm_actor"
      ]));
      flow.appendChild(stage("⑤ 决策 MLP", "128 → 64 → Tanh", [
        "mlp_extractor.policy_net.0", "mlp_extractor.policy_net.1"
      ], { cardOptions: { compact: true } }));
      flow.appendChild(stage("⑥ 四维动作", "各维语义列于卡片", [
        "action.applied"
      ]));
      canvas.appendChild(flow);
    }

    const branchConfig = {
      lidar: {
        title: "雷达",
        branch: "lidar",
        note: "只保留雷达真实编码节点及其驾驶下游。按实际数据流顺序排列，详情中的父节点沿用源图；融合处保留四路父节点元数据。"
      },
      depth: {
        title: "深度",
        branch: "depth",
        note: "只保留深度真实编码节点及其驾驶下游。按实际数据流顺序排列，详情中的父节点沿用源图；融合处保留四路父节点元数据。"
      },
      detections: {
        title: "检测",
        branch: "detections",
        note: "保留检测输入、逐框编码、有效性掩码与池化结果；池化来源索引也按源图作为检测支路节点显示。"
      },
      state: {
        title: "状态",
        branch: "proprioception",
        note: "只保留车辆与目标状态的真实编码节点及其驾驶下游。按实际数据流顺序排列，详情中的父节点沿用源图。"
      }
    };
    const representativeNode = {
      main: "action.applied",
      lidar: "features_extractor.laser.7",
      depth: "features_extractor.depth.7",
      detections: "detections.pooled",
      state: "features_extractor.proprioception.1",
      memory: "lstm_actor",
      distribution: "action.applied",
      critic: "value_net"
    };

    function renderBranch(view) {
      const cfg = branchConfig[view];
      const branchNodes = nodes
        .filter((node) => node.branch === cfg.branch)
        .sort((a, b) => a.column - b.column || nodes.indexOf(a) - nodes.indexOf(b));
      setHeader(cfg.title + "支路", cfg.note, "支路与驾驶下游");
      const flow = el("div", "branch-layout");
      const first = el("section", "stage branch-stage");
      const head = el("div", "stage-head");
      head.appendChild(el("div", "stage-title", "支路输入与编码"));
      head.appendChild(el("div", "stage-subtitle", "按实际数据流顺序 · " + branchNodes.length + " 个真实节点"));
      first.appendChild(head);
      const list = el("div", "stage-nodes");
      branchNodes.forEach((node) => list.appendChild(card(node.key, { compact: true })));
      first.appendChild(list);
      flow.appendChild(first);
      flow.appendChild(group("融合", "四路特征 · 160 维", ["features_extractor"], { compact: true }));
      flow.appendChild(group("驾驶记忆", "门控细节见「驾驶记忆」", ["lstm_actor"], { compact: true }));
      flow.appendChild(group("动作头", "MLP · 64", [
        "mlp_extractor.policy_net.0", "mlp_extractor.policy_net.1", "action_net"
      ], { compact: true }));
      flow.appendChild(group("执行输出", "四维动作", ["action.applied"], { compact: true }));
      canvas.appendChild(flow);
    }

    function statePair(keys) {
      const wrapper = el("div", "state-pair");
      keys.forEach((key) => wrapper.appendChild(card(key, { compact: true })));
      return wrapper;
    }

    function renderMemory(kind) {
      const critic = kind === "critic";
      const prefix = critic ? "lstm_critic" : "lstm_actor";
      const label = critic ? "价值记忆" : "驾驶记忆";
      const gates = [prefix + ".i", prefix + ".f", prefix + ".g", prefix + ".o"];
      const outputs = [prefix + ".c", prefix + ".h"];
      setHeader(
        critic ? "价值记忆结构" : "驾驶记忆结构",
        critic
          ? "本视图只呈现 critic 价值旁路；门值来自权重重建并用真实 h₁/c₁ 核验。"
          : "h₀/c₀ 作为输入状态成组，i/f/g/o 门排成 2×2 网格，c₁/h₁ 作为输出状态成组。门值来自权重重建并用真实 h₁/c₁ 核验。",
        critic ? "独立价值诊断支路" : "Actor LSTM · 128"
      );
      const flow = el("div", "memory-layout");
      const input = el("section", "stage");
      const inputHead = el("div", "stage-head");
      inputHead.appendChild(el("div", "stage-title", "本步输入"));
      inputHead.appendChild(el("div", "stage-subtitle", "融合特征 + 上一步状态"));
      input.appendChild(inputHead);
      const inputList = el("div", "stage-nodes");
      inputList.appendChild(card("features_extractor", { compact: true }));
      inputList.appendChild(statePair([prefix + ".h0", prefix + ".c0"]));
      input.appendChild(inputList);
      flow.appendChild(input);

      const gateBox = el("section", "stage");
      const gateHead = el("div", "stage-head");
      gateHead.appendChild(el("div", "stage-title", "LSTM 门控"));
      gateHead.appendChild(el("div", "stage-subtitle", "i · f · g · o"));
      gateBox.appendChild(gateHead);
      const gateGrid = el("div", "gate-grid");
      gates.forEach((key) => gateGrid.appendChild(card(key, { compact: true })));
      gateBox.appendChild(gateGrid);
      flow.appendChild(gateBox);

      const output = el("section", "stage");
      const outputHead = el("div", "stage-head");
      outputHead.appendChild(el("div", "stage-title", "输出状态"));
      outputHead.appendChild(el("div", "stage-subtitle", "本步更新后的 c₁ / h₁"));
      output.appendChild(outputHead);
      output.appendChild(statePair(outputs));
      flow.appendChild(output);

      flow.appendChild(group(label + " LSTM", "隐藏维度 128", [prefix], { compact: true }));
      canvas.appendChild(flow);
    }

    function renderDistribution() {
      setHeader(
        "动作分布",
        "均值 μ 与全局 log_std / σ 分开展示。当前记录是确定性驾驶输出，action.applied 的直接父节点为动作均值分支。",
        "Actor 输出与分布参数"
      );
      const split = el("div", "split-layout");
      const mean = el("section", "lane");
      mean.appendChild(el("h3", "", "驾驶均值 μ"));
      mean.appendChild(el("p", "lane-note", "策略网络产生四维均值，随后记录实际应用动作。"));
      const meanNodes = el("div", "lane-nodes");
      [
        "lstm_actor",
        "mlp_extractor.policy_net.0",
        "mlp_extractor.policy_net.1",
        "action_net",
        "action.applied"
      ].forEach((key) => meanNodes.appendChild(card(key, { compact: true })));
      mean.appendChild(meanNodes);
      split.appendChild(mean);

      const spread = el("section", "lane");
      spread.appendChild(el("h3", "", "分布尺度"));
      spread.appendChild(el("p", "lane-note", "log_std 与 σ 使用源图节点；它们不改写本记录帧的确定性动作。"));
      const spreadNodes = el("div", "lane-nodes vertical");
      ["action.log_std", "action.std"].forEach((key) => spreadNodes.appendChild(card(key, { compact: true })));
      spread.appendChild(spreadNodes);
      split.appendChild(spread);
      canvas.appendChild(split);
    }

    function renderCritic() {
      const banner = el("p", "critic-banner", "价值诊断支路：用于估计状态价值 V(s)，不控制车辆动作。");
      canvas.appendChild(banner);
      renderMemory("critic");
      const mlp = el("div", "flow-row two");
      mlp.appendChild(group("价值头", "128 → 64", ["mlp_extractor.value_net.0", "mlp_extractor.value_net.1"], { compact: true }));
      mlp.appendChild(group("价值输出", "标量 V(s)", ["value_net"], { compact: true }));
      canvas.appendChild(mlp);
    }

    function updateCount() {
      visibleCount.textContent = shownKeys.size + " / " + nodes.length + " 节点";
    }

    function render(view) {
      shownKeys = new Set();
      canvas.replaceChildren();
      if (view === "main") renderMain();
      else if (branchConfig[view]) renderBranch(view);
      else if (view === "memory") renderMemory("actor");
      else if (view === "distribution") renderDistribution();
      else if (view === "critic") renderCritic();
      if (!shownKeys.has(selectedKey)) selectedKey = representativeNode[view];
      updateCount();
      document.querySelectorAll(".node-card").forEach((button) => {
        button.classList.toggle("selected", button.dataset.nodeKey === selectedKey);
      });
      renderDetail(selectedKey);
    }

    function renderDetail(key) {
      const node = byKey.get(key);
      if (!node) return;
      selectedKey = key;
      document.getElementById("detail-title").textContent = node.title;
      document.getElementById("detail-key").textContent = node.key;
      document.getElementById("detail-shape").textContent = "shape " + shapeText(node.shape);
      document.getElementById("detail-source").textContent = "来源 " + node.source;
      document.getElementById("detail-description").textContent = node.description || node.summary || "";
      const parentBox = document.getElementById("detail-parents");
      parentBox.replaceChildren();
      if (node.parents && node.parents.length) {
        node.parents.forEach((parent) => parentBox.appendChild(el("code", "", parent)));
      } else {
        parentBox.appendChild(el("span", "empty-parents", "无父节点"));
      }
      const imageBox = document.getElementById("detail-image-wrap");
      imageBox.replaceChildren();
      if (node.image) {
        const image = el("img", "detail-image");
        image.src = node.image;
        image.alt = node.title + "：记录帧中的真实伪彩图";
        image.decoding = "async";
        imageBox.appendChild(image);
      } else {
        imageBox.appendChild(el("span", "detail-placeholder", "此节点没有图像数据"));
      }
      document.querySelectorAll(".node-card").forEach((button) => {
        button.classList.toggle("selected", button.dataset.nodeKey === selectedKey);
      });
    }

    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((item) => {
          item.setAttribute("aria-selected", String(item === tab));
        });
        render(tab.dataset.view);
      });
    });
    canvas.addEventListener("click", (event) => {
      const button = event.target.closest("[data-node-key]");
      if (button) renderDetail(button.dataset.nodeKey);
    });

    render("main");
    renderDetail("action.applied");
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
