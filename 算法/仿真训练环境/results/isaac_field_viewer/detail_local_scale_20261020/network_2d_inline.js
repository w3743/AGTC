
  (() => {
    "use strict";

    const POLL_MS = 250;
    const SVG_NS = "http://www.w3.org/2000/svg";
    const BRANCH_LABELS = {
      lidar: "雷达", depth: "深度", detections: "检测", state: "状态", feedback: "负载与云台反馈", fusion: "融合",
      actor: "动作", actor_memory: "驾驶记忆", critic: "价值诊断", critic_memory: "价值诊断",
      output: "动作", value: "价值诊断"
    };
    const REGION_BRANCHES = {
      encoding: new Set(["lidar", "depth", "detections", "state", "feedback"]),
      fusion: new Set(["fusion"]),
      memory: new Set(["actor_memory", "critic_memory"]),
      output: new Set(["actor", "output"]),
      critic: new Set(["critic", "critic_memory", "value"])
    };
    const CARD_W = 170;
    const CARD_H = 110;
    const SENSOR_BRANCHES = new Set(["lidar", "depth", "detections", "state", "feedback"]);
    const els = {
      svg: document.getElementById("network-svg"),
      partitions: document.getElementById("partition-layer"),
      labels: document.getElementById("region-label-layer"),
      edgeLayer: document.getElementById("edge-layer"),
      nodeLayer: document.getElementById("node-layer"),
      empty: document.getElementById("empty-state"),
      connection: document.getElementById("connection-status"),
      frame: document.getElementById("frame-value"),
      inputTime: document.getElementById("input-time-value"),
      outputTime: document.getElementById("output-time-value"),
      runtime: document.getElementById("runtime-value"),
      training: document.getElementById("training-value"),
      mode: document.getElementById("mode-value"),
      critic: document.getElementById("critic-value"),
      graphStatus: document.getElementById("graph-status"),
      message: document.getElementById("message-value"),
      scaleReadout: document.getElementById("scale-readout"),
      scaleValue: document.getElementById("scale-value"),
      freeze: document.getElementById("freeze-control"),
      detailPanel: document.getElementById("details-panel"),
      detailToggle: document.getElementById("detail-toggle"),
      detailCollapse: document.getElementById("detail-collapse-button"),
      detailTitle: document.getElementById("detail-title"),
      detailPlaceholder: document.getElementById("detail-placeholder"),
      detailContent: document.getElementById("detail-content"),
      detailImage: document.getElementById("detail-image"),
      detailImageEmpty: document.getElementById("detail-image-empty"),
      detailLocalScale: document.getElementById("detail-local-scale"),
      detailColorKey: document.getElementById("detail-color-key"),
      detailScaleReadout: document.getElementById("detail-scale-readout"),
      tensor: document.getElementById("tensor-select"),
      index: document.getElementById("index-input"),
      fieldList: document.getElementById("field-list")
    };
    const scaleReadoutBaseTitle = els.scaleReadout.title;

    let selectedKey = "features_extractor";
    let tensorName = "激活值";
    let elementIndex = 0;
    let frozen = false;
    let fetching = false;
    let controlBusy = false;
    let requestTimer = 0;
    let lastFrame = null;
    let renderedKeys = new Map();
    let graphEdgeElements = [];
    let graphCrossingElements = [];
    let topologySignature = "";
    let currentBounds = null;
    let hasInitialFit = false;
    let viewBox = { x: 0, y: 0, width: 1000, height: 700 };
    let pan = null;
    let suppressClickUntil = 0;
    let detailVisible = true;
    let hoveredKey = null;
    let parameterOptions = [];
    let layerOptionsSignature = "";
    let legendSignature = "";

    const fmt = (value, digits = 3) => {
      if (value === null || value === undefined || value === "") return "—";
      if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(digits) : String(value);
      if (Array.isArray(value)) return "[" + value.map(item => fmt(item, digits)).join(", ") + "]";
      if (typeof value === "object") {
        try { return JSON.stringify(value); } catch (_) { return String(value); }
      }
      return String(value);
    };
    const legendNumber = value => typeof value === "number" && Number.isFinite(value) ? Number(Math.abs(value).toPrecision(6)).toString() : "—";
    const signedLegendNumber = value => {
      if (typeof value !== "number" || !Number.isFinite(value)) return "—";
      return value < 0 ? "−" + legendNumber(value) : (value > 0 ? "+" + legendNumber(value) : "0");
    };
    const asText = value => value === null || value === undefined ? "" : String(value);
    const shapeText = value => Array.isArray(value) ? "(" + value.join(" × ") + ")" : asText(value || "形状待定");
    const safeImageUri = value => typeof value === "string" && /^data:image\/png;base64,[A-Za-z0-9+/=\s]+$/.test(value) ? value : "";
    const clearChildren = element => { while (element.firstChild) element.removeChild(element.firstChild); };
    const svgElement = (name, attrs = {}) => {
      const node = document.createElementNS(SVG_NS, name);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
      return node;
    };
    const setText = (id, value) => { document.getElementById(id).textContent = fmt(value); };
    function setViewBox(next) {
      viewBox = { x: next.x, y: next.y, width: Math.max(1, next.width), height: Math.max(1, next.height) };
      els.svg.setAttribute("viewBox", [viewBox.x, viewBox.y, viewBox.width, viewBox.height].map(v => Number(v.toFixed(3))).join(" "));
      updateCrossingGeometry();
    }

    function viewportSize() {
      const rect = els.svg.getBoundingClientRect();
      return { width: Math.max(1, rect.width), height: Math.max(1, rect.height), rect };
    }

    function fitBounds(bounds, padding = 55) {
      if (!bounds) return;
      const { width: pxW, height: pxH } = viewportSize();
      let w = Math.max(1, bounds.maxX - bounds.minX + padding * 2);
      let h = Math.max(1, bounds.maxY - bounds.minY + padding * 2);
      const aspect = pxW / pxH;
      if (w / h < aspect) w = h * aspect;
      else h = w / aspect;
      const cx = (bounds.minX + bounds.maxX) / 2;
      const cy = (bounds.minY + bounds.maxY) / 2;
      setViewBox({ x: cx - w / 2, y: cy - h / 2, width: w, height: h });
    }

    function fitAll() {
      if (currentBounds) {
        fitBounds(currentBounds);
        hasInitialFit = true;
      }
    }

    function focusRegion(regionName) {
      const branches = REGION_BRANCHES[regionName];
      if (regionName !== "input" && !branches) return;
      const selected = [...renderedKeys.values()].filter(record => {
        if (!record.position) return false;
        if (regionName === "input") return record.node.key.startsWith("input.");
        if (regionName === "encoding") return !record.node.key.startsWith("input.") && branches.has(record.node.branch);
        return branches.has(record.node.branch);
      });
      if (!selected.length) return;
      const bounds = {
        minX: Math.min(...selected.map(record => record.position.x)) - 16,
        minY: Math.min(...selected.map(record => record.position.y)) - 30,
        maxX: Math.max(...selected.map(record => record.position.x + CARD_W)) + 16,
        maxY: Math.max(...selected.map(record => record.position.y + CARD_H)) + 16
      };
      document.querySelectorAll(".region-link[data-region]").forEach(button => {
        button.setAttribute("aria-pressed", String(button.dataset.region === regionName));
      });
      fitBounds(bounds, 58);
    }

    function edgeBranch(source, target, branchByKey) {
      const sourceBranch = branchByKey.get(source) || "";
      const targetBranch = branchByKey.get(target) || "";
      if (SENSOR_BRANCHES.has(sourceBranch)) return sourceBranch;
      if (SENSOR_BRANCHES.has(targetBranch)) return targetBranch;
      if ([sourceBranch, targetBranch].some(branch => branch === "critic" || branch === "critic_memory" || branch === "value")) return "critic";
      if ([sourceBranch, targetBranch].some(branch => branch === "actor" || branch === "actor_memory")) return "actor";
      if (sourceBranch === "output" || targetBranch === "output") return "output";
      return targetBranch || sourceBranch || "actor";
    }

    function updateCrossingGeometry() {
      if (!graphCrossingElements.length) return;
      const { width: viewportWidth, height: viewportHeight } = viewportSize();
      const unitsPerPixel = Math.max(viewBox.width / viewportWidth, viewBox.height / viewportHeight);
      const pixelsPerUnit = 1 / unitsPerPixel;
      const lowZoom = pixelsPerUnit < 0.45;
      for (const item of graphCrossingElements) {
        const crossing = item.crossing;
        const underTangent = crossing.under_tangent;
        const overTangent = crossing.over_tangent;
        if (!underTangent || !overTangent) continue;
        const underHalf = unitsPerPixel * (lowZoom ? 4.8 : 5.5);
        const gapStart = [crossing.x - underTangent[0] * underHalf, crossing.y - underTangent[1] * underHalf];
        const gapEnd = [crossing.x + underTangent[0] * underHalf, crossing.y + underTangent[1] * underHalf];
        item.gapElement.setAttribute("d", `M ${gapStart[0]} ${gapStart[1]} L ${gapEnd[0]} ${gapEnd[1]}`);

        const tangent = overTangent;
        const normal = [-tangent[1], tangent[0]];
        if (normal[1] > 0) { normal[0] *= -1; normal[1] *= -1; }
        const half = unitsPerPixel * (lowZoom ? 3.3 : 4.8);
        const height = unitsPerPixel * (lowZoom ? 1.25 : 2.4);
        const start = [crossing.x - tangent[0] * half, crossing.y - tangent[1] * half];
        const peak = [crossing.x + normal[0] * height, crossing.y + normal[1] * height];
        const end = [crossing.x + tangent[0] * half, crossing.y + tangent[1] * half];
        const firstC1 = [start[0] + tangent[0] * (half * 2 / 3), start[1] + tangent[1] * (half * 2 / 3)];
        const firstC2 = [peak[0] - normal[0] * (height * .55), peak[1] - normal[1] * (height * .55)];
        const secondC1 = [peak[0] + normal[0] * (height * .55), peak[1] + normal[1] * (height * .55)];
        const secondC2 = [end[0] - tangent[0] * (half * 2 / 3), end[1] - tangent[1] * (half * 2 / 3)];
        item.element.setAttribute("d", `M ${start[0]} ${start[1]} C ${firstC1[0]} ${firstC1[1]} ${firstC2[0]} ${firstC2[1]} ${peak[0]} ${peak[1]} C ${secondC1[0]} ${secondC1[1]} ${secondC2[0]} ${secondC2[1]} ${end[0]} ${end[1]}`);
      }
    }

    function setZoom(factor, centerX = null, centerY = null) {
      const { width: pxW, height: pxH } = viewportSize();
      const cx = centerX === null ? pxW / 2 : centerX;
      const cy = centerY === null ? pxH / 2 : centerY;
      const worldX = viewBox.x + (cx / pxW) * viewBox.width;
      const worldY = viewBox.y + (cy / pxH) * viewBox.height;
      const nextW = Math.max(80, Math.min(100000, viewBox.width * factor));
      const nextH = Math.max(60, Math.min(100000, viewBox.height * factor));
      setViewBox({
        x: worldX - (cx / pxW) * nextW,
        y: worldY - (cy / pxH) * nextH,
        width: nextW,
        height: nextH
      });
    }

    function set100Percent() {
      const { width, height } = viewportSize();
      const cx = currentBounds ? (currentBounds.minX + currentBounds.maxX) / 2 : 0;
      const cy = currentBounds ? (currentBounds.minY + currentBounds.maxY) / 2 : 0;
      setViewBox({ x: cx - width / 2, y: cy - height / 2, width, height });
    }

    function normalizeNodes(nodes) {
      if (!Array.isArray(nodes)) return [];
      return nodes.filter(item => item && typeof item === "object" && item.key !== null && item.key !== undefined)
        .map(item => ({ ...item, key: String(item.key), branch: String(item.branch || ""), parents: Array.isArray(item.parents) ? item.parents.map(String) : [] }));
    }

    function fusionLabel(node) {
      if (!node) return "融合";
      const shape = Array.isArray(node.shape) ? node.shape : null;
      if (shape && shape.length && shape.every(value => Number.isInteger(value) && value > 0)) {
        const width = shape.reduce((product, value) => product * value, 1);
        if (Number.isSafeInteger(width) && width > 0) return "融合 · " + width + "维";
      }
      const match = asText(node.title).match(/(\d+)维/);
      return match ? "融合 · " + match[1] + "维" : "融合";
    }

    function graphFromLayout(nodes, layout) {
      const positions = new Map(Object.entries(layout && layout.positions || {}));
      const visible = nodes.filter(node => positions.has(node.key));
      const keys = new Set(visible.map(node => node.key));
      const branchByKey = new Map(visible.map(node => [node.key, node.branch]));
      const edges = (layout && Array.isArray(layout.edges) ? layout.edges : [])
        .filter(edge => edge && keys.has(String(edge.source)) && keys.has(String(edge.target)) && typeof edge.path === "string")
        .map(edge => {
          const source = String(edge.source);
          const target = String(edge.target);
          return { source, target, id: `${source}→${target}`, branch: edgeBranch(source, target, branchByKey), path: edge.path };
        });
      const bounds = layout && layout.bounds;
      return {
        nodes: visible,
        positions,
        edges,
        regions: layout && Array.isArray(layout.regions) ? layout.regions : [],
        crossings: layout && Array.isArray(layout.crossings) ? layout.crossings : [],
        edgePairs: edges.map(edge => [edge.source, edge.target]),
        bounds: bounds && [bounds.minX, bounds.minY, bounds.maxX, bounds.maxY].every(Number.isFinite)
          ? { minX: bounds.minX, minY: bounds.minY, maxX: bounds.maxX, maxY: bounds.maxY }
          : null,
        layoutStatus: layout && layout.status || "missing",
        layoutEngine: layout && layout.engine || "布局数据缺失",
        layoutReason: layout && layout.reason || "",
        diagnostics: layout && layout.diagnostics || {}
      };
    }

    function textNode(x, y, className, text) {
      const node = svgElement("text", { x, y, class: className });
      node.textContent = text;
      return node;
    }

    function shortTitle(node) {
      const key = asText(node.key);
      const branch = asText(node.branch);
      const inputLabels = { lidar: "雷达输入", depth: "深度输入", detections: "检测输入", state: "状态输入", feedback: "惯性与驱动反馈" };
      const encodedLabels = { lidar: "雷达编码", depth: "深度编码", detections: "检测编码", state: "状态编码", feedback: "反馈编码" };
      if (key.startsWith("input.")) return inputLabels[branch] || "输入";
      if (key === "features_extractor") return fusionLabel(node);
      if (key === "lstm_actor") return "驾驶记忆 LSTM";
      if (key === "action.applied" || key === "action_net") return "左右履带＋云台";
      const text = asText(node.title || key || "未命名层");
      const max = 18;
      return text.length > max ? text.slice(0, max - 1) + "…" : text;
    }

    function createCard(node, position) {
      const group = svgElement("g", {
        class: "node-card",
        transform: "translate(" + position.x + " " + position.y + ")",
        tabindex: "0",
        role: "button"
      });
      group.dataset.nodeKey = node.key;
      group.setAttribute("aria-label", nodeDescription(node) + "，" + shapeText(node.shape));
      group.setAttribute("title", nodeDescription(node));
      const background = svgElement("rect", { class: "card-bg", x: 0, y: 0, width: CARD_W, height: CARD_H, rx: 9 });
      const title = textNode(10, 18, "svg-title", "");
      const shape = textNode(10, 35, "svg-shape", "");
      const image = svgElement("image", { x: 11, y: 43, width: 148, height: 55, preserveAspectRatio: "xMidYMid meet", "clip-path": "url(#card-image-clip)" });
      const missing = textNode(CARD_W / 2, 75, "svg-missing", "尚未采集");
      missing.setAttribute("text-anchor", "middle");
      group.append(background, title, shape, image, missing);
      group.addEventListener("click", event => {
        event.stopPropagation();
        if (performance.now() < suppressClickUntil) return;
        chooseNode(node.key);
      });
      group.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); chooseNode(node.key); }
      });
      const setHovered = () => {
        if (hoveredKey === node.key) return;
        hoveredKey = node.key;
        updateSelectionStyle();
      };
      const clearHovered = () => {
        if (document.activeElement === group || hoveredKey !== node.key) return;
        hoveredKey = null;
        updateSelectionStyle();
      };
      group.addEventListener("pointerenter", setHovered);
      group.addEventListener("pointerleave", clearHovered);
      group.addEventListener("focus", setHovered);
      group.addEventListener("blur", clearHovered);
      els.nodeLayer.appendChild(group);
      return { group, title, shape, image, missing, node };
    }

    function nodeDescription(node) {
      const title = asText(node.title || node.key);
      const description = asText(node.description);
      return description ? title + "；" + description : title;
    }

    function updateCard(record, node, position) {
      record.node = node;
      record.position = position;
      record.group.setAttribute("transform", "translate(" + position.x + " " + position.y + ")");
      record.group.setAttribute("aria-label", nodeDescription(node) + "，" + shapeText(node.shape));
      record.group.setAttribute("title", nodeDescription(node));
      record.title.textContent = shortTitle(node);
      record.shape.textContent = shapeText(node.shape);
      const uri = safeImageUri(node.image);
      if (uri) {
        if (record.image.getAttribute("href") !== uri) record.image.setAttribute("href", uri);
        record.image.style.display = "";
        record.missing.style.display = "none";
      } else {
        record.image.removeAttribute("href");
        record.image.style.display = "none";
        record.missing.style.display = "";
        record.missing.textContent = "未采集 / 暂无图像";
      }
      record.group.classList.toggle("selected", node.key === selectedKey);
    }

    function renderLabels(graph) {
      clearChildren(els.labels);
      for (const region of graph.regions) {
        const text = textNode(region.x + 10, region.y + 25, "region-label", region.label);
        els.labels.appendChild(text);
      }
    }

    function renderPartitions(graph) {
      clearChildren(els.partitions);
      for (const region of graph.regions) {
        const box = svgElement("rect", {
          class: "partition-bg", x: region.x, y: region.y,
          width: region.width, height: region.height, rx: 16
        });
        els.partitions.appendChild(box);
      }
    }

    function renderEdges(graph) {
      clearChildren(els.edgeLayer);
      graph.edgeElements = [];
      graph.crossingElements = [];
      for (const edge of graph.edges) {
        const path = svgElement("path", {
          class: `flow-edge branch-${edge.branch}`,
          d: edge.path,
          "data-source": edge.source,
          "data-target": edge.target,
          "aria-hidden": "true"
        });
        els.edgeLayer.appendChild(path);
        graph.edgeElements.push({ edge, element: path });
      }
      const crossings = graph.crossings.filter(crossing =>
        Array.isArray(crossing.under_tangent) && crossing.under_tangent.length === 2 &&
        Array.isArray(crossing.over_tangent) && crossing.over_tangent.length === 2 &&
        crossing.under && crossing.over
      );
      const edgeById = new Map(graph.edges.map(edge => [edge.id, edge]));
      for (const crossing of crossings) {
        const gap = svgElement("path", {
          class: "crossing-gap",
          d: "",
          "aria-hidden": "true"
        });
        els.edgeLayer.appendChild(gap);
        const overEdge = edgeById.get(crossing.over);
        const bridge = svgElement("path", {
          class: `crossing-bridge branch-${overEdge ? overEdge.branch : "actor"}`,
          d: "",
          "data-under": crossing.under,
          "data-over": crossing.over,
          "aria-hidden": "true"
        });
        const title = svgElement("title");
        title.textContent = `跨线弧：${crossing.over} 跨过 ${crossing.under}；两条边都是真实数据依赖`;
        bridge.appendChild(title);
        els.edgeLayer.appendChild(bridge);
        graph.crossingElements.push({ under: crossing.under, over: crossing.over, crossing, gapElement: gap, element: bridge });
      }
    }

    function updateSelectionStyle() {
      const activeKey = hoveredKey && renderedKeys.has(hoveredKey) ? hoveredKey : selectedKey;
      const related = new Set();
      for (const item of graphEdgeElements) {
        const { source, target } = item.edge;
        if (source === activeKey || target === activeKey) {
          related.add(source);
          related.add(target);
        }
      }
      for (const [key, record] of renderedKeys) {
        record.group.classList.toggle("selected", key === selectedKey);
        record.group.classList.toggle("hovered", key === hoveredKey);
        record.group.classList.toggle("related", key !== activeKey && related.has(key));
      }
    }

    function drawGraph(rawNodes, layoutData) {
      const nodes = normalizeNodes(rawNodes);
      const layerSelect = document.getElementById("layer-select");
      const choices = nodes.map(node => [node.key, node.title || node.key]);
      if (!choices.some(([key]) => key === selectedKey)) choices.push([selectedKey, selectedKey]);
      const choicesSignature = JSON.stringify(choices);
      if (choicesSignature !== layerOptionsSignature) {
        clearChildren(layerSelect);
        for (const [key, title] of choices) addOption(layerSelect, key, title + " · " + key);
        layerOptionsSignature = choicesSignature;
      }
      layerSelect.value = selectedKey;
      const graph = graphFromLayout(nodes, layoutData);
      const signature = asText(layoutData && layoutData.topology) || JSON.stringify(graph.edgePairs);
      const topologyChanged = signature !== topologySignature;
      if (graph.bounds) currentBounds = graph.bounds;
      els.empty.hidden = graph.nodes.length > 0;
      const metricCounts = graph.diagnostics.counts || {};
      const crossingCount = Number(metricCounts.crossing_points || graph.crossings.length || 0);
      const engineText = graph.layoutStatus === "graphviz" ? "Graphviz 曲线" : "降级曲线布局";
      els.graphStatus.textContent = `${graph.nodes.length} 层完整展开 · ${engineText} · ${crossingCount} 处交叉`;
      els.graphStatus.title = [graph.layoutEngine, graph.layoutReason,
        `卡片重叠 ${metricCounts.card_overlap_pairs ?? "—"}；曲线穿卡 ${metricCounts.curve_card_intersections ?? "—"}；共线路径 ${metricCounts.shared_path_pairs ?? "—"}；曲线交叉 ${metricCounts.crossing_points ?? "—"}`
      ].filter(Boolean).join("\n");

      if (topologyChanged) {
        clearChildren(els.nodeLayer);
        renderedKeys = new Map();
        graphEdgeElements = [];
        graphCrossingElements = [];
        if (hoveredKey && !graph.nodes.some(node => node.key === hoveredKey)) hoveredKey = null;
        renderPartitions(graph);
        renderLabels(graph);
        renderEdges(graph);
        for (const node of graph.nodes) {
          const position = graph.positions.get(node.key);
          if (!position) continue;
          const record = createCard(node, position);
          renderedKeys.set(node.key, record);
          updateCard(record, node, position);
        }
        graphEdgeElements = graph.edgeElements || [];
        graphCrossingElements = graph.crossingElements || [];
        topologySignature = signature;
      } else {
        const current = new Map(graph.nodes.map(node => [node.key, node]));
        for (const [key, record] of renderedKeys) {
          const node = current.get(key);
          if (node) updateCard(record, node, graph.positions.get(key));
        }
      }
      if (!hasInitialFit && currentBounds) fitAll();
      updateCrossingGeometry();
      updateSelectionStyle();
      lastFrame = lastFrame || {};
      lastFrame.nodes = nodes;
    }

    function addOption(select, value, label) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    }

    function updateParameterOptions(parameters) {
      const names = Array.isArray(parameters) ? parameters.map(asText).filter(Boolean) : [];
      const next = ["激活值", ...names];
      if (JSON.stringify(next) === JSON.stringify(parameterOptions)) return;
      parameterOptions = next;
      clearChildren(els.tensor);
      for (const item of next) addOption(els.tensor, item, item === "激活值" ? "激活值" : "参数：" + item);
      if (!next.includes(tensorName)) tensorName = "激活值";
      els.tensor.value = tensorName;
    }

    function renderFields(fields) {
      clearChildren(els.fieldList);
      const items = Array.isArray(fields) ? fields : [];
      if (!items.length) {
        const li = document.createElement("li");
        li.textContent = "该层没有单独字段说明。";
        els.fieldList.appendChild(li);
        return;
      }
      for (const field of items) {
        const li = document.createElement("li");
        li.textContent = asText(field);
        els.fieldList.appendChild(li);
      }
    }

    function renderDetail(detail) {
      if (!detail || typeof detail !== "object") {
        els.detailTitle.textContent = "尚未选中网络层";
        els.detailPlaceholder.hidden = false;
        els.detailContent.hidden = true;
        return;
      }
      if (detail.key !== undefined && String(detail.key) !== selectedKey) return;
      els.detailTitle.textContent = asText(detail.title || detail.key || selectedKey);
      els.detailPlaceholder.hidden = true;
      els.detailContent.hidden = false;
      const uri = safeImageUri(detail.image);
      if (uri) {
        if (els.detailImage.getAttribute("src") !== uri) els.detailImage.src = uri;
        els.detailImage.hidden = false;
        els.detailImageEmpty.hidden = true;
        if (!els.detailImage.parentElement.classList.contains("zoomed")) els.detailImage.title = "点击查看原图";
      } else {
        els.detailImage.removeAttribute("src");
        els.detailImage.hidden = true;
        els.detailImageEmpty.textContent = "未采集激活或当前没有可显示图像";
        els.detailImageEmpty.hidden = false;
      }
      const legend = detail.legend || {};
      const validLegend = typeof legend.scale === "number" && Number.isFinite(legend.scale) && legend.scale > 0;
      els.detailColorKey.hidden = !validLegend || !uri;
      if (validLegend) {
        const local = legend.mode === "adaptive-local";
        els.detailScaleReadout.textContent = (local ? "单图自适应" : "全局统一") + " · ±" + legendNumber(legend.scale);
        els.detailScaleReadout.title = "当前所选张量的预览范围：±" + String(legend.scale) + "；0始终同色，完整覆盖有限值。";
        document.getElementById("detail-legend-min").textContent = signedLegendNumber(legend.minimum);
        document.getElementById("detail-legend-max").textContent = signedLegendNumber(legend.maximum);
        if (Array.isArray(legend.colors)) {
          const colors = legend.colors.filter(rgb => Array.isArray(rgb) && rgb.length === 3 && rgb.every(n => Number.isFinite(n) && n >= 0 && n <= 255));
          if (colors.length >= 2) document.getElementById("detail-spectrum-bar").style.background = "linear-gradient(90deg," + colors.map(rgb => "rgb(" + rgb.join(",") + ")").join(",") + ")";
        }
      }
      setText("detail-shape", shapeText(detail.shape));
      document.getElementById("detail-size").textContent = detail.size === null ? "—" : String(detail.size);
      setText("detail-source", tensorName === "激活值" ? detail.source : "检查点参数（只读）");
      setText("detail-description", detail.description);
      document.getElementById("detail-coordinate").textContent = Array.isArray(detail.coordinate) ? "[" + detail.coordinate.join(", ") + "]" : "—";
      document.getElementById("detail-value").textContent = typeof detail.value === "number" ? detail.value.toPrecision(9) : fmt(detail.value);
      setText("stat-min", detail.min);
      setText("stat-max", detail.max);
      setText("stat-mean", detail.mean);
      setText("stat-std", detail.std);
      document.getElementById("stat-nonfinite").textContent = detail.nonfinite === null ? "—" : String(detail.nonfinite);
      renderFields(detail.fields);
      updateParameterOptions(detail.parameters);
      if (detail.index !== undefined && document.activeElement !== els.index) {
        const index = Number(detail.index);
        if (Number.isFinite(index) && index >= 0) { elementIndex = Math.floor(index); els.index.value = String(elementIndex); }
      }
    }

    function displayTime(value) {
      return value === null || value === undefined ? "—" : fmt(value, 3) + " s";
    }

    function updateHeader(frame) {
      const legend = frame.legend || {};
      const minimum = legend.minimum;
      const maximum = legend.maximum;
      if (typeof minimum === "number" && Number.isFinite(minimum) && typeof maximum === "number" && Number.isFinite(maximum)) {
        const sharedMagnitude = Math.max(Math.abs(minimum), Math.abs(maximum));
        els.scaleValue.textContent = "±" + legendNumber(sharedMagnitude);
        els.scaleReadout.title = scaleReadoutBaseTitle + " 后端完整精度范围：±" + String(sharedMagnitude) + "。";
        document.getElementById("legend-min").textContent = signedLegendNumber(minimum);
        document.getElementById("legend-max").textContent = signedLegendNumber(maximum);
      } else {
        els.scaleValue.textContent = "±—";
        els.scaleReadout.title = scaleReadoutBaseTitle;
        document.getElementById("legend-min").textContent = "—";
        document.getElementById("legend-max").textContent = "—";
      }
      if (Array.isArray(legend.colors) && legend.colors.length >= 2) {
        const colors = legend.colors.filter(rgb => Array.isArray(rgb) && rgb.length === 3 && rgb.every(n => Number.isFinite(n) && n >= 0 && n <= 255));
        const signature = JSON.stringify(colors);
        if (signature !== legendSignature && colors.length >= 2) {
          document.getElementById("spectrum-bar").style.background = "linear-gradient(90deg," + colors.map(rgb => "rgb(" + rgb.join(",") + ")").join(",") + ")";
          document.getElementById("spectrum-bar").title = asText(legend.name);
          legendSignature = signature;
        }
      }
      els.frame.textContent = frame.sequence === null || frame.sequence === undefined ? "—" : String(frame.sequence);
      els.inputTime.textContent = displayTime(frame.sim_time);
      els.outputTime.textContent = displayTime(frame.output_time);
      els.mode.textContent = asText(frame.mode || "—");
      const criticDescription = asText(frame.critic);
      els.critic.textContent = criticDescription ? "独立只读" : "—";
      els.critic.title = criticDescription;
      const state = frame.state && typeof frame.state === "object" ? frame.state : {};
      const running = state.running === true;
      let runtime = state.done ? "回合结束" : (running ? "运行中" : "已暂停");
      if (state.time !== undefined) runtime += " · " + fmt(state.time, 2) + " s";
      els.runtime.textContent = runtime;
      els.runtime.title = asText(state.status);
      els.training.textContent = Number.isInteger(state.training_steps)
        ? (state.training_steps === 0 ? "未训练 · 随机初始化" : state.training_steps.toLocaleString() + " 步")
        : "训练步数未记录";
      const errors = Array.isArray(frame.errors) ? frame.errors.filter(Boolean).map(asText) : (frame.errors ? [asText(frame.errors)] : []);
      const robot = state.robot && typeof state.robot === "object" ? state.robot : null;
      const robotText = robot ? Object.entries(robot).slice(0, 4).map(([key, value]) => key + " " + fmt(value, 2)).join(" · ") : "";
      const commandError = asText(state.status).startsWith("请求失败") ? asText(state.status) : "";
      els.message.textContent = [...errors, commandError, robotText].filter(Boolean).join(" · ") || "实时帧已更新";
      els.connection.textContent = "已连接";
      els.connection.classList.remove("busy");
      els.connection.classList.add("online");
    }

    function setConnection(connected, message = "") {
      els.connection.classList.toggle("online", connected);
      els.connection.classList.toggle("busy", !connected && fetching);
      els.connection.textContent = connected ? "已连接" : (fetching ? "连接中" : "已断开");
      if (message) els.message.textContent = message;
    }

    function queryUrl() {
      const query = new URLSearchParams({
        detail: "1",
        selected: selectedKey,
        tensor: tensorName,
        index: String(Math.max(0, elementIndex)),
        scale: "auto",
        detail_scale: els.detailLocalScale.checked ? "local" : "global"
      });
      return "/api/frame?" + query.toString();
    }

    async function pollOnce() {
      if (fetching || frozen || document.visibilityState === "hidden") return;
      fetching = true;
      if (!els.connection.classList.contains("online")) setConnection(false);
      const requestedKey = selectedKey;
      const requestedTensor = tensorName;
      const requestedLocalScale = els.detailLocalScale.checked;
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 4000);
      try {
        const response = await fetch(queryUrl(), { method: "GET", cache: "no-store", headers: { "Accept": "application/json" }, signal: controller.signal });
        if (!response.ok) throw new Error("HTTP " + response.status);
        const frame = await response.json();
        if (!frame || typeof frame !== "object") throw new Error("帧数据格式无效");
        lastFrame = frame;
        updateHeader(frame);
        drawGraph(frame.nodes, frame.layout);
        if (requestedKey === selectedKey && requestedTensor === tensorName && requestedLocalScale === els.detailLocalScale.checked) renderDetail(frame.detail);
      } catch (error) {
        setConnection(false, "等待本地服务恢复：" + asText(error && error.message || error));
      } finally {
        window.clearTimeout(timeout);
        fetching = false;
        if (!els.connection.classList.contains("online")) {
          els.connection.textContent = "已断开";
          els.connection.classList.remove("busy");
        }
      }
    }

    function schedulePoll(delay = POLL_MS) {
      window.clearTimeout(requestTimer);
      requestTimer = window.setTimeout(async () => {
        await pollOnce();
        schedulePoll(POLL_MS);
      }, delay);
    }

    function requestRefresh() {
      if (frozen) return;
      if (!fetching) {
        window.clearTimeout(requestTimer);
        schedulePoll(0);
      }
    }

    async function sendCommand(command) {
      if (controlBusy) return;
      controlBusy = true;
      const buttons = document.querySelectorAll("[data-command]");
      buttons.forEach(button => { button.disabled = true; });
      els.message.textContent = "已发送命令：" + command;
      try {
        const response = await fetch("/api/control", {
          method: "POST",
          cache: "no-store",
          headers: { "Content-Type": "application/json", "Accept": "application/json" },
          body: JSON.stringify({ command })
        });
        if (!response.ok) throw new Error("HTTP " + response.status);
        let result = null;
        try { result = await response.json(); } catch (_) { /* empty response is valid */ }
        if (result && result.message) els.message.textContent = asText(result.message);
        else els.message.textContent = "命令已接受：" + command;
      } catch (error) {
        els.message.textContent = "命令发送失败：" + asText(error && error.message || error);
        setConnection(false);
      } finally {
        buttons.forEach(button => { button.disabled = false; });
        controlBusy = false;
        requestRefresh();
      }
    }

    function chooseNode(key) {
      setDetailsVisible(true);
      if (key === selectedKey) {
        requestRefresh();
        return;
      }
      selectedKey = key;
      document.getElementById("layer-select").value = key;
      tensorName = "激活值";
      elementIndex = 0;
      els.detailImage.parentElement.classList.remove("zoomed");
      els.detailImage.title = "点击查看原图";
      els.tensor.value = tensorName;
      els.index.value = "0";
      updateSelectionStyle();
      requestRefresh();
    }

    function setDetailsVisible(visible) {
      detailVisible = visible;
      els.detailPanel.classList.toggle("collapsed", !visible);
      els.detailToggle.textContent = visible ? "收起详情" : "详情";
      els.detailToggle.setAttribute("aria-expanded", String(visible));
      els.detailToggle.setAttribute("title", visible ? "收起张量与层信息" : "展开张量与层信息");
      els.detailCollapse.textContent = "收起";
    }

    document.querySelectorAll("[data-command]").forEach(button => {
      button.addEventListener("click", () => sendCommand(button.dataset.command));
    });
    els.freeze.addEventListener("change", () => {
      frozen = els.freeze.checked;
      if (frozen) {
        els.message.textContent = "本页刷新已冻结；仿真和后端命令仍可继续。";
        els.connection.textContent = "画面冻结";
        els.connection.classList.remove("busy");
      } else {
        els.message.textContent = "恢复本页实时刷新。";
        requestRefresh();
      }
    });
    els.tensor.addEventListener("change", () => {
      tensorName = els.tensor.value || "激活值";
      els.detailImage.parentElement.classList.remove("zoomed");
      els.detailImage.title = "点击查看原图";
      requestRefresh();
    });
    els.detailLocalScale.addEventListener("change", () => {
      els.detailColorKey.hidden = true;
      els.detailImage.hidden = true;
      els.detailImageEmpty.textContent = "正在更新预览色标…";
      els.detailImageEmpty.hidden = false;
      requestRefresh();
    });
    els.detailImage.addEventListener("click", () => {
      if (els.detailImage.hidden) return;
      const zoomed = els.detailImage.parentElement.classList.toggle("zoomed");
      els.detailImage.title = zoomed ? "点击缩回适配视图" : "点击查看原图";
    });
    els.index.addEventListener("change", () => {
      const value = Number(els.index.value);
      elementIndex = Number.isFinite(value) ? Math.max(0, Math.floor(value)) : 0;
      els.index.value = String(elementIndex);
      requestRefresh();
    });
    document.getElementById("fit-button").addEventListener("click", fitAll);
    const regionSidebar = document.querySelector(".region-sidebar");
    const navDrawerToggle = document.getElementById("nav-drawer-toggle");
    navDrawerToggle.addEventListener("click", () => {
      const open = !regionSidebar.classList.contains("nav-open");
      regionSidebar.classList.toggle("nav-open", open);
      navDrawerToggle.setAttribute("aria-expanded", String(open));
      navDrawerToggle.setAttribute("aria-label", open ? "收起网络分区导航" : "展开网络分区导航");
      navDrawerToggle.title = open ? "收起网络分区导航" : "展开网络分区导航";
    });
    document.querySelectorAll(".region-link[data-region]").forEach(button => {
      button.addEventListener("click", () => {
        focusRegion(button.dataset.region);
        regionSidebar.classList.remove("nav-open");
        navDrawerToggle.setAttribute("aria-expanded", "false");
        navDrawerToggle.setAttribute("aria-label", "展开网络分区导航");
        navDrawerToggle.title = "展开网络分区导航";
      });
    });
    document.getElementById("layer-select").addEventListener("change", event => chooseNode(event.target.value));
    document.getElementById("focus-layer-button").addEventListener("click", () => {
      const record = renderedKeys.get(selectedKey);
      if (!record || !record.position) return;
      const { width, height } = viewportSize();
      setViewBox({ x: record.position.x + CARD_W / 2 - width / 2,
                   y: record.position.y + CARD_H / 2 - height / 2, width, height });
    });
    document.getElementById("one-to-one-button").addEventListener("click", set100Percent);
    document.getElementById("zoom-in-button").addEventListener("click", () => setZoom(0.82));
    document.getElementById("zoom-out-button").addEventListener("click", () => setZoom(1.22));
    els.detailToggle.addEventListener("click", () => setDetailsVisible(!detailVisible));
    els.detailCollapse.addEventListener("click", () => setDetailsVisible(false));

    els.svg.addEventListener("wheel", event => {
      event.preventDefault();
      const rect = els.svg.getBoundingClientRect();
      setZoom(event.deltaY < 0 ? 0.88 : 1.14, event.clientX - rect.left, event.clientY - rect.top);
    }, { passive: false });
    els.svg.addEventListener("pointerdown", event => {
      if (event.button !== 0 || event.target.closest?.(".node-card")) return;
      const rect = els.svg.getBoundingClientRect();
      pan = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, rect, moved: false };
      els.svg.setPointerCapture(event.pointerId);
      els.svg.style.cursor = "grabbing";
    });
    els.svg.addEventListener("pointermove", event => {
      if (!pan || event.pointerId !== pan.pointerId) return;
      const dx = event.clientX - pan.x, dy = event.clientY - pan.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) pan.moved = true;
      const rect = els.svg.getBoundingClientRect();
      setViewBox({
        x: viewBox.x - dx * viewBox.width / Math.max(1, rect.width),
        y: viewBox.y - dy * viewBox.height / Math.max(1, rect.height),
        width: viewBox.width, height: viewBox.height
      });
      pan.x = event.clientX; pan.y = event.clientY;
    });
    const endPan = event => {
      if (!pan || event.pointerId !== pan.pointerId) return;
      if (pan.moved) suppressClickUntil = performance.now() + 80;
      pan = null;
      els.svg.style.cursor = "";
      if (els.svg.hasPointerCapture(event.pointerId)) els.svg.releasePointerCapture(event.pointerId);
    };
    els.svg.addEventListener("pointerup", endPan);
    els.svg.addEventListener("pointercancel", endPan);
    document.addEventListener("visibilitychange", () => { if (!document.hidden && !frozen) requestRefresh(); });
    window.addEventListener("resize", () => {
      updateCrossingGeometry();
      if (!window.matchMedia("(max-width: 920px)").matches) {
        regionSidebar.classList.remove("nav-open");
        navDrawerToggle.setAttribute("aria-expanded", "false");
      } else if (detailVisible) {
        setDetailsVisible(false);
      }
    });

    updateParameterOptions([]);
    setDetailsVisible(false);
    setViewBox(viewBox);
    schedulePoll(0);
  })();
  