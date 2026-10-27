"use strict";

(() => {
  const elements = {
    connectionStatus: document.getElementById("connectionStatus"),
    resetButton: document.getElementById("resetButton"),
    sessionChip: document.getElementById("sessionChip"),
    sessionId: document.getElementById("sessionId"),
    stepValue: document.getElementById("stepValue"),
    roundTitle: document.getElementById("round-title"),
    sealPill: document.getElementById("sealPill"),
    sealText: document.getElementById("sealText"),
    resultLabel: document.getElementById("resultLabel"),
    resultContent: document.getElementById("resultContent"),
    digitButtons: Array.from(document.querySelectorAll(".digit-button")),
    liveMessage: document.getElementById("liveMessage"),
    hitCount: document.getElementById("hitCount"),
    accuracyValue: document.getElementById("accuracyValue"),
    totalCount: document.getElementById("totalCount"),
    recent30Hits: document.getElementById("recent30Hits"),
    recent90Hits: document.getElementById("recent90Hits"),
    recentCount: document.getElementById("recentCount"),
    recentKeys: document.getElementById("recentKeys"),
    earlyReveal: document.getElementById("earlyReveal"),
    predictionHint: document.getElementById("predictionHint"),
    predictionReadout: document.getElementById("predictionReadout"),
    predictionValue: document.getElementById("predictionValue"),
    probabilityList: document.getElementById("probabilityList"),
    neuronGrid: document.getElementById("neuronGrid"),
    weightTable: document.getElementById("weightTable"),
    weightValues: document.getElementById("weightValues"),
    footerSession: document.getElementById("footerSession"),
    graphStep: document.getElementById("graphStep"),
    trainingModeLabel: document.getElementById("trainingModeLabel"),
    modelModeDetail: document.getElementById("modelModeDetail"),
    inputDigitSelect: document.getElementById("inputDigitSelect"),
    inputMapStatus: document.getElementById("inputMapStatus"),
    inputCountList: document.getElementById("inputCountList"),
    inputTargetList: document.getElementById("inputTargetList"),
    graphZoom: document.getElementById("graphZoom"),
    graphCenterStep: document.getElementById("graphCenterStep"),
    edgeThreshold: document.getElementById("edgeThreshold"),
    edgeThresholdValue: document.getElementById("edgeThresholdValue"),
    graphEdgeCount: document.getElementById("graphEdgeCount"),
    neuronGraph: document.getElementById("neuronGraph"),
    graphEdges: document.getElementById("graphEdges"),
    graphNodes: document.getElementById("graphNodes"),
    graphSelectionEmpty: document.getElementById("graphSelectionEmpty"),
    graphInspector: document.getElementById("graphInspector"),
    selectedNeuronTitle: document.getElementById("selectedNeuronTitle"),
    selectedNeuronActivity: document.getElementById("selectedNeuronActivity"),
    selectedIncomingCount: document.getElementById("selectedIncomingCount"),
    selectedOutgoingCount: document.getElementById("selectedOutgoingCount"),
    selectedIncoming: document.getElementById("selectedIncoming"),
    selectedOutgoing: document.getElementById("selectedOutgoing"),
    clearNeuronSelection: document.getElementById("clearNeuronSelection"),
  };

  let currentState = null;
  let busy = false;
  let selectedNeuronIndex = null;
  let selectedInputDigit = 0;
  let inputDigitManuallySelected = false;
  let stateRevision = 0;
  let pollInFlight = false;

  function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function displayNumber(value, places = 3) {
    if (!isFiniteNumber(value)) return "—";
    return value.toFixed(places);
  }

  function displayWeight(value) {
    if (!isFiniteNumber(value)) return "—";
    if (value === 0) return "0";
    const absolute = Math.abs(value);
    const formatted = absolute < 0.0001 || absolute >= 100000
      ? value.toExponential(4)
      : value.toFixed(5);
    return value > 0 ? `+${formatted}` : formatted;
  }

  function percentage(value) {
    if (!isFiniteNumber(value)) return "0%";
    return `${(value * 100).toFixed(value * 100 < 10 && value > 0 ? 1 : 0)}%`;
  }

  function setConnection(label, state) {
    elements.connectionStatus.className = `connection-pill ${state ? `is-${state}` : ""}`.trim();
    elements.connectionStatus.replaceChildren();
    const dot = document.createElement("span");
    dot.className = "status-dot";
    dot.setAttribute("aria-hidden", "true");
    elements.connectionStatus.append(dot, document.createTextNode(label));
  }

  function setBusy(value, label = "") {
    busy = value;
    if (value) stateRevision += 1;
    elements.digitButtons.forEach((button) => { button.disabled = value || !currentState; });
    elements.resetButton.disabled = value || !currentState;
    if (value) setConnection(label || "正在处理…", "busy");
    else if (currentState) setConnection("本地模型在线", "online");
  }

  function showMessage(message, isError = false) {
    elements.liveMessage.textContent = message || "";
    elements.liveMessage.classList.toggle("is-error", isError);
  }

  function validState(state) {
    return state && typeof state === "object" &&
      typeof state.session_id === "string" &&
      Number.isInteger(state.step) &&
      typeof state.prediction_id === "string" &&
      Array.isArray(state.probabilities) && state.probabilities.length === 10 &&
      Array.isArray(state.neurons) && state.neurons.length === 20 &&
      Array.isArray(state.connections) && state.connections.length === 20 &&
      state.connections.every((row) => Array.isArray(row) && row.length === 20) &&
      state.metrics && typeof state.metrics === "object";
  }

  async function requestJson(url, options) {
    const response = await fetch(url, {
      cache: "no-store",
      ...options,
      headers: { Accept: "application/json", ...(options && options.headers ? options.headers : {}) },
    });
    let body;
    try {
      body = await response.json();
    } catch (_error) {
      throw new Error(`服务返回了无法读取的数据（${response.status}）`);
    }
    if (!response.ok) {
      const message = body && body.error && body.error.message;
      const error = new Error(message || `请求失败（${response.status}）`);
      error.status = response.status;
      error.payload = body;
      throw error;
    }
    return body;
  }

  async function loadState({ keepMessage = false } = {}) {
    stateRevision += 1;
    const state = await requestJson("/api/state");
    if (!validState(state)) throw new Error("本地服务返回的状态格式不完整。");
    currentState = state;
    render();
    setBusy(false);
    if (!keepMessage) showMessage("");
    return state;
  }

  function makeResultNumber(value, predicted = false) {
    const node = document.createElement("strong");
    node.className = `result-number${predicted ? " predicted" : ""}`;
    node.textContent = Number.isInteger(value) ? String(value) : "—";
    return node;
  }

  function renderResult() {
    const result = currentState.last_result;
    elements.resultContent.replaceChildren();
    elements.sealPill.classList.remove("is-revealed");

    if (!result || typeof result !== "object") {
      elements.resultLabel.textContent = "上一轮结果";
      const emptyIcon = document.createElement("span");
      emptyIcon.className = "result-empty-icon";
      emptyIcon.textContent = "?";
      emptyIcon.setAttribute("aria-hidden", "true");
      elements.resultContent.append(emptyIcon, document.createTextNode(currentState.step > 0 ? "这段会话还没有可展示的揭晓结果。" : "按下任意数字，开始第一轮。"));
      elements.roundTitle.textContent = currentState.step > 0 ? "预测已准备好" : "按一个数字开始";
      elements.sealText.textContent = "预测已封存";
      return;
    }

    elements.resultLabel.textContent = `第 ${Number.isInteger(result.step) ? result.step : currentState.step} 次按键 · 刚才的预测`;
    const guessText = document.createElement("span");
    guessText.textContent = "它猜";
    const separator = document.createElement("span");
    separator.className = "result-divider";
    separator.textContent = "→";
    const actualText = document.createElement("span");
    actualText.textContent = "你按";
    const badge = document.createElement("span");
    const correct = result.correct === true;
    badge.className = `result-badge${correct ? "" : " miss"}`;
    badge.textContent = correct ? "猜中了" : "没猜中";
    elements.resultContent.append(
      guessText,
      makeResultNumber(result.prediction, true),
      separator,
      actualText,
      makeResultNumber(result.actual_digit),
      badge,
    );
    const predictionSource = result.prediction_source || result.source;
    if (predictionSource || result.step === 1) {
      const sourceNote = document.createElement("span");
      sourceNote.className = "result-source-note";
      sourceNote.textContent = predictionSource === "uniform_cold_start"
        ? "冷启动均匀预测：各数字概率相同，并列时选 0。"
        : predictionSource === "sparse_rnn"
          ? "窗口训练 · 自适应稀疏输入模型的预测。"
          : `预测来源：${String(predictionSource || "初始均匀预测")}`;
      elements.resultContent.append(sourceNote);
    }
    elements.roundTitle.textContent = "看看它刚才猜得对不对";
    elements.sealPill.classList.add("is-revealed");
    elements.sealText.textContent = "下一轮预测已封存";
  }

  function renderMetrics() {
    const metrics = currentState.metrics || {};
    const correct = Number.isInteger(metrics.correct) ? metrics.correct : 0;
    const total = Number.isInteger(metrics.total) ? metrics.total : 0;
    const accuracy = isFiniteNumber(metrics.accuracy) ? metrics.accuracy : (total > 0 ? correct / total : 0);
    elements.hitCount.textContent = `${correct} / ${total}`;
    elements.accuracyValue.textContent = percentage(accuracy);
    elements.totalCount.textContent = String(total);
    const recent = currentState.recent_hit_counts;
    [[30, elements.recent30Hits], [90, elements.recent90Hits]].forEach(([windowSize, target]) => {
      const stats = recent && recent[String(windowSize)];
      if (stats && Number.isInteger(stats.correct) && Number.isInteger(stats.total)) {
        target.textContent = `${stats.correct} / ${stats.total}`;
        target.title = `最近 ${windowSize} 次按键中的命中数`;
      } else {
        target.textContent = "—";
        target.title = "当前服务未提供最近命中统计";
      }
    });
  }

  function renderModelSummary() {
    const info = currentState.model_info && typeof currentState.model_info === "object"
      ? currentState.model_info
      : {};
    elements.trainingModeLabel.textContent = "窗口训练 · 自适应稀疏输入";
    elements.modelModeDetail.textContent = "每次训练重放最近 32 位；每次更新后对输入权重做 λ=0.02 软阈值收缩。";

    const history = Array.isArray(currentState.history) ? currentState.history : [];
    if (!inputDigitManuallySelected) {
      const latestDigit = history.length ? history[history.length - 1] : 0;
      selectedInputDigit = Number.isInteger(latestDigit) && latestDigit >= 0 && latestDigit <= 9
        ? latestDigit
        : 0;
    }
    elements.inputDigitSelect.value = String(selectedInputDigit);

    const matrix = Array.isArray(currentState.input_connections)
      ? currentState.input_connections
      : Array.isArray(info.input_weights) ? info.input_weights : null;
    const hasInputMatrix = matrix && matrix.length === 20 && matrix.every((row) =>
      Array.isArray(row) && row.length === 10 && row.every(isFiniteNumber));
    elements.inputCountList.replaceChildren();
    elements.inputTargetList.replaceChildren();
    if (!hasInputMatrix) {
      elements.inputMapStatus.textContent = "此服务未提供 W_in 数据，无法判断哪些边精确为零。";
      const unavailable = document.createElement("div");
      unavailable.className = "input-data-missing";
      unavailable.textContent = "输入连接数据不可用";
      elements.inputTargetList.append(unavailable);
      return;
    }

    const counts = Array.from({ length: 10 }, (_value, digit) =>
      matrix.reduce((count, row) => count + (row[digit] !== 0 ? 1 : 0), 0));
    const total = counts.reduce((sum, count) => sum + count, 0);
    for (let digit = 0; digit < 10; digit += 1) {
      const chip = document.createElement("span");
      chip.className = `input-count-chip${digit === selectedInputDigit ? " is-selected" : ""}`;
      chip.setAttribute("aria-label", `数字 ${digit}，${counts[digit]} 个直接连接`);
      const label = document.createElement("span");
      label.textContent = String(digit);
      const count = document.createElement("strong");
      count.textContent = String(counts[digit]);
      chip.append(label, count);
      elements.inputCountList.append(chip);
    }
    elements.inputMapStatus.textContent = `数字 ${selectedInputDigit} → ${counts[selectedInputDigit]} / 20 个神经元有直接输入连接；全部数字共 ${total} 条非零输入连接。`;
    for (let node = 0; node < 20; node += 1) {
      const weight = matrix[node][selectedInputDigit];
      const connected = weight !== 0;
      const item = document.createElement("div");
      item.className = `input-target${connected ? " is-connected" : ""}`;
      item.setAttribute("role", "listitem");
      item.setAttribute("aria-label", `神经元 ${node + 1}，${connected ? "已连接" : "权重精确为零，断开"}，权重 ${String(weight)}`);
      const label = document.createElement("strong");
      label.textContent = `神经元 ${node + 1}`;
      const value = document.createElement("span");
      value.textContent = connected ? displayWeight(weight) : "断开 · 0";
      item.title = `数字 ${selectedInputDigit} → 神经元 ${node + 1} · 精确权重 ${String(weight)}`;
      item.append(label, value);
      elements.inputTargetList.append(item);
    }
  }

  function renderHistory() {
    const history = Array.isArray(currentState.history) ? currentState.history : [];
    const recent = history.slice(-12).reverse();
    elements.recentCount.textContent = String(recent.length);
    elements.recentCount.title = "最近展示的按键数量，最多 12 次";
    elements.recentKeys.replaceChildren();
    if (recent.length === 0) {
      const empty = document.createElement("span");
      empty.className = "empty-note";
      empty.textContent = "记录会出现在这里";
      elements.recentKeys.append(empty);
      return;
    }
    recent.forEach((digit, index) => {
      const chip = document.createElement("span");
      chip.className = "recent-key";
      chip.textContent = Number.isInteger(digit) ? String(digit) : "?";
      chip.setAttribute("aria-label", `${index === 0 ? "最近一次" : `之前第 ${index + 1} 次`}按键 ${chip.textContent}`);
      elements.recentKeys.append(chip);
    });
  }

  function renderPrediction() {
    const show = elements.earlyReveal.checked;
    const probabilities = currentState.probabilities.map((value) => isFiniteNumber(value) ? value : 0);
    const maxProbability = Math.max(0, ...probabilities);
    elements.predictionReadout.hidden = !show;
    elements.predictionHint.hidden = show;
    if (!show) return;
    elements.predictionValue.textContent = Number.isInteger(currentState.prediction) ? String(currentState.prediction) : "—";
    elements.probabilityList.replaceChildren();
    probabilities.forEach((value, digit) => {
      const row = document.createElement("div");
      row.className = `probability-row${digit === currentState.prediction ? " is-top" : ""}`;
      const digitLabel = document.createElement("span");
      digitLabel.className = "probability-digit";
      digitLabel.textContent = String(digit);
      const track = document.createElement("span");
      track.className = "probability-track";
      const fill = document.createElement("span");
      fill.className = "probability-fill";
      fill.style.width = `${maxProbability > 0 ? (value / maxProbability) * 100 : 0}%`;
      track.append(fill);
      const percent = document.createElement("span");
      percent.className = "probability-percent";
      percent.textContent = percentage(value);
      row.append(digitLabel, track, percent);
      elements.probabilityList.append(row);
    });
  }

  function activityColor(value, maxAbs) {
    if (!isFiniteNumber(value) || value === 0) return "#e1e9e2";
    const strength = maxAbs > 0 ? Math.abs(value) / maxAbs : 0;
    const alpha = 0.16 + strength * 0.72;
    return value > 0 ? `rgba(217, 104, 81, ${alpha})` : `rgba(84, 124, 163, ${alpha})`;
  }

  function renderNeurons() {
    const values = currentState.neurons.map((value) => isFiniteNumber(value) ? value : 0);
    const maxAbs = Math.max(0, ...values.map((value) => Math.abs(value)));
    elements.neuronGrid.replaceChildren();
    values.forEach((value, index) => {
      const item = document.createElement("div");
      item.className = "neuron-item";
      item.setAttribute("role", "listitem");
      item.setAttribute("aria-label", `神经元 ${index + 1}，活动值 ${String(currentState.neurons[index])}`);
      item.tabIndex = 0;
      item.title = `神经元 ${index + 1} · 活动值 ${String(currentState.neurons[index])}`;
      const dot = document.createElement("span");
      dot.className = "neuron-dot";
      dot.style.backgroundColor = activityColor(value, maxAbs);
      const label = document.createElement("strong");
      label.textContent = `${index + 1} · ${displayNumber(value)}`;
      item.append(dot, label);
      elements.neuronGrid.append(item);
    });
  }

  function renderWeights() {
    const matrix = currentState.connections;
    const flat = matrix.flat().filter(isFiniteNumber);
    const maxAbs = Math.max(0, ...flat.map((value) => Math.abs(value)));
    const table = document.createElement("table");
    table.id = "weightTable";
    table.className = "weight-table";
    table.setAttribute("aria-label", "20 行 20 列神经元连接权重矩阵");
    const caption = document.createElement("caption");
    caption.className = "visually-hidden";
    caption.textContent = "行是目标神经元，列是来源神经元。每个格子的说明包含精确权重值。";
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    const corner = document.createElement("th");
    corner.scope = "col";
    corner.textContent = "到\\从";
    headRow.append(corner);
    for (let col = 0; col < 20; col += 1) {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = String(col + 1);
      headRow.append(th);
    }
    head.append(headRow);
    const body = document.createElement("tbody");
    const rawRows = [];
    matrix.forEach((row, rowIndex) => {
      const tr = document.createElement("tr");
      const rowHeader = document.createElement("th");
      rowHeader.scope = "row";
      rowHeader.textContent = String(rowIndex + 1);
      tr.append(rowHeader);
      rawRows.push(row.map((value) => displayNumber(value, 6)).join("  "));
      row.forEach((value, colIndex) => {
        const cell = document.createElement("td");
        const numberText = isFiniteNumber(value) ? String(value) : "不可用";
        cell.title = `神经元 ${colIndex + 1} → 神经元 ${rowIndex + 1}：${numberText}`;
        cell.setAttribute("aria-label", `第 ${colIndex + 1} 个神经元到第 ${rowIndex + 1} 个神经元，权重 ${numberText}`);
        const strength = isFiniteNumber(value) && maxAbs > 0 ? Math.abs(value) / maxAbs : 0;
        if (isFiniteNumber(value) && value !== 0) {
          const alpha = 0.14 + strength * 0.72;
          cell.style.backgroundColor = value > 0 ? `rgba(217, 104, 81, ${alpha})` : `rgba(84, 124, 163, ${alpha})`;
        } else {
          cell.style.backgroundColor = "#f0f4f0";
        }
        tr.append(cell);
      });
      body.append(tr);
    });
    table.append(caption, head, body);
    elements.weightTable.replaceWith(table);
    elements.weightTable = table;
    elements.weightValues.textContent = rawRows.join("\n");
  }

  const SVG_NS = "http://www.w3.org/2000/svg";
  const GRAPH_CENTER = { x: 360, y: 260 };
  const GRAPH_RADIUS = 200;

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }

  function activityFill(value) {
    const neutral = [243, 247, 243];
    const endpoint = value > 0 ? [217, 104, 81] : [84, 124, 163];
    const amount = isFiniteNumber(value) ? Math.min(1, Math.abs(value)) : 0;
    const color = neutral.map((base, index) => Math.round(base + (endpoint[index] - base) * amount));
    return `rgb(${color.join(", ")})`;
  }

  function graphNodePosition(index) {
    const angle = -Math.PI / 2 + (index / 20) * Math.PI * 2;
    return {
      x: GRAPH_CENTER.x + Math.cos(angle) * GRAPH_RADIUS,
      y: GRAPH_CENTER.y + Math.sin(angle) * GRAPH_RADIUS,
      radialX: Math.cos(angle),
      radialY: Math.sin(angle),
    };
  }

  function graphEdgePath(sourceIndex, targetIndex, positions) {
    const source = positions[sourceIndex];
    const target = positions[targetIndex];
    if (sourceIndex === targetIndex) {
      const tangentX = -source.radialY;
      const tangentY = source.radialX;
      const startX = source.x - tangentX * 9 + source.radialX * 17;
      const startY = source.y - tangentY * 9 + source.radialY * 17;
      const endX = source.x + tangentX * 9 + source.radialX * 17;
      const endY = source.y + tangentY * 9 + source.radialY * 17;
      const controlDistance = 54;
      const sideDistance = 36;
      const firstControlX = source.x - tangentX * sideDistance + source.radialX * controlDistance;
      const firstControlY = source.y - tangentY * sideDistance + source.radialY * controlDistance;
      const secondControlX = source.x + tangentX * sideDistance + source.radialX * controlDistance;
      const secondControlY = source.y + tangentY * sideDistance + source.radialY * controlDistance;
      return `M ${startX.toFixed(1)} ${startY.toFixed(1)} C ${firstControlX.toFixed(1)} ${firstControlY.toFixed(1)}, ${secondControlX.toFixed(1)} ${secondControlY.toFixed(1)}, ${endX.toFixed(1)} ${endY.toFixed(1)}`;
    }

    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const distance = Math.hypot(dx, dy);
    const unitX = dx / distance;
    const unitY = dy / distance;
    const startX = source.x + unitX * 23;
    const startY = source.y + unitY * 23;
    const endX = target.x - unitX * 27;
    const endY = target.y - unitY * 27;
    const lowIndex = Math.min(sourceIndex, targetIndex);
    const highIndex = Math.max(sourceIndex, targetIndex);
    const pairHash = lowIndex * 19 + highIndex * 31;
    const bend = (pairHash % 2 ? 1 : -1) * (12 + (pairHash % 5) * 3);
    const controlX = (startX + endX) / 2 - unitY * bend;
    const controlY = (startY + endY) / 2 + unitX * bend;
    return `M ${startX.toFixed(1)} ${startY.toFixed(1)} Q ${controlX.toFixed(1)} ${controlY.toFixed(1)}, ${endX.toFixed(1)} ${endY.toFixed(1)}`;
  }

  function appendConnectionList(list, connections, selectedIndex, direction) {
    const rows = [];
    for (let otherIndex = 0; otherIndex < 20; otherIndex += 1) {
      const weight = direction === "incoming"
        ? connections[selectedIndex][otherIndex]
        : connections[otherIndex][selectedIndex];
      if (!isFiniteNumber(weight) || weight === 0) continue;
      rows.push({ otherIndex, weight });
    }
    rows.sort((left, right) => Math.abs(right.weight) - Math.abs(left.weight));
    list.replaceChildren();
    if (rows.length === 0) {
      const empty = document.createElement("li");
      empty.className = "no-connections";
      empty.textContent = "暂无非零连接";
      list.append(empty);
      return 0;
    }
    rows.forEach(({ otherIndex, weight }) => {
      const item = document.createElement("li");
      item.className = weight > 0 ? "connection-positive" : "connection-negative";
      item.textContent = direction === "incoming"
        ? `神经元 ${otherIndex + 1} → ${selectedIndex + 1}　${String(weight)}`
        : `神经元 ${selectedIndex + 1} → ${otherIndex + 1}　${String(weight)}`;
      item.title = item.textContent;
      list.append(item);
    });
    return rows.length;
  }

  function renderNeuronInspector() {
    const selectedIndex = selectedNeuronIndex;
    const selected = selectedIndex !== null;
    elements.graphSelectionEmpty.hidden = selected;
    elements.graphInspector.hidden = !selected;
    elements.clearNeuronSelection.disabled = !selected;
    if (!selected) return;

    const outputText = selectedIndex < 10 ? ` · 输出数字 ${selectedIndex}` : "";
    const activity = currentState.neurons[selectedIndex];
    elements.selectedNeuronTitle.textContent = `神经元 ${selectedIndex + 1}${outputText}`;
    elements.selectedNeuronActivity.textContent = isFiniteNumber(activity) ? String(activity) : "不可用";
    const connections = currentState.connections;
    elements.selectedIncomingCount.textContent = String(appendConnectionList(
      elements.selectedIncoming, connections, selectedIndex, "incoming",
    ));
    elements.selectedOutgoingCount.textContent = String(appendConnectionList(
      elements.selectedOutgoing, connections, selectedIndex, "outgoing",
    ));
  }

  function renderNetworkGraph(focusNodeIndex = null) {
    if (!currentState) return;
    const focusedNode = document.activeElement && document.activeElement.closest
      ? document.activeElement.closest(".graph-node")
      : null;
    if (focusNodeIndex === null && focusedNode && elements.neuronGraph.contains(focusedNode)) {
      focusNodeIndex = Number(focusedNode.dataset.nodeIndex);
    }

    const matrix = currentState.connections;
    const finiteWeights = matrix.flat().filter(isFiniteNumber);
    const maxAbs = Math.max(0, ...finiteWeights.map((value) => Math.abs(value)));
    const thresholdPercent = Number(elements.edgeThreshold.value);
    const threshold = maxAbs * thresholdPercent / 100;
    elements.edgeThresholdValue.textContent = `${thresholdPercent}% × 最大值（${displayNumber(threshold, 5)}）`;
    elements.graphStep.textContent = String(currentState.step);
    elements.graphCenterStep.textContent = `第 ${currentState.step} 次按键后`;

    const positions = Array.from({ length: 20 }, (_value, index) => graphNodePosition(index));
    const visibleEdges = [];
    for (let targetIndex = 0; targetIndex < 20; targetIndex += 1) {
      for (let sourceIndex = 0; sourceIndex < 20; sourceIndex += 1) {
        const weight = matrix[targetIndex][sourceIndex];
        if (!isFiniteNumber(weight) || weight === 0) continue;
        const passesThreshold = Math.abs(weight) >= threshold;
        const touchesSelection = selectedNeuronIndex !== null &&
          (sourceIndex === selectedNeuronIndex || targetIndex === selectedNeuronIndex);
        if (passesThreshold || touchesSelection) {
          visibleEdges.push({ sourceIndex, targetIndex, weight });
        }
      }
    }

    elements.graphEdges.replaceChildren();
    visibleEdges.forEach(({ sourceIndex, targetIndex, weight }) => {
      const selectedLink = selectedNeuronIndex !== null &&
        (sourceIndex === selectedNeuronIndex || targetIndex === selectedNeuronIndex);
      const edge = svgElement("path", {
        class: `graph-edge ${weight > 0 ? "is-positive" : "is-negative"}${selectedLink ? " is-selected-link" : ""}`,
        d: graphEdgePath(sourceIndex, targetIndex, positions),
        "marker-end": weight > 0 ? "url(#arrow-positive)" : "url(#arrow-negative)",
        "aria-hidden": "true",
      });
      const strength = maxAbs > 0 ? Math.abs(weight) / maxAbs : 0;
      edge.style.strokeWidth = String(0.8 + strength * 1.8 + (selectedLink ? 0.8 : 0));
      edge.style.opacity = selectedLink ? "0.9" : String(selectedNeuronIndex !== null ? 0.055 : 0.2 + strength * 0.47);
      const title = svgElement("title");
      title.textContent = `神经元 ${sourceIndex + 1} → 神经元 ${targetIndex + 1} · 权重 ${String(weight)}`;
      edge.append(title);
      elements.graphEdges.append(edge);
    });
    const edgeCountText = `显示 ${visibleEdges.length} / 400 条有向连接${selectedNeuronIndex === null
      ? ` · |权重| ≥ ${displayNumber(threshold, 5)}`
      : " · 选中节点的全部进出连接均显示"}`;
    elements.graphEdgeCount.textContent = edgeCountText;

    elements.graphNodes.replaceChildren();
    currentState.neurons.forEach((rawValue, index) => {
      const value = isFiniteNumber(rawValue) ? rawValue : 0;
      const position = positions[index];
      const selected = selectedNeuronIndex === index;
      const group = svgElement("g", {
        class: `graph-node${index < 10 ? " is-output" : ""}${selected ? " is-selected" : ""}`,
        transform: `translate(${position.x.toFixed(1)} ${position.y.toFixed(1)})`,
        "data-node-index": index,
        role: "button",
        tabindex: "0",
        "aria-pressed": selected ? "true" : "false",
        "aria-label": `神经元 ${index + 1}${index < 10 ? `，输出数字 ${index}` : ""}，活动值 ${isFiniteNumber(rawValue) ? String(rawValue) : "不可用"}`,
      });
      const nodeTitle = svgElement("title");
      nodeTitle.textContent = `神经元 ${index + 1}${index < 10 ? ` · 输出数字 ${index}` : ""} · 活动值 ${isFiniteNumber(rawValue) ? String(rawValue) : "不可用"}`;
      const circle = svgElement("circle", { class: "graph-node-circle", r: "24" });
      circle.style.fill = activityFill(value);
      const number = svgElement("text", { class: "graph-node-number", x: "0", y: index < 10 ? "-3" : "5", "text-anchor": "middle" });
      number.textContent = String(index + 1);
      group.append(nodeTitle, circle, number);
      if (index < 10) {
        const output = svgElement("text", { class: "graph-node-output", x: "0", y: "12", "text-anchor": "middle" });
        output.textContent = `输出 ${index}`;
        group.append(output);
      }
      elements.graphNodes.append(group);
    });

    renderNeuronInspector();
    if (focusNodeIndex !== null && Number.isInteger(focusNodeIndex)) {
      const focusTarget = elements.graphNodes.querySelector(`[data-node-index="${focusNodeIndex}"]`);
      if (focusTarget) focusTarget.focus({ preventScroll: true });
    }
  }

  function selectNeuron(index) {
    selectedNeuronIndex = selectedNeuronIndex === index ? null : index;
    renderNetworkGraph(index);
  }

  function render() {
    elements.sessionChip.hidden = false;
    elements.sessionId.textContent = currentState.session_id;
    elements.sessionId.title = currentState.session_id;
    elements.stepValue.textContent = String(currentState.step);
    elements.footerSession.textContent = `会话 ${currentState.session_id.slice(0, 8)} · ${currentState.step} 次按键`;
    renderResult();
    renderMetrics();
    renderHistory();
    renderModelSummary();
    renderPrediction();
    renderNeurons();
    renderWeights();
    renderNetworkGraph();
    elements.digitButtons.forEach((button) => { button.disabled = busy; });
    elements.resetButton.disabled = busy;
  }

  async function pressDigit(digit) {
    if (busy || !currentState || !Number.isInteger(digit) || digit < 0 || digit > 9) return;
    const predictionId = currentState.prediction_id;
    if (!predictionId) {
      showMessage("找不到当前预测，请稍等片刻同步状态。", true);
      try { await loadState({ keepMessage: true }); } catch (_error) { /* keep the original message */ }
      return;
    }
    showMessage("");
    setBusy(true, "正在学习…");
    try {
      const state = await requestJson("/api/press", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ digit, prediction_id: predictionId }),
      });
      if (!validState(state)) throw new Error("本地服务返回的状态格式不完整。");
      currentState = state;
      render();
      setBusy(false);
      const result = state.last_result;
      showMessage(result && result.correct ? `猜中了！你按了 ${digit}。` : `你按了 ${digit}，已揭晓上一轮预测。`);
    } catch (error) {
      if (error.status === 409) {
        try {
          await loadState({ keepMessage: true });
          showMessage("这一轮预测已变化，状态已同步；请再按一次。", true);
        } catch (syncError) {
          setBusy(false);
          setConnection("同步失败", "error");
          showMessage(`预测已变化，但同步失败：${syncError.message}`, true);
        }
      } else {
        setBusy(false);
        setConnection("请求失败", "error");
        showMessage(error.message || "按键没有提交成功，请重试。", true);
      }
    }
  }

  async function startNewSession() {
    if (busy) return;
    showMessage("");
    setBusy(true, "正在开始新会话…");
    try {
      const state = await requestJson("/api/reset", { method: "POST" });
      if (!validState(state)) throw new Error("本地服务返回的状态格式不完整。");
      currentState = state;
      render();
      setBusy(false);
      showMessage("新会话已开始；旧会话记录仍保存在本地，导出时会一并包含。");
    } catch (error) {
      setBusy(false);
      setConnection("请求失败", "error");
      showMessage(error.message || "暂时无法开始新会话。", true);
    }
  }

  async function pollState() {
    if (!currentState || busy || document.hidden || pollInFlight) return;
    pollInFlight = true;
    const revision = stateRevision;
    try {
      const state = await requestJson("/api/state");
      if (revision !== stateRevision || busy || !validState(state)) return;
      if (state.prediction_id !== currentState.prediction_id) {
        currentState = state;
        render();
        setBusy(false);
      }
    } catch (_error) {
      // State polling is read-only and silent; the next interval can recover.
    } finally {
      pollInFlight = false;
    }
  }

  elements.digitButtons.forEach((button) => {
    button.addEventListener("click", () => pressDigit(Number(button.dataset.digit)));
  });

  elements.earlyReveal.addEventListener("change", renderPrediction);
  elements.inputDigitSelect.addEventListener("change", () => {
    const digit = Number(elements.inputDigitSelect.value);
    if (!Number.isInteger(digit) || digit < 0 || digit > 9) return;
    selectedInputDigit = digit;
    inputDigitManuallySelected = true;
    renderModelSummary();
  });
  elements.graphZoom.addEventListener("click", () => {
    const zoomed = elements.neuronGraph.classList.toggle("is-zoomed");
    elements.graphZoom.setAttribute("aria-pressed", String(zoomed));
    elements.graphZoom.textContent = zoomed ? "显示全图" : "放大查看";
  });
  elements.edgeThreshold.addEventListener("input", () => renderNetworkGraph());
  elements.clearNeuronSelection.addEventListener("click", () => {
    selectedNeuronIndex = null;
    renderNetworkGraph();
  });
  elements.neuronGraph.addEventListener("click", (event) => {
    const target = event.target && event.target.closest ? event.target.closest(".graph-node") : null;
    if (!target || !elements.neuronGraph.contains(target)) return;
    selectNeuron(Number(target.dataset.nodeIndex));
  });
  elements.neuronGraph.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && selectedNeuronIndex !== null) {
      event.preventDefault();
      const focusedNode = event.target && event.target.closest ? event.target.closest(".graph-node") : null;
      selectedNeuronIndex = null;
      renderNetworkGraph(focusedNode ? Number(focusedNode.dataset.nodeIndex) : null);
      return;
    }
    const target = event.target && event.target.closest ? event.target.closest(".graph-node") : null;
    if (target && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      selectNeuron(Number(target.dataset.nodeIndex));
    }
  });
  elements.resetButton.addEventListener("click", startNewSession);
  document.addEventListener("keydown", (event) => {
    if (event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
    const target = event.target;
    if (target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) return;
    if (/^[0-9]$/.test(event.key)) {
      event.preventDefault();
      pressDigit(Number(event.key));
    }
  });

  loadState().then(() => {
    window.setInterval(pollState, 1500);
  }).catch((error) => {
    setConnection("无法连接", "error");
    elements.resetButton.disabled = true;
    elements.digitButtons.forEach((button) => { button.disabled = true; });
    showMessage(`${error.message} 请确认本地训练服务已经启动，再刷新页面。`, true);
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) pollState();
  });
})();
