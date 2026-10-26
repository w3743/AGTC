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
  };

  let currentState = null;
  let busy = false;

  function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function displayNumber(value, places = 3) {
    if (!isFiniteNumber(value)) return "—";
    return value.toFixed(places);
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
        : predictionSource === "plastic_rnn"
          ? "训练后的神经网络预测。"
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

  function render() {
    elements.sessionChip.hidden = false;
    elements.sessionId.textContent = currentState.session_id;
    elements.sessionId.title = currentState.session_id;
    elements.stepValue.textContent = String(currentState.step);
    elements.footerSession.textContent = `会话 ${currentState.session_id.slice(0, 8)} · ${currentState.step} 次按键`;
    renderResult();
    renderMetrics();
    renderHistory();
    renderPrediction();
    renderNeurons();
    renderWeights();
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

  elements.digitButtons.forEach((button) => {
    button.addEventListener("click", () => pressDigit(Number(button.dataset.digit)));
  });

  elements.earlyReveal.addEventListener("change", renderPrediction);
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

  loadState().catch((error) => {
    setConnection("无法连接", "error");
    elements.resetButton.disabled = true;
    elements.digitButtons.forEach((button) => { button.disabled = true; });
    showMessage(`${error.message} 请确认本地训练服务已经启动，再刷新页面。`, true);
  });
})();
