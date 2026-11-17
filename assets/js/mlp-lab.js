/* ============================================================
   MLP 实验室：控件、训练循环、四块画布
   ============================================================ */
(function () {
  "use strict";
  var E = window.MLPEngine;
  if (!E) return;

  /* ---------------- 状态 ---------------- */
  var state = {
    dataset: "circle",
    hidden: 1,
    neurons: 6,
    act: "relu",
    lr: 0.1,
    reg: 0,
    batch: 16,
    opt: "sgd",
    noise: 0.08,
    speed: 6,
    running: false,
  };

  var net = null;
  var points = [];
  var grid = null;
  var hist = [];
  var probe = { x: 0.35, y: 0.15 };
  var acc = 0;
  var last = 0;
  var lastDraw = 0;
  var lastGrid = 0;
  var gridDirty = true;
  var res = 48;
  var GRID_MS = 120;

  var $ = function (s) { return document.querySelector(s); };

  /* 画布配色：跟着主题走（默认浅米色，可切深色） */
  function palette() {
    if (document.documentElement.getAttribute("data-theme") === "dark") {
      return {
        dark: true,
        bg: "#0a0d12",
        c0: [106, 169, 255], c1: [246, 185, 92], mid: [20, 26, 34],
        pos: "94,234,212", neg: "248,113,113",
        nodeFill: "#0d1117", nodeStroke: "rgba(255,255,255,0.22)", outStroke: "rgba(246,185,92,0.8)",
        label: "rgba(231,234,240,0.55)", sub: "rgba(148,158,172,0.8)",
        grid: "rgba(255,255,255,0.06)", line: "#5eead4", acc: "rgba(246,185,92,0.55)",
        pt0: "#bcd8ff", pt1: "#ffd79a", ptStroke: "rgba(6,8,11,0.75)", ink: "#ffffff",
        empty: "rgba(148,158,172,0.55)",
      };
    }
    return {
      dark: false,
      bg: "#fffaf3",
      c0: [136, 180, 255], c1: [255, 166, 136], mid: [255, 250, 243],
      pos: "15,156,139", neg: "232,96,60",
      nodeFill: "#ffffff", nodeStroke: "rgba(63,47,34,0.22)", outStroke: "rgba(217,130,27,0.8)",
      label: "rgba(72,58,46,0.7)", sub: "rgba(120,104,88,0.85)",
      grid: "rgba(63,47,34,0.1)", line: "#0f9c8b", acc: "rgba(232,96,60,0.6)",
      pt0: "#2f5f9e", pt1: "#b4472a", ptStroke: "rgba(255,255,255,0.9)", ink: "#2e2721",
      empty: "rgba(120,104,88,0.65)",
    };
  }

  var el = {
    dataset: $("#ctlDataset"), hidden: $("#ctlHidden"), neurons: $("#ctlNeurons"),
    act: $("#ctlAct"), lr: $("#ctlLr"), lrOut: $("#ctlLrOut"),
    reg: $("#ctlReg"), regOut: $("#ctlRegOut"), batch: $("#ctlBatch"),
    opt: $("#ctlOpt"), noise: $("#ctlNoise"), noiseOut: $("#ctlNoiseOut"),
    speed: $("#ctlSpeed"), speedOut: $("#ctlSpeedOut"),
    run: $("#btnRun"), step: $("#btnStep"), reset: $("#btnReset"),
    hint: $("#hint"),
    net: $("#netCanvas"), boundary: $("#boundaryCanvas"),
    loss: $("#lossCanvas"), weights: $("#wCanvas"),
    statStep: $("#statStep"), statReal: $("#statReal"), statLoss: $("#statLoss"),
    statAcc: $("#statAcc"), statParams: $("#statParams"), probeVal: $("#probeVal"),
  };
  if (!el.net) return;

  /* ---------------- 画布尺寸 ---------------- */
  function fit(canvas, cssH) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = canvas.clientWidth || canvas.parentElement.clientWidth;
    if (!w) return null;
    var h = cssH || canvas.clientHeight || w;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.height = h + "px";
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: w, h: h, ctx: ctx };
  }

  var sizes = { net: null, boundary: null, loss: null, weights: null };

  function measure() {
    var nw = el.net.clientWidth || el.net.parentElement.clientWidth;
    sizes.net = fit(el.net, Math.max(190, Math.min(280, nw * 0.52)));
    var bw = el.boundary.clientWidth || el.boundary.parentElement.clientWidth;
    sizes.boundary = fit(el.boundary, Math.min(bw, 430));
    sizes.loss = fit(el.loss, 132);
    var ww = el.weights.clientWidth || el.weights.parentElement.clientWidth;
    sizes.weights = fit(el.weights, Math.max(70, Math.min(150, ww * 0.45)));
  }

  /* ---------------- 构建 ---------------- */
  function rebuild() {
    var hidden = [];
    for (var i = 0; i < state.hidden; i++) hidden.push(state.neurons);
    var sz = [2].concat(hidden, [1]);
    net = new E.MLP({ sizes: sz, act: state.act, lr: state.lr, reg: state.reg, opt: state.opt });
    points = E.makeDataset(state.dataset, 220, state.noise, 20260417);
    hist = [];
    gridDirty = true;
    el.statParams.textContent = String(net.paramCount());
    el.hint.textContent = E.DATASETS[state.dataset].hint;
  }

  /* ---------------- 训练 ---------------- */
  function trainOnce() {
    var idx = [];
    for (var i = 0; i < state.batch; i++) idx.push(Math.floor(Math.random() * points.length));
    net.trainBatch(points, idx);
    if (net.steps % 4 === 0 || hist.length === 0) {
      var ev = net.evaluate(points);
      hist.push({ s: net.steps, loss: ev.loss, acc: ev.acc });
      if (hist.length > 900) hist.shift();
    }
    gridDirty = true;
  }

  function updateGrid() {
    grid = net.evalGrid(res);
    gridDirty = false;
  }

  /* ---------------- 决策边界 ---------------- */
  var off = null;

  function drawBoundary() {
    var s = sizes.boundary;
    if (!s) return;
    var ctx = s.ctx, w = s.w, h = s.h;
    var P = palette();

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = P.bg;
    ctx.fillRect(0, 0, w, h);

    if (grid) {
      if (!off) {
        off = document.createElement("canvas");
        off.width = res;
        off.height = res;
      }
      var o = off.getContext("2d");
      var img = o.createImageData(res, res);
      var d = img.data;
      var cB = P.c0, cA = P.c1, m = P.mid;
      for (var i = 0; i < grid.length; i++) {
        var t = grid[i], rr, gg, bb;
        if (t < 0.5) {
          var k = t / 0.5;
          rr = m[0] + (cB[0] - m[0]) * k;
          gg = m[1] + (cB[1] - m[1]) * k;
          bb = m[2] + (cB[2] - m[2]) * k;
        } else {
          var k2 = (t - 0.5) / 0.5;
          rr = m[0] + (cA[0] - m[0]) * k2;
          gg = m[1] + (cA[1] - m[1]) * k2;
          bb = m[2] + (cA[2] - m[2]) * k2;
        }
        var j = i * 4;
        d[j] = rr; d[j + 1] = gg; d[j + 2] = bb; d[j + 3] = 255;
      }
      o.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, w, h);
    }

    // 十字参考线
    ctx.strokeStyle = P.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
    ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
    ctx.stroke();

    // 样本点
    for (var k = 0; k < points.length; k++) {
      var pt = points[k];
      var px = ((pt.x + 1) / 2) * w;
      var py = ((1 - pt.y) / 2) * h;
      ctx.beginPath();
      ctx.arc(px, py, 2.7, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 1 ? P.pt1 : P.pt0;
      ctx.globalAlpha = 0.95;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1;
      ctx.strokeStyle = P.ptStroke;
      ctx.stroke();
    }

    // 探针
    var qx = ((probe.x + 1) / 2) * w, qy = ((1 - probe.y) / 2) * h;
    ctx.beginPath();
    ctx.arc(qx, qy, 9, 0, Math.PI * 2);
    ctx.strokeStyle = P.ink;
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(qx, qy, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = P.ink;
    ctx.fill();
  }

  /* ---------------- 网络图 ---------------- */
  function drawNet() {
    var s = sizes.net;
    if (!s) return;
    var ctx = s.ctx, w = s.w, h = s.h;
    ctx.clearRect(0, 0, w, h);

    var cache = net.forward([probe.x, probe.y]);
    var a = cache.a;
    var padX = 46, padY = 24;
    var bottom = h - 26;
    var L = net.sizes.length;
    var xAt = function (i) { return padX + (i * (w - padX * 2)) / (L - 1); };

    var maxN = Math.max.apply(null, net.sizes);
    var r = Math.max(6.5, Math.min(15, (bottom - padY) / (maxN * 2.6)));
    var yAt = function (l, i) {
      var n = net.sizes[l];
      var span = bottom - padY;
      if (n === 1) return (padY + bottom) / 2;
      return padY + i * (span / (n - 1));
    };

    // 边：颜色表示正负，粗细表示绝对值
    var P = palette();
    for (var l = 1; l < L; l++) {
      var W = net.W[l];
      var mx = 0;
      for (var i0 = 0; i0 < W.length; i0++) for (var j0 = 0; j0 < W[i0].length; j0++) mx = Math.max(mx, Math.abs(W[i0][j0]));
      mx = mx || 1;
      for (var i = 0; i < W.length; i++) {
        for (var j = 0; j < W[i].length; j++) {
          var wv = W[i][j];
          var mag = Math.abs(wv) / mx;
          var alpha = 0.08 + mag * (P.dark ? 0.6 : 0.7);
          ctx.strokeStyle = "rgba(" + (wv >= 0 ? P.pos : P.neg) + "," + alpha + ")";
          ctx.lineWidth = 0.5 + mag * 2.6;
          ctx.beginPath();
          ctx.moveTo(xAt(l - 1), yAt(l - 1, j));
          ctx.lineTo(xAt(l), yAt(l, i));
          ctx.stroke();
        }
      }
    }

    // 节点：亮度表示当前探针样本通过时的激活值
    for (var l2 = 0; l2 < L; l2++) {
      for (var i2 = 0; i2 < net.sizes[l2]; i2++) {
        var x = xAt(l2), y = yAt(l2, i2);
        var act = a[l2] ? a[l2][i2] : 0;
        var isOut = l2 === L - 1;
        var isIn = l2 === 0;
        var mag2 = Math.min(1, Math.abs(act) / (isOut ? 1 : 2.2));

        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = P.nodeFill;
        ctx.fill();

        if (mag2 > 0.02) {
          ctx.beginPath();
          ctx.arc(x, y, r * (0.35 + 0.65 * mag2), 0, Math.PI * 2);
          ctx.fillStyle = isOut
            ? "rgba(" + P.neg + "," + (0.18 + 0.6 * mag2) + ")"
            : "rgba(" + P.pos + "," + (0.12 + 0.5 * mag2) + ")";
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.strokeStyle = isOut ? P.outStroke : P.nodeStroke;
        ctx.lineWidth = isIn ? 1 : 1.2;
        ctx.stroke();

        if (isIn) {
          ctx.fillStyle = P.label;
          ctx.font = "10px ui-monospace, Consolas, monospace";
          ctx.textAlign = "right";
          ctx.textBaseline = "middle";
          ctx.fillText(i2 === 0 ? "x\u2081" : "x\u2082", x - r - 6, y);
        }
      }
    }

    // 层标签
    ctx.font = "10px ui-monospace, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = P.sub;
    for (var l3 = 0; l3 < L; l3++) {
      var name = l3 === 0 ? "输入" : l3 === L - 1 ? "输出" : "隐藏 " + l3;
      ctx.fillText(name, xAt(l3), bottom + 8);
    }

    var p = a[L - 1][0];
    el.probeVal.textContent = p.toFixed(3);
    el.probeVal.style.color = p >= 0.5 ? "var(--amber)" : "var(--blue)";
  }

  /* ---------------- 损失曲线 ---------------- */
  function drawLoss() {
    var s = sizes.loss;
    if (!s) return;
    var ctx = s.ctx, w = s.w, h = s.h;
    var padL = 30, padR = 30, padT = 12, padB = 16;
    ctx.clearRect(0, 0, w, h);
    var P = palette();

    if (!hist.length) {
      ctx.fillStyle = P.empty;
      ctx.font = "11px ui-monospace, Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("点「开始训练」", w / 2, h / 2);
      return;
    }

    var maxLoss = 0.02;
    for (var i = 0; i < hist.length; i++) maxLoss = Math.max(maxLoss, hist[i].loss);
    maxLoss = Math.min(maxLoss, 2.5) * 1.12;

    var X = function (i) { return padL + (i / Math.max(1, hist.length - 1)) * (w - padL - padR); };
    var Y = function (v) { return padT + (1 - Math.min(1, v / maxLoss)) * (h - padT - padB); };

    ctx.strokeStyle = P.grid;
    ctx.lineWidth = 1;
    for (var g = 0; g <= 2; g++) {
      var yy = padT + (g / 2) * (h - padT - padB);
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
    }

    ctx.beginPath();
    for (var k = 0; k < hist.length; k++) {
      var x = X(k), y = Y(hist[k].loss);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = P.line;
    ctx.lineWidth = 1.8;
    ctx.stroke();

    ctx.beginPath();
    for (var k2 = 0; k2 < hist.length; k2++) {
      var x2 = X(k2), y2 = padT + (1 - hist[k2].acc) * (h - padT - padB);
      if (k2 === 0) ctx.moveTo(x2, y2); else ctx.lineTo(x2, y2);
    }
    ctx.strokeStyle = P.acc;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "10px ui-monospace, Consolas, monospace";
    ctx.textBaseline = "middle";
    ctx.fillStyle = P.sub;
    ctx.textAlign = "right";
    ctx.fillText(maxLoss.toFixed(2), padL - 6, padT + 4);
    ctx.fillText("0", padL - 6, h - padB);
  }

  /* ---------------- 权重热图 ---------------- */
  function drawWeights() {
    var s = sizes.weights;
    if (!s) return;
    var ctx = s.ctx, w = s.w, h = s.h;
    ctx.clearRect(0, 0, w, h);
    var P = palette();

    var layers = net.sizes.length - 1;
    var colW = w / layers;
    var mx = 0;
    for (var l = 1; l <= layers; l++) {
      for (var i = 0; i < net.W[l].length; i++) {
        for (var j = 0; j < net.W[l][i].length; j++) mx = Math.max(mx, Math.abs(net.W[l][i][j]));
      }
    }
    mx = Math.max(mx, 1e-6);

    for (var l2 = 1; l2 <= layers; l2++) {
      var W = net.W[l2];
      var rows = W.length, cols = W[0].length;
      var cell = Math.min(13, (colW - 14) / cols, (h - 18) / rows);
      var ox = (l2 - 1) * colW + (colW - cols * cell) / 2;
      var oy = (h - 12 - rows * cell) / 2;

      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          var v = W[r][c] / mx;
          var al = 0.12 + 0.8 * Math.min(1, Math.abs(v));
          ctx.fillStyle = "rgba(" + (v >= 0 ? P.pos : P.neg) + "," + al + ")";
          ctx.fillRect(ox + c * cell, oy + r * cell, cell - 1.2, cell - 1.2);
        }
      }
      ctx.fillStyle = P.sub;
      ctx.font = "9px ui-monospace, Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText("W" + l2, (l2 - 1) * colW + colW / 2, h - 1);
    }
  }

  /* ---------------- 统计 ---------------- */
  function drawStats() {
    var ev = net.evaluate(points);
    el.statStep.textContent = String(net.steps);
    el.statLoss.textContent = ev.loss.toFixed(3);
    el.statAcc.textContent = (ev.acc * 100).toFixed(1) + "%";
    el.statParams.textContent = String(net.paramCount());
    el.statReal.textContent = ((net.steps * state.batch) / points.length).toFixed(1);
  }

  /* ---------------- 主循环 ---------------- */
  function frame(ts) {
    if (!last) last = ts;
    var dt = Math.min(0.1, (ts - last) / 1000);
    last = ts;

    if (state.running) {
      acc += dt * state.speed;
      var guard = 0;
      while (acc >= 1 && guard < 20) { trainOnce(); acc -= 1; guard++; }
      if (guard >= 20) acc = 0;
    }

    if (gridDirty && ts - lastGrid > GRID_MS) {
      updateGrid();
      lastGrid = ts;
    }

    if (ts - lastDraw > 55) {
      lastDraw = ts;
      drawNet();
      drawLoss();
      drawWeights();
      drawStats();
    }
    drawBoundary();

    requestAnimationFrame(frame);
  }

  /* ---------------- 控件 ---------------- */
  function syncOutputs() {
    el.lrOut.textContent = state.lr < 0.01 ? state.lr.toFixed(4) : state.lr.toFixed(3);
    el.regOut.textContent = state.reg === 0 ? "关" : state.reg.toFixed(4);
    el.noiseOut.textContent = state.noise.toFixed(2);
    el.speedOut.textContent = state.speed + " 步/秒";
    setFill(el.lr); setFill(el.reg); setFill(el.noise); setFill(el.speed);
  }

  function setFill(input) {
    if (!input) return;
    var min = parseFloat(input.min), max = parseFloat(input.max), v = parseFloat(input.value);
    var pct = ((v - min) / (max - min)) * 100;
    input.style.setProperty("--fill", pct + "%");
  }

  function bindRange(input, key, toVal, after) {
    if (!input) return;
    input.addEventListener("input", function () {
      state[key] = toVal(parseFloat(input.value));
      syncOutputs();
      if (after) after();
    });
  }

  function lrFromSlider(v) { return Math.pow(10, -3 + (v / 100) * 3); }

  el.lr.value = "67";
  state.lr = lrFromSlider(67);
  bindRange(el.lr, "lr", lrFromSlider, function () { if (net) net.lr = state.lr; });
  bindRange(el.reg, "reg", function (v) { return v / 1000; }, function () { if (net) net.reg = state.reg; });
  bindRange(el.noise, "noise", function (v) { return v / 100; }, function () { rebuild(); });
  bindRange(el.speed, "speed", function (v) { return Math.round(v); });

  if (el.dataset) el.dataset.addEventListener("change", function () { state.dataset = el.dataset.value; rebuild(); });
  if (el.hidden) el.hidden.addEventListener("change", function () { state.hidden = parseInt(el.hidden.value, 10); rebuild(); });
  if (el.neurons) el.neurons.addEventListener("change", function () { state.neurons = parseInt(el.neurons.value, 10); rebuild(); });
  if (el.act) el.act.addEventListener("change", function () { state.act = el.act.value; rebuild(); });
  if (el.batch) el.batch.addEventListener("change", function () { state.batch = parseInt(el.batch.value, 10); });
  if (el.opt) el.opt.addEventListener("change", function () {
    state.opt = el.opt.value;
    if (net) { net.opt = state.opt; net.init(); hist = []; gridDirty = true; }
  });

  function setRunning(on) {
    state.running = on;
    el.run.textContent = on ? "暂停" : "开始训练";
    el.run.classList.toggle("primary", !on);
    el.run.setAttribute("aria-pressed", on ? "true" : "false");
  }

  el.run.addEventListener("click", function () { setRunning(!state.running); });
  el.step.addEventListener("click", function () { setRunning(false); for (var i = 0; i < 5; i++) trainOnce(); });
  el.reset.addEventListener("click", function () { setRunning(false); rebuild(); });

  // 鼠标在决策边界上移动 = 探针，网络图会显示这个点被每个神经元看成了什么
  var b = el.boundary;
  function setProbe(cx, cy) {
    var rect = b.getBoundingClientRect();
    probe.x = Math.max(-1, Math.min(1, ((cx - rect.left) / rect.width) * 2 - 1));
    probe.y = Math.max(-1, Math.min(1, 1 - ((cy - rect.top) / rect.height) * 2));
  }
  b.addEventListener("mousemove", function (e) { setProbe(e.clientX, e.clientY); });
  b.addEventListener("touchmove", function (e) {
    var t = e.touches[0];
    if (!t) return;
    setProbe(t.clientX, t.clientY);
    e.preventDefault();
  }, { passive: false });

  window.addEventListener("resize", function () {
    measure();
    gridDirty = true;
  });

  /* ---------------- 代码 tabs ---------------- */
  var CODE = {
    forward:
      "// 前向：z = W·a + b，再过激活函数\n" +
      "forward(x) {\n" +
      "  const a = [x];            // a[0] 就是输入 [x₁, x₂]\n" +
      "  for (let l = 1; l < L; l++) {\n" +
      "    const z = W[l].map(row => dot(row, a[l-1]) + b[l]);\n" +
      "    a[l] = (l === L-1) ? sigmoid(z) : act(z);\n" +
      "  }\n" +
      "  return a;                 // a[L-1][0] 就是预测概率\n" +
      "}",
    backward:
      "// 反向：从输出层把误差一路传回去\n" +
      "backward(a, y) {\n" +
      "  let delta = [null];\n" +
      "  // 输出层用 sigmoid + 二元交叉熵，梯度恰好是 ŷ - y\n" +
      "  delta[L] = a[L].map(p => p - y);\n" +
      "  for (let l = L; l > 1; l--) {\n" +
      "    gW[l] += outer(delta[l], a[l-1]);   // ∂L/∂W\n" +
      "    gB[l] += delta[l];                  // ∂L/∂b\n" +
      "    const da = W[l].T.dot(delta[l]);\n" +
      "    delta[l-1] = da.mul(actPrime(z[l-1], a[l-1]));\n" +
      "  }\n" +
      "}",
    update:
      "// 更新：沿负梯度走一小步，lr 决定步子多大\n" +
      "for (const p of params) {\n" +
      "  p.w -= lr * (p.grad + reg * p.w);   // L2 正则拽一下权重\n" +
      "}\n" +
      "\n" +
      "// Adam 会把梯度按一阶/二阶动量归一化：\n" +
      "m = 0.9*m + 0.1*g;\n" +
      "v = 0.999*v + 0.001*g*g;\n" +
      "p.w -= lr * (m/(1-0.9^t)) / (sqrt(v/(1-0.999^t)) + 1e-8);",
    params:
      "// 参数就是这些数字：权重矩阵 W 和偏置 b\n" +
      "sizes = [2, 6, 1]\n" +
      "W[1] 是 6×2 矩阵   // 12 个权重：每个隐藏神经元\n" +
      "                  // 对 x₁、x₂ 各自的在意程度\n" +
      "b[1] 是长度 6 向量  // 6 个偏置：神经元的初始倾向\n" +
      "W[2] 是 1×6 矩阵   // 6 个权重：隐藏层投票给输出\n" +
      "b[2] 是 1 个数     // 输出的初始倾向\n" +
      "\n" +
      "// 一共 12 + 6 + 6 + 1 = 25 个可训练参数\n" +
      "// 训练 = 反复微调这 25 个数，让 loss 变小",
  };

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function highlight(src) {
    var s = esc(src);
    s = s.replace(/(\/\/[^\n]*)/g, '<span class="c">$1</span>');
    s = s.replace(/\b(const|let|for|of|return|if|else|function|new|this|while)\b/g, '<span class="k">$1</span>');
    s = s.replace(/(^|[^\w.])(\d+(?:\.\d+)?)(?![\w])/g, '$1<span class="n">$2</span>');
    return s;
  }

  var codeBody = $("#codeBody");
  var tabs = document.querySelectorAll(".code-tab");
  function showCode(key) {
    if (!codeBody) return;
    codeBody.innerHTML = highlight(CODE[key] || "");
    tabs.forEach(function (t) {
      var on = t.dataset.code === key;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
    });
  }
  tabs.forEach(function (t) {
    t.addEventListener("click", function () { showCode(t.dataset.code); });
  });
  if (tabs.length) showCode(tabs[0].dataset.code);

  /* ---------------- 启动 ---------------- */
  measure();
  rebuild();
  syncOutputs();
  window.addEventListener("load", function () { measure(); gridDirty = true; });
  requestAnimationFrame(frame);

  // 方便分享「训练过的状态」：地址后面加 #train 会先跑 700 步再继续
  if (location.hash === "#train") {
    for (var t0 = 0; t0 < 700; t0++) trainOnce();
    setRunning(true);
  }
})();
