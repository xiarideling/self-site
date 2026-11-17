/* ============================================================
   要不要给猫开罐头 · 2 → 3 → 1 单样本手算
   把「前向 → 反向 → 更新」一步步走完，数字每一步都写出来
   ============================================================ */
(function () {
  "use strict";

  var INIT = {
    W1: [[0.5, -1.0], [0.3, -0.5], [1.5, -0.25]],
    b1: [-0.1, 0.1, 0.1],
    W2: [[0.8, 0.7, 1.0]],
    b2: [-0.08],
  };
  var X = [0.8, 0.4];
  var Y = 1;

  var W1, b1, W2, b2, stage, rounds, lr;

  var $ = function (id) { return document.getElementById(id); };
  var els = {
    t1: $("hcTable1"), t2: $("hcTable2"), prob: $("hcProb"), probBar: $("hcBar"),
    loss: $("hcLoss"), rounds: $("hcRounds"), log: $("hcLog"),
    fwd: $("hcForward"), bwd: $("hcBackward"), upd: $("hcUpdate"), reset: $("hcReset"),
    lr: $("hcLr"), lrOut: $("hcLrOut"),
  };
  if (!els.t1) return;

  function sigmoid(z) {
    if (z >= 0) { var e = Math.exp(-z); return 1 / (1 + e); }
    var e2 = Math.exp(z);
    return e2 / (1 + e2);
  }

  function f(v) {
    if (!isFinite(v)) return "—";
    var s = v.toFixed(3);
    if (s === "-0.000") s = "0.000";
    return s;
  }

  function reset() {
    W1 = INIT.W1.map(function (r) { return r.slice(); });
    b1 = INIT.b1.slice();
    W2 = INIT.W2.map(function (r) { return r.slice(); });
    b2 = INIT.b2.slice();
    stage = 0;
    rounds = 0;
    lr = parseFloat(els.lr.value);
    render(null);
  }

  /* 前向：返回所有中间量 */
  function forward() {
    var z1 = [0, 0, 0], a1 = [0, 0, 0];
    for (var i = 0; i < 3; i++) {
      z1[i] = W1[i][0] * X[0] + W1[i][1] * X[1] + b1[i];
      a1[i] = z1[i] > 0 ? z1[i] : 0;
    }
    var z2 = b2[0];
    for (var j = 0; j < 3; j++) z2 += W2[0][j] * a1[j];
    var p = sigmoid(z2);
    var loss = -(Y * Math.log(p) + (1 - Y) * Math.log(1 - p));
    return { z1: z1, a1: a1, z2: z2, p: p, loss: loss };
  }

  /* 反向：返回梯度 */
  function backward(g) {
    var d2 = g.p - Y; // sigmoid + BCE
    var dW2 = [0, 0, 0], db2 = d2;
    for (var j = 0; j < 3; j++) dW2[j] = d2 * g.a1[j];

    var d1 = [0, 0, 0];
    for (var i = 0; i < 3; i++) {
      var dz1 = W2[0][i] * d2 * (g.z1[i] > 0 ? 1 : 0);
      d1[i] = dz1;
    }
    var dW1 = [[0, 0], [0, 0], [0, 0]], db1 = [0, 0, 0];
    for (var r = 0; r < 3; r++) {
      dW1[r][0] = d1[r] * X[0];
      dW1[r][1] = d1[r] * X[1];
      db1[r] = d1[r];
    }
    return { d2: d2, dW2: dW2, db2: db2, d1: d1, dW1: dW1, db1: db1 };
  }

  /* ---------- 渲染 ---------- */
  function render(extra) {
    var g = forward();
    var names = ["心疼派", "纠结派", "认命派"];
    var b = extra || {};
    var showD = !!b.d1 && stage === 2;

    // 表 1：隐藏层
    var h = '<table class="matrix"><thead><tr><th>神经元</th><th>w₁</th><th>w₂</th><th>b</th><th>z¹</th><th>a¹</th>';
    if (showD) h += "<th>δ¹</th>";
    h += "</tr></thead><tbody>";
    for (var i = 0; i < 3; i++) {
      h += "<tr><td style='text-align:left'>" + names[i] + "</td><td>" + f(W1[i][0]) + "</td><td>" + f(W1[i][1]) + "</td><td>" + f(b1[i]) + "</td><td>" + f(g.z1[i]) + "</td><td class='hl'>" + f(g.a1[i]) + "</td>";
      if (showD) h += "<td>" + f(b.d1[i]) + "</td>";
      h += "</tr>";
    }
    h += "</tbody></table>";

    // 表 2：输出层
    var h2 = '<table class="matrix"><thead><tr><th>输出</th><th>v₁</th><th>v₂</th><th>v₃</th><th>b</th><th>z²</th><th>ŷ</th></tr></thead><tbody>';
    h2 += "<tr><td style='text-align:left'>开罐头</td><td>" + f(W2[0][0]) + "</td><td>" + f(W2[0][1]) + "</td><td>" + f(W2[0][2]) + "</td><td>" + f(b2[0]) + "</td><td>" + f(g.z2) + "</td><td class='hl'>" + f(g.p) + "</td></tr>";
    h2 += "</tbody></table>";

    els.t1.innerHTML = h;
    els.t2.innerHTML = h2;
    els.prob.textContent = (g.p * 100).toFixed(1) + "%";
    els.probBar.style.width = (g.p * 100).toFixed(1) + "%";
    els.loss.textContent = f(g.loss);
    els.rounds.textContent = String(rounds);

    // 日志
    var lines = [];
    lines.push("<span class='dim'>z¹ = W¹x + b¹ = [" + f(g.z1[0]) + ", " + f(g.z1[1]) + ", " + f(g.z1[2]) + "]</span>");
    lines.push("<span class='dim'>a¹ = ReLU(z¹) = [" + f(g.a1[0]) + ", " + f(g.a1[1]) + ", " + f(g.a1[2]) + "]</span>");
    lines.push("<span class='dim'>z² = W²a¹ + b² = " + f(g.z2) + "，ŷ = σ(z²) = " + f(g.p) + "</span>");
    if (b.d2 !== undefined) {
      lines.push("<em>δ² = ŷ − y = " + f(b.d2) + "</em>");
      lines.push("<em>∂L/∂W² = δ²·a¹ᵀ = [" + f(b.dW2[0]) + ", " + f(b.dW2[1]) + ", " + f(b.dW2[2]) + "]</em>");
      lines.push("<em>δ¹ = (W²ᵀδ²) ⊙ ReLU′(z¹) = [" + f(b.d1[0]) + ", " + f(b.d1[1]) + ", " + f(b.d1[2]) + "]</em>");
      lines.push("<em>∂L/∂W¹ = δ¹·xᵀ，形状 3×2</em>");
    }
    if (b.updated) {
      lines.push("<b style='color:var(--cyan)'>W ← W − " + lr.toFixed(1) + "·∂L/∂W，参数已更新 → 新的 ŷ = " + f(g.p) + "</b>");
    }
    els.log.innerHTML = lines.map(function (s) { return "<div>" + s + "</div>"; }).join("");

    els.fwd.disabled = stage >= 1;
    els.bwd.disabled = stage !== 1;
    els.upd.disabled = stage !== 2;
  }

  els.fwd.addEventListener("click", function () { stage = 1; render(null); });
  els.bwd.addEventListener("click", function () {
    if (stage !== 1) return;
    var g = forward();
    var b = backward(g);
    stage = 2;
    render(b);
  });
  els.upd.addEventListener("click", function () {
    if (stage !== 2) return;
    var g = forward();
    var b = backward(g);
    for (var i = 0; i < 3; i++) {
      W1[i][0] -= lr * b.dW1[i][0];
      W1[i][1] -= lr * b.dW1[i][1];
      b1[i] -= lr * b.db1[i];
      W2[0][i] -= lr * b.dW2[i];
    }
    b2[0] -= lr * b.db2;
    rounds += 1;
    stage = 0;
    render({ d1: b.d1, d2: b.d2, dW2: b.dW2, updated: true });
  });
  els.reset.addEventListener("click", reset);
  els.lr.addEventListener("input", function () {
    lr = parseFloat(els.lr.value);
    els.lrOut.textContent = lr.toFixed(1);
  });

  els.lrOut.textContent = parseFloat(els.lr.value).toFixed(1);
  reset();
})();
