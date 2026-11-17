/* ============================================================
   MLP 引擎 —— 纯手写，无依赖
   前向 / 反向 / SGD·Momentum·Adam / L2 / 四种数据集
   浏览器里挂到 window.MLPEngine，node 里 module.exports
   ============================================================ */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.MLPEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  /* ---------- 激活函数 ---------- */
  function sigmoid(z) {
    if (z >= 0) { var e = Math.exp(-z); return 1 / (1 + e); }
    var e2 = Math.exp(z);
    return e2 / (1 + e2);
  }

  var ACTS = {
    relu: {
      label: "ReLU",
      f: function (z) { return z > 0 ? z : 0; },
      d: function (z) { return z > 0 ? 1 : 0; },
    },
    tanh: {
      label: "tanh",
      f: function (z) { return Math.tanh(z); },
      d: function (z, a) { return 1 - a * a; },
    },
    sigmoid: {
      label: "Sigmoid",
      f: sigmoid,
      d: function (z, a) { return a * (1 - a); },
    },
  };

  /* ---------- 小工具 ---------- */
  function zeros(n) {
    var a = new Array(n);
    for (var i = 0; i < n; i++) a[i] = 0;
    return a;
  }

  function randn() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ---------- 数据集 ---------- */
  var DATASETS = {
    circle: { label: "同心圆", hint: "线性不可分，需要至少一个隐藏层" },
    xor: { label: "异或", hint: "经典反例：没有隐藏层必然失败" },
    spiral: { label: "双螺旋", hint: "需要多层 + 足够多的神经元" },
    moons: { label: "双月", hint: "弯曲边界，考验隐藏层容量" },
    gauss: { label: "高斯团", hint: "几乎线性可分，最简单" },
  };

  function makeDataset(name, n, noise, seed) {
    var rnd = mulberry32(seed == null ? 20260417 : seed);
    var pts = [];
    var i, c, t, r, half = Math.floor(n / 2);
    var nz = noise == null ? 0.08 : noise;

    if (name === "spiral") {
      for (c = 0; c < 2; c++) {
        for (i = 0; i < half; i++) {
          r = (i / half) * 0.95 + 0.04;
          t = 1.75 * (i / half) * Math.PI * 2 + c * Math.PI;
          pts.push({
            x: r * Math.sin(t) + randn() * nz * 0.12,
            y: r * Math.cos(t) + randn() * nz * 0.12,
            label: c,
          });
        }
      }
    } else if (name === "circle") {
      for (c = 0; c < 2; c++) {
        for (i = 0; i < half; i++) {
          t = rnd() * Math.PI * 2;
          r = (c === 1 ? 0.82 : 0.36) + randn() * nz * 0.35;
          pts.push({ x: r * Math.cos(t), y: r * Math.sin(t), label: c });
        }
      }
    } else if (name === "xor") {
      for (i = 0; i < n; i++) {
        var sx = rnd() < 0.5 ? -1 : 1;
        var sy = rnd() < 0.5 ? -1 : 1;
        pts.push({
          x: sx * (0.28 + rnd() * 0.42) + randn() * nz * 0.3,
          y: sy * (0.28 + rnd() * 0.42) + randn() * nz * 0.3,
          label: sx * sy > 0 ? 1 : 0,
        });
      }
    } else if (name === "moons") {
      for (c = 0; c < 2; c++) {
        for (i = 0; i < half; i++) {
          t = Math.PI * (i / half);
          if (c === 0) {
            pts.push({ x: Math.cos(t) * 0.85 - 0.1, y: Math.sin(t) * 0.85 - 0.15, label: 0 });
          } else {
            pts.push({ x: Math.cos(t + Math.PI) * 0.85 + 0.1, y: Math.sin(t + Math.PI) * 0.85 + 0.35, label: 1 });
          }
        }
      }
      for (i = 0; i < n; i++) {
        pts[i].x += randn() * nz * 0.16;
        pts[i].y += randn() * nz * 0.16;
      }
    } else {
      for (c = 0; c < 2; c++) {
        var cx = c === 0 ? -0.5 : 0.5;
        for (i = 0; i < half; i++) {
          pts.push({ x: cx + randn() * 0.16 + nz * randn() * 0.5, y: randn() * 0.16 + nz * randn() * 0.5, label: c });
        }
      }
    }

    // 裁到 [-1,1]，避免点被画到画布外
    for (i = 0; i < pts.length; i++) {
      pts[i].x = Math.max(-0.98, Math.min(0.98, pts[i].x));
      pts[i].y = Math.max(-0.98, Math.min(0.98, pts[i].y));
    }
    return pts;
  }

  /* ---------- 网络 ---------- */
  function MLP(cfg) {
    cfg = cfg || {};
    this.sizes = (cfg.sizes || [2, 4, 1]).slice();
    this.actName = cfg.act || "relu";
    this.act = ACTS[this.actName] || ACTS.relu;
    this.lr = cfg.lr == null ? 0.1 : cfg.lr;
    this.reg = cfg.reg || 0;
    this.opt = cfg.opt || "sgd";
    this.beta = 0.9;
    this.t = 0;
    this.steps = 0;
    this.init();
  }

  MLP.prototype.init = function () {
    var L = this.sizes.length - 1;
    this.W = [null]; this.b = [null];
    this.gW = [null]; this.gB = [null];
    this.vW = [null]; this.vb = [null];
    this.mW = [null]; this.mb = [null];
    this.t = 0; this.steps = 0;

    for (var l = 1; l <= L; l++) {
      var nin = this.sizes[l - 1], nout = this.sizes[l];
      var scale = this.actName === "relu" ? Math.sqrt(2 / nin) : Math.sqrt(1 / nin);
      var W = new Array(nout), b = zeros(nout);
      for (var i = 0; i < nout; i++) {
        var row = new Array(nin);
        for (var j = 0; j < nin; j++) row[j] = randn() * scale;
        W[i] = row;
      }
      this.W[l] = W; this.b[l] = b;
      this.gW[l] = W.map(function (r) { return zeros(r.length); });
      this.gB[l] = zeros(nout);
      this.vW[l] = W.map(function (r) { return zeros(r.length); });
      this.vb[l] = zeros(nout);
      this.mW[l] = W.map(function (r) { return zeros(r.length); });
      this.mb[l] = zeros(nout);
    }
  };

  MLP.prototype.paramCount = function () {
    var c = 0;
    for (var l = 1; l < this.sizes.length; l++) c += this.sizes[l] * this.sizes[l - 1] + this.sizes[l];
    return c;
  };

  /* 前向：返回每一层的 z 与 a（a[0] 是输入） */
  MLP.prototype.forward = function (x) {
    var a = [x], z = [null];
    for (var l = 1; l < this.sizes.length; l++) {
      var W = this.W[l], b = this.b[l], prev = a[l - 1];
      var out = new Array(W.length), zl = new Array(W.length);
      var last = l === this.sizes.length - 1;
      for (var i = 0; i < W.length; i++) {
        var s = b[i], row = W[i];
        for (var j = 0; j < row.length; j++) s += row[j] * prev[j];
        zl[i] = s;
        out[i] = last ? sigmoid(s) : this.act.f(s);
      }
      z[l] = zl; a[l] = out;
    }
    return { a: a, z: z };
  };

  MLP.prototype.predict = function (x) {
    var r = this.forward(x);
    return r.a[this.sizes.length - 1][0];
  };

  /* 反向：把梯度累加进 gW / gB（配合 trainBatch 使用） */
  MLP.prototype.backward = function (cache, y) {
    var L = this.sizes.length - 1;
    var a = cache.a, z = cache.z;
    // 输出层：sigmoid + 二元交叉熵，dz = ŷ - y
    var delta = [null];
    var out = a[L];
    delta[L] = new Array(out.length);
    for (var i = 0; i < out.length; i++) delta[L][i] = out[i] - y;

    for (var l = L; l >= 1; l--) {
      var prevA = a[l - 1];
      var gW = this.gW[l], gB = this.gB[l], W = this.W[l];
      for (var r = 0; r < W.length; r++) {
        var d = delta[l][r];
        gB[r] += d;
        var grow = gW[r];
        for (var c = 0; c < prevA.length; c++) grow[c] += d * prevA[c];
      }
      if (l > 1) {
        var da = zeros(this.sizes[l - 1]);
        for (var rr = 0; rr < W.length; rr++) {
          var dd = delta[l][rr];
          for (var cc = 0; cc < da.length; cc++) da[cc] += dd * W[rr][cc];
        }
        var dl = new Array(da.length);
        for (var k = 0; k < da.length; k++) {
          dl[k] = da[k] * this.act.d(z[l - 1][k], a[l - 1][k]);
        }
        delta[l - 1] = dl;
      }
    }
  };

  MLP.prototype.zeroGrads = function () {
    for (var l = 1; l < this.sizes.length; l++) {
      for (var r = 0; r < this.gW[l].length; r++) {
        var row = this.gW[l][r];
        for (var c = 0; c < row.length; c++) row[c] = 0;
      }
      var gb = this.gB[l];
      for (var i = 0; i < gb.length; i++) gb[i] = 0;
    }
  };

  /* 用一批样本走一次前向+反向，然后更新参数 */
  MLP.prototype.trainBatch = function (points, idx) {
    this.zeroGrads();
    var n = idx.length, i;
    for (i = 0; i < n; i++) {
      var p = points[idx[i]];
      this.backward(this.forward([p.x, p.y]), p.label);
    }
    var inv = 1 / n;
    this.t += 1;
    this.steps += 1;
    this.update(inv);
    return n;
  };

  MLP.prototype.update = function (scale) {
    var lr = this.lr, reg = this.reg, opt = this.opt, t = this.t;
    var b1 = 0.9, b2 = 0.999, eps = 1e-8;
    for (var l = 1; l < this.sizes.length; l++) {
      var W = this.W[l], gW = this.gW[l], b = this.b[l], gB = this.gB[l];
      var vW = this.vW[l], mW = this.mW[l], vb = this.vb[l], mb = this.mb[l];

      for (var r = 0; r < W.length; r++) {
        var row = W[r], grow = gW[r], vrow = vW[r], mrow = mW[r];
        for (var c = 0; c < row.length; c++) {
          var g = grow[c] * scale + reg * row[c];
          if (opt === "momentum") {
            vrow[c] = this.beta * vrow[c] + g;
            row[c] -= lr * vrow[c];
          } else if (opt === "adam") {
            mrow[c] = b1 * mrow[c] + (1 - b1) * g;
            vrow[c] = b2 * vrow[c] + (1 - b2) * g * g;
            var mh = mrow[c] / (1 - Math.pow(b1, t));
            var vh = vrow[c] / (1 - Math.pow(b2, t));
            row[c] -= lr * mh / (Math.sqrt(vh) + eps);
          } else {
            row[c] -= lr * g;
          }
        }
        gB[r] = gB[r] * scale;
        if (opt === "momentum") {
          vb[r] = this.beta * vb[r] + gB[r];
          b[r] -= lr * vb[r];
        } else if (opt === "adam") {
          mb[r] = b1 * mb[r] + (1 - b1) * gB[r];
          vb[r] = b2 * vb[r] + (1 - b2) * gB[r] * gB[r];
          var mhb = mb[r] / (1 - Math.pow(b1, t));
          var vhb = vb[r] / (1 - Math.pow(b2, t));
          b[r] -= lr * mhb / (Math.sqrt(vhb) + eps);
        } else {
          b[r] -= lr * gB[r];
        }
      }
    }
  };

  /* 整个数据集上的平均损失与准确率 */
  MLP.prototype.evaluate = function (points) {
    var loss = 0, ok = 0;
    for (var i = 0; i < points.length; i++) {
      var p = points[i], pr = this.predict([p.x, p.y]);
      var pc = Math.min(1 - 1e-7, Math.max(1e-7, pr));
      loss -= p.label * Math.log(pc) + (1 - p.label) * Math.log(1 - pc);
      if ((pr >= 0.5 ? 1 : 0) === p.label) ok++;
    }
    return { loss: loss / points.length, acc: ok / points.length };
  };

  /* 给决策边界用的网格预测：返回 [0,1] 概率数组，行优先，左上 -> 右下 */
  MLP.prototype.evalGrid = function (res) {
    var out = new Float32Array(res * res);
    for (var r = 0; r < res; r++) {
      var y = 1 - (2 * r) / (res - 1);
      for (var c = 0; c < res; c++) {
        var x = (2 * c) / (res - 1) - 1;
        out[r * res + c] = this.predict([x, y]);
      }
    }
    return out;
  };

  return {
    ACTS: ACTS,
    DATASETS: DATASETS,
    makeDataset: makeDataset,
    MLP: MLP,
    sigmoid: sigmoid,
    mulberry32: mulberry32,
  };
});
