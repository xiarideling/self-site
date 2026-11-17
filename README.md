# 夏日の苓 · 个人站

一个纯静态的个人网站：左边是试用过的 AI Agent 清单，右边是一台真的会训练的多层感知机。

没有构建步骤，没有依赖，没有后端。改了文件刷新就生效。

## 结构

```
site/
├── index.html              首页
├── agents.html             01 · 试用过的 Agent
├── mlp.html                02 · MLP 实验室
├── 404.html                找不到页面时的兜底
├── assets/
│   ├── favicon.svg
│   ├── img/                头像与两个 Agent 的图标
│   ├── css/style.css       全部样式（默认浅米色，可切深色）
│   └── js/
│       ├── site.js         导航高亮、主题切换、入场动画
│       ├── agents-data.js  ← 改 Agent 清单只改这个文件
│       ├── agents.js       清单渲染 + 分类筛选
│       ├── mlp-engine.js   MLP 引擎：前向 / 反向 / SGD·Momentum·Adam / L2
│       ├── mlp-lab.js      实验台：控件、训练循环、四块画布
│       └── handcalc.js     给猫开罐头 2→3→1 单样本手算
└── .nojekyll               让 GitHub Pages 跳过 Jekyll
```

## 改内容

### Agent 清单

只改 `assets/js/agents-data.js`。每条记录：

```js
{
  name: "WorkBuddy",           // 名称
  org: "",                     // 出品方，小字，可留空
  logo: "W",                   // 卡片左上角的一两个字符
  desc: "一句话说明它替你做了什么。",
  status: "弃用",              // 常用 / 在试 / 弃用
  cats: ["通用"],              // 分类，筛选按钮会自动生成
  url: "",                     // 没有就写 ""
  since: "",                   // 开始使用的时间，可留空
  note: "一条最具体的结论。"     // 可选
}
```

分类不用提前定义，`cats` 里出现过的词会自动变成筛选按钮。

### 站点信息

名字出现在三个 HTML 的 `<nav class="brand">` 和 `<footer>` 里，直接搜「夏日の苓」替换。

### 头像

默认用 `assets/img/avatar.jpg`，就是从 `xiarideling` 的 GitHub 头像存下来的本地副本，
国内访问也不会裂图。想让它跟着 GitHub 自动更新，打开 `assets/js/site.js` 最上面那行：

```js
var GITHUB_USER = "xiarideling";   // 填用户名，头像会走 https://github.com/<用户名>.png
```

留空就用本地文件。

### Agent 图标

`assets/img/` 下放着两个官方图标：`deepseek.svg`（DeepSeek 的鲸鱼）和 `workbuddy.svg`（腾讯 WorkBuddy）。
在 `agents-data.js` 里用 `icon` 字段指过去；想换别的就丢一张图进去、改路径。

### MLP 页的默认状态

`assets/js/mlp-lab.js` 顶部的 `state` 对象就是初始超参数：

```js
var state = {
  dataset: "circle",  // circle / xor / spiral / moons / gauss
  hidden: 1,          // 隐藏层数
  neurons: 6,         // 每层神经元
  act: "relu",        // relu / tanh / sigmoid
  lr: 0.1,
  reg: 0,
  batch: 16,
  opt: "sgd",         // sgd / momentum / adam
  noise: 0.08,
  speed: 6,
};
```

MLP 页地址后面加 `#train`，打开时会先跑 700 步并继续训练，方便直接分享训练后的状态。

## 本地预览

```bash
# 任意静态服务器都行，比如
npx serve site
# 或
python -m http.server 8000 --directory site
```

直接双击 `index.html` 也能看，但用本地服务器更接近线上环境。

## 部署到 GitHub Pages

1. 新建仓库（公开），把 `site/` 里的内容放到仓库根目录。
2. Settings → Pages → Source 选 `Deploy from a branch`，分支 `main`，目录 `/ (root)`。
3. 等一两分钟，访问 `https://<用户名>.github.io/<仓库名>/`。

如果要让站点在 `https://<用户名>.github.io/` 根路径，仓库名必须叫 `<用户名>.github.io`。

以后想绑自己的域名：Settings → Pages → Custom domain，GitHub 会自动从 `github.io` 地址 301 跳转过去，旧链接不会断。

## 一点说明

MLP 页里的矩阵乘法、反向传播、三种优化器都是手写的，没有用任何机器学习库。
这样做的目的不是炫技，而是让每一个数字都能对上页面上的公式——改个学习率，你能看见它为什么学不动。

所有交互都在浏览器里完成，没有埋点，没有请求外部接口。
