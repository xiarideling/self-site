/* Agent 列表渲染 + 分类筛选 */
(function () {
  "use strict";
  var data = window.AGENTS || [];
  var grid = document.getElementById("agentGrid");
  var bar = document.getElementById("filterBar");
  var count = document.getElementById("agentCount");
  if (!grid || !bar) return;

  var active = "全部";
  var cats = ["全部"];
  data.forEach(function (a) {
    (a.cats || []).forEach(function (c) {
      if (cats.indexOf(c) === -1) cats.push(c);
    });
  });

  function chip(label, n) {
    var b = document.createElement("button");
    b.className = "chip";
    b.type = "button";
    b.textContent = label + " " + n;
    b.setAttribute("aria-pressed", label === active ? "true" : "false");
    b.addEventListener("click", function () {
      active = label;
      Array.prototype.forEach.call(bar.children, function (c) {
        c.setAttribute("aria-pressed", c === b ? "true" : "false");
      });
      render();
    });
    return b;
  }

  cats.forEach(function (c) {
    var n = c === "全部" ? data.length : data.filter(function (a) { return (a.cats || []).indexOf(c) !== -1; }).length;
    bar.appendChild(chip(c, n));
  });

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function statusClass(s) {
    if (s === "常用") return "tag cyan";
    if (s === "在试") return "tag blue";
    return "tag amber";
  }

  function card(a) {
    var el = document.createElement("article");
    el.className = "agent";
    var tags = (a.cats || []).map(function (c) { return '<span class="tag">' + esc(c) + "</span>"; }).join("");
    var foot = "";
    if (a.since || a.url) {
      foot =
        '<div class="agent-foot">' +
          '<span class="agent-org">' + (a.since ? "自 " + esc(a.since) : "") + "</span>" +
          (a.url ? '<a href="' + esc(a.url) + '" target="_blank" rel="noopener noreferrer">去试试 ↗</a>' : "<span></span>") +
        "</div>";
    }
    var logoHtml = a.icon
      ? '<div class="agent-logo has-img"><img src="' + esc(a.icon) + '" alt="" loading="lazy" /></div>'
      : '<div class="agent-logo">' + esc(a.logo || a.name.slice(0, 1)) + "</div>";
    el.innerHTML =
      '<div class="agent-head">' + logoHtml +
        "<div><div class=\"agent-name\">" + esc(a.name) + "</div>" +
        '<div class="agent-org">' + esc(a.org || "") + "</div></div>" +
      "</div>" +
      '<p class="agent-desc">' + esc(a.desc) + "</p>" +
      (a.note ? '<p class="agent-desc small dim" style="margin:0">“' + esc(a.note) + "”</p>" : "") +
      '<div style="display:flex;gap:.35rem;flex-wrap:wrap;margin-top:.35rem">' +
        '<span class="' + statusClass(a.status) + '">' + esc(a.status) + "</span>" + tags +
      "</div>" +
      foot;
    return el;
  }

  function render() {
    grid.innerHTML = "";
    var list = data.filter(function (a) {
      return active === "全部" || (a.cats || []).indexOf(active) !== -1;
    });
    list.forEach(function (a) { grid.appendChild(card(a)); });
    if (count) count.textContent = String(list.length);
  }

  render();
})();
