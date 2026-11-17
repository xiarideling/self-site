/* 站点通用脚本：主题切换、导航高亮、入场动画、头像 */
(function () {
  "use strict";

  /* 头像默认用本地 assets/img/avatar.jpg（就是从 xiarideling 的 GitHub 头像存的，
     国内访问也不会裂）。想让头像跟着 GitHub 自动更新，把用户名填进来即可。 */
  var GITHUB_USER = "";

  /* ---- 主题（默认浅色，记住选择） ---- */
  var root = document.documentElement;

  function isDark() { return root.getAttribute("data-theme") === "dark"; }

  function applyTheme(mode) {
    if (mode === "dark") root.setAttribute("data-theme", "dark");
    else root.removeAttribute("data-theme");
    syncIcon();
  }

  var saved = null;
  try { saved = localStorage.getItem("xrl-theme"); } catch (e) {}
  if (saved === "dark") applyTheme("dark");

  // 也支持用链接直接指定：?theme=dark
  try {
    var qTheme = new URLSearchParams(location.search).get("theme");
    if (qTheme === "dark") applyTheme("dark");
    if (qTheme === "light") applyTheme("light");
  } catch (e) {}

  function syncIcon() {
    var dark = isDark();
    document.querySelectorAll("[data-theme-toggle]").forEach(function (b) {
      b.textContent = dark ? "☀" : "☾";
      b.setAttribute("aria-label", dark ? "切换到浅色" : "切换到深色");
      b.setAttribute("title", dark ? "切换到浅色" : "切换到深色");
    });
  }

  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-theme-toggle]");
    if (!btn) return;
    var next = isDark() ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem("xrl-theme", next); } catch (err) {}
  });

  /* ---- 头像 ---- */
  if (GITHUB_USER) {
    document.querySelectorAll("[data-avatar]").forEach(function (img) {
      img.src = "https://github.com/" + encodeURIComponent(GITHUB_USER) + ".png";
    });
  }

  /* ---- 当前页导航 ---- */
  var here = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".nav-link[href]").forEach(function (a) {
    var target = a.getAttribute("href").split("/").pop();
    if (target === here) a.setAttribute("aria-current", "page");
  });

  /* ---- 入场动画（首屏直接显示，避免动画卡住导致内容不可见） ---- */
  var items = document.querySelectorAll(".reveal");
  var vh = window.innerHeight || 800;
  var pending = [];
  Array.prototype.forEach.call(items, function (el) {
    if (el.getBoundingClientRect().top < vh * 0.92) el.classList.add("in");
    else pending.push(el);
  });

  if ("IntersectionObserver" in window && pending.length) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -30px 0px" });
    pending.forEach(function (el) { io.observe(el); });
  } else {
    pending.forEach(function (el) { el.classList.add("in"); });
  }

  // 兜底：2 秒后无论如何都显示，绝不让内容永远停在 opacity:0
  setTimeout(function () {
    Array.prototype.forEach.call(items, function (el) { el.classList.add("in"); });
  }, 1800);

  /* ---- 年份 ---- */
  var y = document.querySelector("[data-year]");
  if (y) y.textContent = String(new Date().getFullYear());

  syncIcon();
})();
