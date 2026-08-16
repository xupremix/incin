(function () {
  "use strict";
  const root = document.documentElement;
  const chapter = document.getElementById("chapter");
  const nav = document.getElementById("chapter-nav");
  const previous = document.getElementById("previous");
  const next = document.getElementById("next");
  const sidebar = document.getElementById("sidebar");
  const search = document.getElementById("search");
  const results = document.getElementById("search-results");
  let manifest = null;
  let searchIndex = [];

  function basePath() {
    return window.location.pathname.endsWith("/") ? window.location.pathname : window.location.pathname.replace(/[^/]+$/, "");
  }
  function route() {
    const value = window.location.hash.replace(/^#\/?/, "") || "introduction";
    const parts = value.split("#");
    return { slug: parts[0] || "introduction", heading: parts[1] || "" };
  }
  function chapterIndex(slug) { return manifest.chapters.findIndex((item) => item.slug === slug); }
  function setLink(element, item, label) {
    if (!item) { element.hidden = true; return; }
    element.hidden = false; element.href = "#/" + item.slug; element.textContent = label + item.title;
  }
  async function load(slug, heading) {
    const item = manifest.chapters.find((entry) => entry.slug === slug) || manifest.chapters[0];
    const response = await fetch(basePath() + "chapters/" + item.slug + ".html");
    if (!response.ok) throw new Error("Chapter failed to load: " + response.status);
    chapter.innerHTML = await response.text();
    document.title = item.title + " - The Incin Book";
    nav.querySelectorAll("a[data-slug]").forEach((link) => {
      if (link.dataset.slug === item.slug) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    const index = chapterIndex(item.slug);
    setLink(previous, manifest.chapters[index - 1], "← Previous: ");
    setLink(next, manifest.chapters[index + 1], "Next: ");
    sidebar.classList.remove("open");
    requestAnimationFrame(() => {
      const target = heading && document.getElementById(heading);
      if (target) target.scrollIntoView(); else window.scrollTo(0, 0);
      chapter.focus({ preventScroll: true });
    });
  }
  function navigate(event) {
    const anchor = event.target.closest("a[href^='#/']");
    if (anchor && anchor.hash) { event.preventDefault(); history.pushState({}, "", anchor.hash); loadRoute(); }
  }
  async function loadRoute() { const target = route(); try { await load(target.slug, target.heading); } catch (error) { chapter.innerHTML = "<h1>Chapter unavailable</h1><p>" + error + "</p>"; } }
  function switchTheme() {
    const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = nextTheme; localStorage.setItem("incin-book-theme", nextTheme);
    document.getElementById("theme-toggle").textContent = nextTheme === "dark" ? "Light" : "Dark";
  }
  function doSearch() {
    const query = search.value.trim().toLowerCase();
    results.replaceChildren(); results.hidden = !query;
    if (!query) return;
    searchIndex.filter((item) => (item.title + " " + item.text).toLowerCase().includes(query)).slice(0, 12).forEach((item) => {
      const link = document.createElement("a"); link.href = "#/" + item.slug; link.textContent = item.title; link.setAttribute("role", "option"); results.append(link);
    });
  }
  document.getElementById("sidebar-toggle").addEventListener("click", function () { const open = sidebar.classList.toggle("open"); this.setAttribute("aria-expanded", String(open)); });
  document.getElementById("theme-toggle").addEventListener("click", switchTheme);
  document.getElementById("search-form").addEventListener("submit", (event) => event.preventDefault());
  search.addEventListener("input", doSearch);
  document.addEventListener("click", navigate);
  window.addEventListener("popstate", loadRoute);
  window.addEventListener("hashchange", loadRoute);
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, textarea, select, button, a")) return;
    if (event.key === "ArrowLeft" && !previous.hidden) { window.location.hash = previous.hash; }
    if (event.key === "ArrowRight" && !next.hidden) { window.location.hash = next.hash; }
    if (event.key === "/") { event.preventDefault(); search.focus(); }
  });
  (async function () {
    const saved = localStorage.getItem("incin-book-theme");
    if (saved === "dark" || saved === "light") root.dataset.theme = saved;
    document.getElementById("theme-toggle").textContent = root.dataset.theme === "dark" ? "Light" : "Dark";
    [manifest, searchIndex] = await Promise.all([fetch(basePath() + "chapters.json").then((r) => r.json()), fetch(basePath() + "search-index.json").then((r) => r.json())]);
    await loadRoute();
  }());
}());
