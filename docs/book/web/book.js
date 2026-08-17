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
  let selectedResult = -1;

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
  function setSidebarOpen(open) {
    sidebar.classList.toggle("open", open);
    document.getElementById("sidebar-toggle").setAttribute("aria-expanded", String(open));
  }
  async function load(slug, heading) {
    const item = manifest.chapters.find((entry) => entry.slug === slug) || manifest.chapters[0];
    if (item.slug !== slug) history.replaceState({}, "", "#/" + item.slug);
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
    setSidebarOpen(false);
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
  function updateSearchSelection() {
    results.querySelectorAll("a[role=option]").forEach((link, index) => {
      link.setAttribute("aria-selected", String(index === selectedResult));
    });
  }
  function doSearch() {
    const query = search.value.trim().toLowerCase();
    results.replaceChildren(); results.hidden = !query; selectedResult = -1;
    if (!query) return;
    searchIndex.filter((item) => (item.title + " " + item.text).toLowerCase().includes(query)).slice(0, 12).forEach((item) => {
      const link = document.createElement("a"); link.href = "#/" + item.slug; link.textContent = item.title; link.setAttribute("role", "option"); link.setAttribute("aria-selected", "false"); results.append(link);
    });
  }
  setSidebarOpen(false);
  document.getElementById("sidebar-toggle").addEventListener("click", function () { setSidebarOpen(!sidebar.classList.contains("open")); });
  document.getElementById("theme-toggle").addEventListener("click", switchTheme);
  document.getElementById("search-form").addEventListener("submit", (event) => event.preventDefault());
  search.addEventListener("input", doSearch);
  search.addEventListener("keydown", (event) => {
    const options = results.querySelectorAll("a[role=option]");
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      if (!options.length) return;
      event.preventDefault();
      selectedResult = (selectedResult + (event.key === "ArrowDown" ? 1 : options.length - 1)) % options.length;
      updateSearchSelection();
    } else if (event.key === "Enter" && selectedResult >= 0) {
      event.preventDefault(); options[selectedResult].click();
    } else if (event.key === "Escape") {
      results.hidden = true; selectedResult = -1;
    }
  });
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
