(function () {
  "use strict";

  const html = document.documentElement;
  const body = document.body;
  const chapter = document.getElementById("chapter");
  const nav = document.getElementById("chapter-nav");
  const previous = document.getElementById("previous");
  const next = document.getElementById("next");
  const navPrev = document.getElementById("nav-prev");
  const navNext = document.getElementById("nav-next");
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const themeToggle = document.getElementById("theme-toggle");
  const themeList = document.getElementById("theme-list");
  const searchToggle = document.getElementById("search-toggle");
  const searchWrapper = document.getElementById("search-wrapper");
  const search = document.getElementById("search");
  const searchResults = document.getElementById("search-results");

  let manifest = null;
  let searchIndex = [];
  let selectedResult = -1;

  /* ==========================================================================
     Base Paths & Routing
     ========================================================================== */

  function basePath() {
    return window.location.pathname.endsWith("/")
      ? window.location.pathname
      : window.location.pathname.replace(/[^/]+$/, "");
  }

  function route() {
    const value = window.location.hash.replace(/^#\/?/, "") || "introduction";
    const parts = value.split("#");
    return { slug: parts[0] || "introduction", heading: parts[1] || "" };
  }

  function chapterIndex(slug) {
    return manifest ? manifest.chapters.findIndex((item) => item.slug === slug) : -1;
  }

  function updateNavLinks(currentIndex) {
    const prevItem = currentIndex > 0 ? manifest.chapters[currentIndex - 1] : null;
    const nextItem = currentIndex >= 0 && currentIndex < manifest.chapters.length - 1
      ? manifest.chapters[currentIndex + 1]
      : null;

    if (prevItem) {
      previous.hidden = false;
      previous.href = "#/" + prevItem.slug;
      previous.textContent = "← Previous: " + prevItem.title;
      navPrev.classList.remove("hidden");
      navPrev.href = "#/" + prevItem.slug;
      navPrev.title = "Previous: " + prevItem.title;
    } else {
      previous.hidden = true;
      navPrev.classList.add("hidden");
    }

    if (nextItem) {
      next.hidden = false;
      next.href = "#/" + nextItem.slug;
      next.textContent = "Next: " + nextItem.title + " →";
      navNext.classList.remove("hidden");
      navNext.href = "#/" + nextItem.slug;
      navNext.title = "Next: " + nextItem.title;
    } else {
      next.hidden = true;
      navNext.classList.add("hidden");
    }
  }

  /* ==========================================================================
     Theme Management (Navy, Rust, Light, Coal, Ayu)
     ========================================================================== */

  function setTheme(theme) {
    const validThemes = ["navy", "rust", "light", "coal", "ayu"];
    if (!validThemes.includes(theme)) theme = "navy";

    html.className = theme;
    html.dataset.theme = theme;
    localStorage.setItem("incin-book-theme", theme);
    localStorage.setItem("mdbook-theme", theme);

    if (themeList) {
      themeList.querySelectorAll(".theme-option").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.theme === theme);
      });
    }
  }

  function initTheme() {
    const saved = localStorage.getItem("incin-book-theme") || localStorage.getItem("mdbook-theme") || "navy";
    setTheme(saved);
  }

  /* ==========================================================================
     Sidebar Toggling
     ========================================================================== */

  function setSidebarOpen(open) {
    if (window.innerWidth <= 900) {
      body.classList.toggle("sidebar-open", open);
      body.classList.remove("sidebar-hidden");
    } else {
      body.classList.toggle("sidebar-hidden", !open);
      body.classList.remove("sidebar-open");
      localStorage.setItem("mdbook-sidebar", open ? "visible" : "hidden");
    }
    if (sidebarToggle) {
      sidebarToggle.setAttribute("aria-expanded", String(open));
    }
  }

  function initSidebar() {
    if (window.innerWidth > 900) {
      const saved = localStorage.getItem("mdbook-sidebar") !== "hidden";
      setSidebarOpen(saved);
    } else {
      setSidebarOpen(false);
    }
  }

  /* ==========================================================================
     Highlighter & Output Parser
     ========================================================================== */

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function highlightRustCode(code) {
    const tokens = [];
    let text = code;

    // Pattern definitions
    const RUST_RE = new RegExp(
      [
        '(?<comment>//[^\n]*|/\\*[\\s\\S]*?\\*/)',
        '(?<attribute>#!?\\[[\\s\\S]*?\\])',
        '(?<string>"(?:\\\\.|[^"\\\\])*"|b"(?:\\\\.|[^"\\\\])*"|r#*"(?:[\\s\\S]*?)"#*)',
        '(?<char>\'(?:\\\\.|[^\'\\\\])\'|b\'(?:\\\\.|[^\'\\\\])\')',
        '(?<lifetime>\'[a-zA-Z_]\\w*\\b)',
        '(?<keyword>\\b(?:as|async|await|break|const|continue|crate|dyn|else|enum|extern|false|fn|for|if|impl|in|let|loop|match|mod|move|mut|pub|ref|return|self|Self|static|struct|super|trait|true|type|unsafe|use|where|while)\\b)',
        '(?<type>\\b(?:Tensor|Backend|Shape|Dim|DimCons|Nil|DType|Device|ConstDevice|ConstDType|Cpu|Cuda|DefaultBackend|Grad|NoGrad|Result|Option|Some|None|Ok|Err|String|Vec|Box|Arc|Rc|PhantomData|Unsigned|UInt|UTerm|Dyn|Ranked|f32|f64|i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize|bool|char|str)\\b)',
        '(?<macro>\\b[a-zA-Z_]\\w*!)',
        '(?<number>\\b(?:0x[0-9a-fA-F_]+|0b[01_]+|0o[0-7_]+|\\d[\\d_]*(?:\\.[\\d_]+)?(?:[eE][+-]?[\\d_]+)?(?:f32|f64|i8|i16|i32|i64|isize|u8|u16|u32|u64|usize)?)\\b)',
        '(?<fn>\\b[a-zA-Z_]\\w*(?=\\s*\\())',
      ].join('|'),
      'g'
    );

    let lastIndex = 0;
    let result = '';
    let match;

    while ((match = RUST_RE.exec(text)) !== null) {
      result += escapeHtml(text.slice(lastIndex, match.index));
      const groups = match.groups;
      if (groups.comment) {
        result += '<span class="hl-cmt">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.attribute) {
        result += '<span class="hl-meta">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.string || groups.char) {
        result += '<span class="hl-str">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.lifetime) {
        result += '<span class="hl-sym">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.keyword) {
        result += '<span class="hl-kw">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.type) {
        result += '<span class="hl-type">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.macro) {
        result += '<span class="hl-macro">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.number) {
        result += '<span class="hl-num">' + escapeHtml(match[0]) + '</span>';
      } else if (groups.fn) {
        result += '<span class="hl-fn">' + escapeHtml(match[0]) + '</span>';
      } else {
        result += escapeHtml(match[0]);
      }
      lastIndex = RUST_RE.lastIndex;
    }
    result += escapeHtml(text.slice(lastIndex));
    return result;
  }

  function highlightConsoleOutput(raw) {
    const lines = raw.split('\n');
    return lines.map((line) => {
      let escaped = escapeHtml(line);

      // Compiler error header
      escaped = escaped.replace(/\b(error\[E\d+\]):/g, '<span class="diag-err">$1:</span>');
      escaped = escaped.replace(/\b(error):/g, '<span class="diag-err">error:</span>');
      escaped = escaped.replace(/\b(warning):/g, '<span class="diag-warn">warning:</span>');
      escaped = escaped.replace(/\b(help|note):/g, '<span class="diag-help">$1:</span>');
      escaped = escaped.replace(/(--&gt;\s+[^\s:]+:\d+:\d+)/g, '<span class="diag-loc">$1</span>');
      escaped = escaped.replace(/^(\s*\d*\s*\|)/g, '<span class="diag-pipe">$1</span>');
      escaped = escaped.replace(/\b(Compiling|Checking|Finished|Running)\b/g, '<span class="diag-verb">$1</span>');
      escaped = escaped.replace(/\b(test\s+[^\s]+\s+\.\.\.\s+ok)\b/g, '<span class="test-ok">$1</span>');
      escaped = escaped.replace(/\b(test\s+[^\s]+\s+\.\.\.\s+FAILED)\b/g, '<span class="test-fail">$1</span>');

      return escaped;
    }).join('\n');
  }

  function enhanceCodeBlocks(container) {
    const preBlocks = container.querySelectorAll("pre");

    preBlocks.forEach((pre) => {
      const code = pre.querySelector("code");
      if (!code) return;

      const rawText = code.textContent || "";
      const isRust = code.classList.contains("language-rust") || code.classList.contains("rust");
      const isConsole = code.classList.contains("language-console") ||
                        code.classList.contains("language-text") ||
                        code.classList.contains("language-shell") ||
                        code.classList.contains("language-sh") ||
                        rawText.includes("error[E") ||
                        rawText.includes("--> ") ||
                        rawText.includes("Compiling ");

      // Diagnostic & Terminal Styling
      if (isConsole) {
        pre.classList.add("terminal-block");
        if (rawText.includes("error[E") || rawText.includes("error:") || rawText.includes("FAILED")) {
          pre.classList.add("diag-error-block");
        } else if (rawText.includes("warning:")) {
          pre.classList.add("diag-warn-block");
        }

        const badge = document.createElement("div");
        badge.className = "terminal-badge";
        badge.textContent = rawText.includes("error[E") || rawText.includes("error:")
          ? "Compiler Diagnostic"
          : (rawText.includes("test ") ? "Test Output" : "Terminal");
        pre.appendChild(badge);

        code.innerHTML = highlightConsoleOutput(rawText);
      } else if (isRust) {
        // Check for boring spans
        const boringSpans = code.querySelectorAll(".boring");
        const hasBoring = boringSpans.length > 0;

        if (hasBoring) {
          boringSpans.forEach((span) => {
            span.innerHTML = highlightRustCode(span.textContent || "");
          });
        }

        // Highlight nodes outside boring spans
        Array.from(code.childNodes).forEach((node) => {
          if (node.nodeType === Node.TEXT_NODE && node.textContent) {
            const span = document.createElement("span");
            span.innerHTML = highlightRustCode(node.textContent);
            node.parentNode.replaceChild(span, node);
          }
        });
      }

      // Add Header Bar (Copy + Toggle Boring)
      const headerBar = document.createElement("div");
      headerBar.className = "code-header-bar";

      // Copy Button
      const copyBtn = document.createElement("button");
      copyBtn.className = "copy-code-btn";
      copyBtn.type = "button";
      copyBtn.innerHTML = '📋 Copy';
      copyBtn.title = "Copy code to clipboard";

      copyBtn.addEventListener("click", () => {
        let textToCopy = "";
        if (code.querySelector(".boring") && !pre.classList.contains("show-boring")) {
          const clone = code.cloneNode(true);
          clone.querySelectorAll(".boring").forEach((b) => b.remove());
          textToCopy = clone.textContent;
        } else {
          textToCopy = code.textContent;
        }

        navigator.clipboard.writeText(textToCopy.trim()).then(() => {
          copyBtn.innerHTML = '✓ Copied';
          copyBtn.classList.add("copied");
          setTimeout(() => {
            copyBtn.innerHTML = '📋 Copy';
            copyBtn.classList.remove("copied");
          }, 2000);
        });
      });

      headerBar.appendChild(copyBtn);

      // Toggle Hidden Lines Button (for Rust snippets with # boilerplate)
      if (code.querySelector(".boring")) {
        const toggleBtn = document.createElement("button");
        toggleBtn.className = "toggle-boring-btn";
        toggleBtn.type = "button";
        toggleBtn.innerHTML = '👁 Show hidden';
        toggleBtn.title = "Show hidden doctest boilerplate lines";

        toggleBtn.addEventListener("click", () => {
          const show = pre.classList.toggle("show-boring");
          toggleBtn.innerHTML = show ? '👁 Hide' : '👁 Show hidden';
        });

        headerBar.appendChild(toggleBtn);
      }

      pre.appendChild(headerBar);
    });
  }

  /* ==========================================================================
     Chapter Loader & Renderer
     ========================================================================== */

  async function load(slug, heading) {
    if (!manifest) return;
    const item = manifest.chapters.find((entry) => entry.slug === slug) || manifest.chapters[0];
    if (item.slug !== slug) history.replaceState({}, "", "#/" + item.slug);

    chapter.innerHTML = '<div class="loading-indicator">Loading ' + escapeHtml(item.title) + '...</div>';

    const response = await fetch(basePath() + "chapters/" + item.slug + ".html");
    if (!response.ok) throw new Error("Chapter failed to load (" + response.status + ")");

    chapter.innerHTML = await response.text();
    document.title = item.title + " - The Incin Book";

    // Update active sidebar entry
    nav.querySelectorAll("a[data-slug]").forEach((link) => {
      if (link.dataset.slug === item.slug) {
        link.setAttribute("aria-current", "page");
        // Ensure section details is open
        const details = link.closest("details");
        if (details) details.open = true;
      } else {
        link.removeAttribute("aria-current");
      }
    });

    const index = chapterIndex(item.slug);
    updateNavLinks(index);
    enhanceCodeBlocks(chapter);

    if (window.innerWidth <= 900) {
      setSidebarOpen(false);
    }

    requestAnimationFrame(() => {
      const target = heading && document.getElementById(heading);
      if (target) target.scrollIntoView();
      else window.scrollTo(0, 0);
      chapter.focus({ preventScroll: true });
    });
  }

  async function loadRoute() {
    const target = route();
    try {
      await load(target.slug, target.heading);
    } catch (error) {
      chapter.innerHTML = "<h1>Chapter unavailable</h1><p>" + escapeHtml(String(error)) + "</p>";
    }
  }

  function navigate(event) {
    const anchor = event.target.closest("a[href^='#/']");
    if (anchor && anchor.hash) {
      event.preventDefault();
      history.pushState({}, "", anchor.hash);
      loadRoute();
    }
  }

  /* ==========================================================================
     Search Implementation
     ========================================================================== */

  function toggleSearch(show) {
    if (show === undefined) show = searchWrapper.classList.contains("hidden");
    searchWrapper.classList.toggle("hidden", !show);
    if (show) {
      search.focus();
      search.select();
    } else {
      search.value = "";
      searchResults.hidden = true;
      selectedResult = -1;
    }
  }

  function updateSearchSelection() {
    searchResults.querySelectorAll("a[role=option]").forEach((link, index) => {
      link.setAttribute("aria-selected", String(index === selectedResult));
      if (index === selectedResult) link.scrollIntoView({ block: "nearest" });
    });
  }

  function doSearch() {
    const query = search.value.trim().toLowerCase();
    searchResults.replaceChildren();
    searchResults.hidden = !query;
    selectedResult = -1;
    if (!query) return;

    const matches = searchIndex
      .filter((item) => (item.title + " " + item.text).toLowerCase().includes(query))
      .slice(0, 10);

    if (!matches.length) {
      const empty = document.createElement("div");
      empty.style.padding = "0.75rem 1.25rem";
      empty.style.color = "var(--sidebar-group)";
      empty.textContent = "No matching chapters found";
      searchResults.append(empty);
      return;
    }

    matches.forEach((item) => {
      const link = document.createElement("a");
      link.href = "#/" + item.slug;
      link.textContent = item.title;
      link.setAttribute("role", "option");
      link.setAttribute("aria-selected", "false");
      link.addEventListener("click", () => toggleSearch(false));
      searchResults.append(link);
    });
  }

  /* ==========================================================================
     Event Listeners & Bootstrapping
     ========================================================================== */

  // Sidebar toggle
  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => {
      const isOpen = window.innerWidth <= 900
        ? body.classList.contains("sidebar-open")
        : !body.classList.contains("sidebar-hidden");
      setSidebarOpen(!isOpen);
    });
  }

  // Theme dropdown toggle
  if (themeToggle) {
    themeToggle.addEventListener("click", (e) => {
      e.stopPropagation();
      const current = html.dataset.theme || "navy";
      const nextTheme = current === "navy" ? "light" : (current === "light" ? "rust" : (current === "rust" ? "coal" : (current === "coal" ? "ayu" : "navy")));
      setTheme(nextTheme);
      if (themeList) {
        themeList.hidden = !themeList.hidden;
        themeToggle.setAttribute("aria-expanded", String(!themeList.hidden));
      }
    });

    if (themeList) {
      themeList.querySelectorAll(".theme-option").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          setTheme(btn.dataset.theme);
          themeList.hidden = true;
          themeToggle.setAttribute("aria-expanded", "false");
        });
      });
    }

    document.addEventListener("click", (e) => {
      if (themeList && !themeToggle.contains(e.target) && !themeList.contains(e.target)) {
        themeList.hidden = true;
        themeToggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  // Search
  if (searchToggle) {
    searchToggle.addEventListener("click", () => toggleSearch(true));
  }

  if (searchWrapper) {
    searchWrapper.addEventListener("click", (e) => {
      if (e.target === searchWrapper) toggleSearch(false);
    });
  }

  if (search) {
    search.addEventListener("input", doSearch);
    search.addEventListener("keydown", (event) => {
      const options = searchResults.querySelectorAll("a[role=option]");
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        if (!options.length) return;
        event.preventDefault();
        selectedResult = (selectedResult + (event.key === "ArrowDown" ? 1 : options.length - 1)) % options.length;
        updateSearchSelection();
      } else if (event.key === "Enter" && selectedResult >= 0) {
        event.preventDefault();
        options[selectedResult].click();
      } else if (event.key === "Escape") {
        toggleSearch(false);
      }
    });
  }

  // Global Keybindings
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, textarea, select, button")) return;

    if (event.key === "/" || event.key === "s") {
      event.preventDefault();
      toggleSearch(true);
    } else if (event.key === "Escape") {
      toggleSearch(false);
    } else if (event.key === "ArrowLeft" && !previous.hidden) {
      window.location.hash = previous.hash;
    } else if (event.key === "ArrowRight" && !next.hidden) {
      window.location.hash = next.hash;
    }
  });

  document.addEventListener("click", navigate);
  window.addEventListener("popstate", loadRoute);
  window.addEventListener("hashchange", loadRoute);

  // Initialize
  (async function init() {
    initTheme();
    initSidebar();

    try {
      [manifest, searchIndex] = await Promise.all([
        fetch(basePath() + "chapters.json").then((r) => r.json()),
        fetch(basePath() + "search-index.json").then((r) => r.json()),
      ]);
      await loadRoute();
    } catch (err) {
      chapter.innerHTML = "<h1>Initialization error</h1><p>" + escapeHtml(String(err)) + "</p>";
    }
  }());
}());

