/* Capability reference behaviour.
 *
 * The data is injected at build time into a JSON script tag rather than
 * fetched, so the page works from a file:// checkout and has no load-failure
 * path. Theme switching writes the same storage keys book.js does, so a theme
 * chosen here carries back to the book and vice versa. */
(function () {
  "use strict";

  var el = document.getElementById("api-data");
  var DATA;
  try {
    DATA = JSON.parse(el.textContent);
  } catch (e) {
    el.insertAdjacentHTML("afterend",
      '<p class="api-empty">The capability payload did not parse. Re-run ' +
      '<code>python3 tools/build-api-site-data.py</code> and rebuild the site.</p>');
    return;
  }

  /* -- escaping ----------------------------------------------------------
     Every string below that reaches innerHTML is build-generated JSON, but
     CodeQL (and any future hand-written row) treats that as an XSS sink.
     Escape text and validate URLs at the boundary so a `<` in a doc comment
     or a `javascript:` href can never become markup. */
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
  function safeUrl(u, fallback) {
    var s = String(u == null ? "" : u);
    if (/^(https?:\/\/|\.\/|\.\.\/|#|mailto:)/.test(s)) return s;
    return fallback || "#";
  }

  /* -- theme, shared with the book -------------------------------------- */
  var THEMES = ["navy", "rust", "light", "coal", "ayu"];
  var html = document.documentElement;
  var toggle = document.getElementById("theme-toggle");
  var list = document.getElementById("theme-list");

  function setTheme(t) {
    if (THEMES.indexOf(t) < 0) t = "navy";
    html.className = t;
    html.dataset.theme = t;
    try {
      localStorage.setItem("incin-book-theme", t);
      localStorage.setItem("mdbook-theme", t);
    } catch (e) { /* private mode: the page still themes for this visit */ }
    list.querySelectorAll(".theme-option").forEach(function (b) {
      b.classList.toggle("active", b.dataset.theme === t);
    });
  }
  setTheme(html.dataset.theme || "navy");

  toggle.addEventListener("click", function () {
    var open = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", open ? "false" : "true");
    list.hidden = open;
  });
  list.addEventListener("click", function (e) {
    var b = e.target.closest(".theme-option");
    if (!b) return;
    setTheme(b.dataset.theme);
    list.hidden = true;
    toggle.setAttribute("aria-expanded", "false");
  });
  document.addEventListener("click", function (e) {
    if (!list.hidden && !e.target.closest(".theme-wrapper")) {
      list.hidden = true;
      toggle.setAttribute("aria-expanded", "false");
    }
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !list.hidden) {
      list.hidden = true;
      toggle.setAttribute("aria-expanded", "false");
      toggle.focus();
    }
  });

  /* -- matrix ------------------------------------------------------------ */
  var B = DATA.backends;
  var LABEL = { cpu: "CPU", cuda: "CUDA", wgpu: "WGPU", metal: "Metal" };
  var state = { q: "", backends: new Set(), traits: new Set() };

  /* Percentages land on the same nine steps the operation meters use, so a
     colour means the same proportion wherever it appears on the page. */
  function step9(pct) {
    return Math.min(9, Math.max(1, Math.round((pct / 100) * 9)));
  }

  document.getElementById("apiMetrics").innerHTML = B.map(function (b) {
    var sup = DATA.operations.filter(function (o) { return o.backends[b].dtypes.length; });
    var native = sup.filter(function (o) { return o.backends[b].impl === "native"; }).length;
    var strided = sup.filter(function (o) { return o.backends[b].layouts.indexOf("strided") >= 0; }).length;
    var pct = Math.round((sup.length / DATA.operations.length) * 100);
    return '<div class="api-metric">' +
      '<div class="b">' + sup.length + '<span> / ' + DATA.operations.length + '</span></div>' +
      '<div class="l">' + LABEL[b] + ' operations</div>' +
      '<div class="api-bar"><i class="cov-' + step9(pct) + '" style="width:' + pct + '%"></i></div>' +
      '<div class="s">' + native + ' native &middot; ' + strided + ' strided</div>' +
      '</div>';
  }).join("");

  /* `after` is which list to redraw: the chips are used by both the operation
     table and the type reference, and calling the operations renderer from the
     type chips would filter the wrong list. */
  function chipRow(host, items, bag, after) {
    host.innerHTML = items.map(function (it) {
      return '<button class="api-chip" type="button" aria-pressed="false" data-k="' + esc(it) + '">' + esc(it) + '</button>';
    }).join("");
    host.addEventListener("click", function (e) {
      var btn = e.target.closest(".api-chip");
      if (!btn) return;
      var on = btn.getAttribute("aria-pressed") === "true";
      btn.setAttribute("aria-pressed", on ? "false" : "true");
      if (on) { bag.delete(btn.dataset.k); } else { bag.add(btn.dataset.k); }
      (after || render)();
    });
  }
  chipRow(document.getElementById("apiBackendChips"), B, state.backends);
  chipRow(document.getElementById("apiTraitChips"), ["strided", "training", "composed"], state.traits);

  document.getElementById("apiQ").addEventListener("input", function (e) {
    state.q = e.target.value.trim().toLowerCase();
    render();
  });

  var rows = document.getElementById("apiRows");
  var countEl = document.getElementById("apiCount");

  /* Every bar on this page is drawn against the same fixed whole, so a bar
     means one thing everywhere. An earlier version scaled each row against the
     best backend on that row; because that denominator was 1 on 26 rows, 4 on
     81 and 8 on 54, a completely full bar meant four different amounts of
     support, and no two rows could be compared. The per-row best is still
     worth knowing -- it says whether an operation could do more at all -- so
     it is drawn as a tick on the track instead of as the scale. */
  var DTYPES = (function () {
    var seen = {};
    DATA.operations.forEach(function (o) {
      B.forEach(function (b) {
        o.backends[b].dtypes.forEach(function (d) { seen[d] = 1; });
      });
    });
    return Object.keys(seen).length;
  })();

  function widest(op) {
    return B.reduce(function (n, b) {
      return Math.max(n, op.backends[b].dtypes.length);
    }, 0);
  }

  /* The step a coverage figure falls on, 1..STEPS. The scale is deliberately
     stepped rather than continuous: the underlying quantity is a count of
     element types, so a reader comparing two bars is comparing whole types,
     and a smooth ramp would invent precision the data does not have. */
  var STEPS = 9;
  function step(have) {
    var n = Math.round((have / DTYPES) * STEPS);
    return Math.min(STEPS, Math.max(1, n));
  }

  function meter(b, e, best) {
    if (!e.dtypes.length) {
      return '<div class="api-meter none"><span class="lab">' + esc(b) + '<b>&mdash;</b></span>' +
        '<span class="api-track"></span></div>';
    }
    var have = e.dtypes.length;
    var pct = (have / DTYPES) * 100;
    // The tick earns its place only where it says something the fill does not:
    // that another backend reaches further on this same operation.
    var tick = best > have
      ? '<u class="api-ref" style="left:' + ((best / DTYPES) * 100).toFixed(2) + '%"></u>'
      : "";
    return '<div class="api-meter"><span class="lab">' + esc(b) +
      '<b>' + esc(have) + '<em>/' + esc(DTYPES) + '</em></b></span>' +
      '<span class="api-track"><i class="api-fill cov-' + step(have) +
      '" style="width:' + pct.toFixed(2) + '%"></i>' + tick + '</span></div>';
  }

  /* Which backends run this operation by composing others rather than with a
     kernel of their own. 29 of the 173 operations are composed somewhere, so
     marking them beside the name reads as a signal rather than as noise. */
  function composedOn(op) {
    return B.filter(function (b) { return op.backends[b].impl === "composed"; });
  }

  /* The public entry point, written the way the reader will type it. `::abs`
     is a method on a tensor, so it is shown as `Tensor::abs`; `Adam::step`
     already names its owner. */
  function entry(op) {
    var c = op.catalog;
    if (!c || !c.api) return "";
    var api = c.api.indexOf("::") === 0 ? "Tensor" + c.api : c.api;
    return '<code class="api-sig">' + esc(api) + '</code>';
  }

  function describe(op) {
    var c = op.catalog;
    if (c && c.doc) return '<div class="api-desc">' + esc(c.doc) + '</div>';
    if (!c) return "";
    var operands = c.arity[0] === c.arity[1]
      ? c.arity[0] + (c.arity[0] === 1 ? " operand" : " operands")
      : c.arity[0] + "\u2013" + c.arity[1] + " operands";
    var bits = [c.kind.toLowerCase(), operands];
    if (c.attrs && c.attrs !== "NoAttributes") bits.push(c.attrs);
    return '<div class="api-desc structural">' + esc(bits.join(" \u00b7 ")) + "</div>";
  }

  function matches(op) {
    if (state.q) {
      var hay = op.name + " " + ((op.catalog && op.catalog.doc) || "") +
                " " + ((op.catalog && op.catalog.family) || "");
      if (hay.toLowerCase().indexOf(state.q) < 0) return false;
    }
    var b;
    for (b of state.backends) { if (!op.backends[b].dtypes.length) return false; }
    var scope = state.backends.size ? Array.from(state.backends) : B;
    var t;
    for (t of state.traits) {
      var ok = scope.some(function (bk) {
        var e = op.backends[bk];
        if (!e.dtypes.length) return false;
        if (t === "strided") return e.layouts.indexOf("strided") >= 0;
        if (t === "training") return e.training === true;
        if (t === "composed") return e.impl === "composed";
        return false;
      });
      if (!ok) return false;
    }
    return true;
  }

  /* The reference link lives here rather than in the row: the row is a button,
     and an anchor inside a button is invalid and would fight the expander for
     the click. 141 of the 173 operations resolve to a checked item anchor on
     docs.rs; the rest carry a search, because the published release predates
     them or documents them under another name, and a link that lands nowhere
     is worse than one that admits it is a search. */
  function docsLink(op) {
    var d = op.docs;
    if (!d) return "";
    var label = d.kind === "method"
      ? d.item + " on docs.rs"
      : "search docs.rs for " + d.item;
    return '<div class="api-dcell"><h4>documentation</h4>' +
      '<a class="api-docs" href="' + esc(safeUrl(d.url, "#")) + '" target="_blank" rel="noopener noreferrer">' +
      esc(label) + ' \u2197</a>' +
      (d.kind === "search"
        ? '<p class="api-none">not published under this name in the released version</p>'
        : "") +
      '</div>';
  }

  function detailFor(op) {
    var c = op.catalog;
    var head = "";
    if (c) {
      head = '<div class="api-dcell"><h4>catalog</h4>' +
        '<div class="api-kv"><span class="k">family</span><span class="v">' + esc(c.family) + '</span></div>' +
        '<div class="api-kv"><span class="k">category</span><span class="v">' + esc(c.kind) + '</span></div>' +
        '<div class="api-kv"><span class="k">operands</span><span class="v">' +
          esc(c.arity[0] === c.arity[1] ? c.arity[0] : c.arity[0] + "\u2013" + c.arity[1]) + '</span></div>' +
        '<div class="api-kv"><span class="k">attributes</span><span class="v">' + esc(c.attrs) + '</span></div>' +
        (c.api ? '<div class="api-kv"><span class="k">reached via</span><span class="v">' + esc(c.api) + '</span></div>' : "") +
        '</div>';
    }
    /* The guide fills when the row opens: the usage payload is fetched on
       demand, and the detail node does not exist before that. The example
       cell spans the full row so code never squeezes into a 215px column. */
    var example = '<div class="api-dcell api-example"><h4>example</h4>' +
      '<div class="api-usage" data-keys="' + esc(usageKeysForOp(op).join(" ")) + '">' +
      '<p class="api-none">Loading example&hellip;</p></div></div>';
    return '<div class="api-detail-in">' + docsLink(op) + head + B.map(function (b) {
      var e = op.backends[b];
      if (!e.dtypes.length) {
        return '<div class="api-dcell"><h4>' + esc(b) + '</h4>' +
          '<p class="api-none">not advertised &mdash; refused by the registry</p></div>';
      }
      return '<div class="api-dcell on"><h4>' + esc(b) + '</h4>' +
        '<div class="api-kv"><span class="k">dtypes</span><span class="v"><span class="api-dts">' +
          e.dtypes.map(function (d) { return '<span class="api-dt">' + esc(d) + '</span>'; }).join("") +
        '</span></span></div>' +
        '<div class="api-kv"><span class="k">layouts</span><span class="v">' + esc(e.layouts.join(", ") || "\u2014") + '</span></div>' +
        '<div class="api-kv"><span class="k">rank</span><span class="v">' + esc(e.rank || "\u2014") + '</span></div>' +
        '<div class="api-kv"><span class="k">training</span><span class="v">' + (e.training ? "yes" : "no") + '</span></div>' +
        '<div class="api-kv"><span class="k">impl</span><span class="v">' + esc(e.impl || "\u2014") + '</span></div>' +
        '</div>';
    }).join("") + example + '</div>';
  }

  function render() {
    var list2 = DATA.operations.filter(matches);
    countEl.textContent = list2.length + " of " + DATA.operations.length;
    if (!list2.length) {
      rows.innerHTML = '<p class="api-empty">No operation matches those filters.</p>';
      return;
    }
    rows.innerHTML = list2.map(function (op, i) {
      var best = widest(op);
      var comp = composedOn(op);
      var dot = comp.length
        ? '<span class="api-dot" title="composed on ' + esc(comp.join(", ")) +
          ' \u2014 built from other operations rather than a kernel of its own"></span>'
        : "";
      return '<button type="button" class="api-row" aria-expanded="false" data-i="' + i + '">' +
        '<span class="api-id"><span class="api-name">' + esc(op.name) + dot +
          (op.catalog ? '<span class="api-fam">' + esc(op.catalog.family) + '</span>' : "") +
        '</span>' + entry(op) + describe(op) + '</span>' +
        '<span class="api-strip">' +
          B.map(function (b) { return meter(b, op.backends[b], best); }).join("") +
        '</span></button>' +
        '<div class="api-detail" data-d="' + i + '" hidden></div>';
    }).join("");
    rows._list = list2;
  }

  function toggleRow(btn) {
    var i = btn.dataset.i;
    var det = rows.querySelector('.api-detail[data-d="' + i + '"]');
    var open = btn.getAttribute("aria-expanded") === "true";
    btn.setAttribute("aria-expanded", open ? "false" : "true");
    if (open) { det.hidden = true; return; }
    if (!det.innerHTML) {
      det.innerHTML = detailFor(rows._list[i]);
      var box = det.querySelector(".api-usage[data-keys]");
      if (box) fillUsagePanel(box.dataset.keys.split(" "), box);
    }
    det.hidden = false;
  }
  rows.addEventListener("click", function (e) {
    var btn = e.target.closest(".api-row");
    if (btn) toggleRow(btn);
  });

  var cats = [];
  B.forEach(function (b) {
    (DATA.categories[b] || []).forEach(function (r) { cats.push([b, r]); });
  });
  document.getElementById("apiCatRows").innerHTML = cats.map(function (p) {
    var b = p[0], r = p[1];
    return '<tr><td class="name">' + esc(b) + '</td><td class="name">' + esc(r.operation) + '</td>' +
      '<td><span class="api-dts">' +
        (r.dtypes.map(function (d) { return '<span class="api-dt">' + esc(d) + '</span>'; }).join("") ||
         '<span class="api-none">&mdash;</span>') +
      '</span></td>' +
      '<td class="name">' + esc(r.layouts.join(", ") || "\u2014") + '</td>' +
      '<td class="name">' + esc(r.rank) + '</td>' +
      '<td class="name">' + (r.training ? "yes" : "no") + '</td>' +
      '<td class="name">' + esc(r.impl) + '</td></tr>';
  }).join("");

  document.getElementById("apiSurface").innerHTML = Object.keys(DATA.surface)
    .sort(function (a, b) { return DATA.surface[b] - DATA.surface[a]; })
    .map(function (k) {
      return '<div class="api-scell"><span class="n">' + esc(k) + '</span><span class="c">' +
        Number(DATA.surface[k]).toLocaleString() + '</span></div>';
    }).join("");

  render();

  /* -- sections ---------------------------------------------------------- */
  /* The type reference is fetched the first time it is opened rather than
     inlined: it is several times the weight of everything else on this page,
     and most readers never open it. */
  var TABS = document.getElementById("apiTabs");
  var typesLoaded = false;

  function showSection(id) {
    [].forEach.call(document.querySelectorAll(".api-sec"), function (sec) {
      sec.hidden = sec.id !== "sec-" + id;
    });
    [].forEach.call(TABS.querySelectorAll(".api-tab"), function (tab) {
      tab.setAttribute("aria-selected", String(tab.dataset.sec === id));
    });
    if (id === "types" && !typesLoaded) { loadTypes(); }
    if (window.location.hash.slice(1) !== id) {
      history.replaceState({}, "", "#" + id);
    }
  }

  TABS.addEventListener("click", function (e) {
    var tab = e.target.closest(".api-tab");
    if (tab) showSection(tab.dataset.sec);
  });

  /* -- element types ----------------------------------------------------- */
  /* The card itself is not the button: the usage lines load on demand, and a
     card that fetched on render would pull the payload for readers who never
     open this tab -- and would print a load error into every card for readers
     on a file:// checkout, where fetch cannot run at all. */
  document.getElementById("dtypeCards").innerHTML = (DATA.dtypes || []).map(function (d) {
    var e = d.encoding;
    var store = e
      ? (e.elementsPerBlock === 1
          ? e.bytesPerBlock + (e.bytesPerBlock === 1 ? " byte" : " bytes") + " per element"
          : e.elementsPerBlock + " values packed into " + e.bytesPerBlock + " bytes")
        + " \u00b7 " + e.bitsPerElement + " bits/element \u00b7 " + e.alignment + "-byte aligned"
      : "";
    return '<div class="api-card"><h3><code>' + esc(d.id) + '</code>' +
      (e ? '<span class="api-fam">' + esc(e.kind) + '</span>' : "") + '</h3>' +
      '<p>' + esc(d.doc) + '</p>' +
      (e ? '<div class="api-kv"><span class="k">storage</span><span class="v">' + esc(store) +
           '</span></div>' : "") +
      '<div class="api-kv"><span class="k">operations</span><span class="v">' +
        esc(d.operations) + ' of ' + esc(DATA.operations.length) + '</span></div>' +
      '<div class="api-kv"><span class="k">backends</span><span class="v">' +
        (d.backends.length
          ? d.backends.map(function (b) { return '<span class="api-dt">' + esc(b) + '</span>'; }).join("")
          : "&mdash;") +
      '</span></div>' +
      '<button type="button" class="api-exbtn" data-usage="' + esc(d.id) +
        '" aria-expanded="false">Example</button>' +
      '<div class="api-usage" hidden></div></div>';
  }).join("");

  /* -- backends ---------------------------------------------------------- */
  document.getElementById("backendCards").innerHTML = (DATA.backendDetail || []).map(function (b) {
    var pct = Math.round((b.operations / DATA.operations.length) * 100);
    return '<div class="api-card"><h3>' + esc(LABEL[b.id] || b.id) + '</h3>' +
      '<div class="api-bar"><i class="cov-' + step9(pct) +
        '" style="width:' + pct + '%"></i></div>' +
      '<div class="api-kv"><span class="k">operations</span><span class="v">' +
        esc(b.operations) + ' of ' + esc(DATA.operations.length) + '</span></div>' +
      '<div class="api-kv"><span class="k">own kernel</span><span class="v">' + esc(b.native) + '</span></div>' +
      '<div class="api-kv"><span class="k">composed</span><span class="v">' + esc(b.composed) + '</span></div>' +
      '<div class="api-kv"><span class="k">accepts strided</span><span class="v">' + esc(b.strided) + '</span></div>' +
      '<div class="api-kv"><span class="k">covers training</span><span class="v">' + esc(b.training) + '</span></div>' +
      '<div class="api-kv"><span class="k">element types</span><span class="v">' +
        b.dtypes.map(function (d) { return '<span class="api-dt">' + esc(d) + '</span>'; }).join("") +
      '</span></div></div>';
  }).join("");

  /* -- data flow renders with the authoring guide below, next to the code
     it illustrates. */

  /* -- types ------------------------------------------------------------- */
  var TYPES = null;
  var tyState = { q: "", kinds: new Set(), crates: new Set() };
  var TY_CAP = 300;

  function loadTypes() {
    typesLoaded = true;
    var host = document.getElementById("tyRows");
    host.innerHTML = '<p class="api-empty">Loading the type reference&hellip;</p>';
    fetch("api-types.json").then(function (r) {
      if (!r.ok) throw new Error("api-types.json responded " + r.status);
      return r.json();
    }).then(function (payload) {
      TYPES = payload.types;
      chipRow(document.getElementById("tyKindChips"),
        ["struct", "trait", "enum", "type"], tyState.kinds, renderTypes);
      chipRow(document.getElementById("tyCrateChips"),
        uniq(TYPES.map(function (t) { return t.crate; })), tyState.crates, renderTypes);
      renderTypes();
    }).catch(function (error) {
      host.innerHTML = '<p class="api-empty">The type reference could not be loaded: ' +
        esc(String(error.message || error)) + '</p>';
    });
  }

  function uniq(list) {
    var seen = {};
    return list.filter(function (v) {
      if (seen[v]) return false;
      seen[v] = 1;
      return true;
    }).sort();
  }

  function renderTypes() {
    if (!TYPES) return;
    var q = tyState.q.trim().toLowerCase();
    var list = TYPES.filter(function (t) {
      if (tyState.kinds.size && !tyState.kinds.has(t.kind)) return false;
      if (tyState.crates.size && !tyState.crates.has(t.crate)) return false;
      if (!q) return true;
      return t.name.toLowerCase().indexOf(q) >= 0 || t.module.toLowerCase().indexOf(q) >= 0;
    });
    var shown = list.slice(0, TY_CAP);
    document.getElementById("tyCount").textContent = list.length > shown.length
      ? shown.length + " of " + list.length + " — refine to see the rest"
      : list.length + " of " + TYPES.length;
    document.getElementById("tyRows").innerHTML = shown.length
      ? shown.map(function (t) {
          var where = t.module ? t.crate.replace(/-/g, "_") + "::" + t.module : t.crate.replace(/-/g, "_");
          var name = t.url
            ? '<a class="api-tyname" href="' + esc(safeUrl(t.url, "#")) + '" target="_blank" rel="noopener noreferrer">' +
              esc(t.name) + ' ↗</a>'
            : '<span class="api-tyname plain">' + esc(t.name) + '</span>';
          return expandable(t.name,
            '<span class="api-tykind ' + esc(t.kind) + '">' + esc(t.kind) + '</span>' +
            '<span class="api-tyname plain">' + esc(t.name) + '</span>' +
            '<code class="api-typath">' + esc(where) + '</code>', "api-tyrow open") +
            (t.url ? '<a class="api-tylink" href="' + esc(safeUrl(t.url, "#")) +
              '" target="_blank" rel="noopener noreferrer">docs.rs \u2197</a>' : "");
        }).join("")
      : '<p class="api-empty">No type matches those filters.</p>';
  }

  var tyQ = document.getElementById("tyQ");
  tyQ.addEventListener("input", function () { tyState.q = tyQ.value; renderTypes(); });

  /* -- layouts ----------------------------------------------------------- */
  document.getElementById("layoutCards").innerHTML = ((DATA.layouts || {}).items || [])
    .map(function (i) {
      return '<div class="api-card">' + expandable(i.name,
        '<h3><code>' + esc(i.name) + '</code><span class="api-fam">' + esc(i.kind) + '</span></h3>' +
        '<p>' + esc(i.doc) + '</p>', "api-cardbtn") + '</div>';
    }).join("");

  document.getElementById("layoutRows").innerHTML = ((DATA.layouts || {}).byBackend || [])
    .map(function (b) {
      var total = DATA.operations.length;
      function bar(label, n) {
        var pct = Math.round((n / total) * 100);
        return '<div class="api-meter"><span class="lab">' + esc(label) +
          '<b>' + esc(n) + '<em>/' + esc(total) + '</em></b></span>' +
          '<span class="api-track"><i class="api-fill cov-' + step9(pct) +
          '" style="width:' + pct + '%"></i></span></div>';
      }
      return '<div class="api-lrow"><span class="api-lname">' + esc(LABEL[b.id] || b.id) + '</span>' +
        '<span class="api-strip two">' + bar("contiguous", b.contiguous) +
        bar("strided", b.strided) + '</span></div>';
    }).join("");

  /* -- shapes ------------------------------------------------------------ */
  function renderShapes() {
    var q = (document.getElementById("shQ").value || "").trim().toLowerCase();
    var groups = (DATA.shapes || []).map(function (g) {
      var items = g.items.filter(function (i) {
        return !q || i.name.toLowerCase().indexOf(q) >= 0 || i.doc.toLowerCase().indexOf(q) >= 0;
      });
      return { module: g.module, items: items };
    }).filter(function (g) { return g.items.length; });
    var shown = groups.reduce(function (n, g) { return n + g.items.length; }, 0);
    var all = (DATA.shapes || []).reduce(function (n, g) { return n + g.items.length; }, 0);
    document.getElementById("shCount").textContent = shown + " of " + all;
    document.getElementById("shapeGroups").innerHTML = groups.length
      ? groups.map(function (g) {
          return '<section class="api-grp"><h3 class="api-h"><code>shapes::' + esc(g.module) +
            '</code></h3><div class="api-list">' + g.items.map(function (i) {
              return expandable(i.name,
                '<span class="api-tykind">' + esc(i.kind) + '</span>' +
                '<span class="api-tyname plain">' + esc(i.name) + '</span>' +
                '<span class="api-typath doc">' + esc(i.doc) + '</span>', "api-tyrow open");
            }).join("") + '</div></section>';
        }).join("")
      : '<p class="api-empty">No shape type matches that filter.</p>';
  }
  document.getElementById("shQ").addEventListener("input", renderShapes);
  renderShapes();

  /* -- target API -------------------------------------------------------- */
  document.getElementById("targetRows").innerHTML = (DATA.targetApi || []).map(function (m) {
    return expandable(m.name,
      '<span class="api-tykind">fn</span>' +
      '<span class="api-tyname plain">' + esc(m.name) + '</span>' +
      '<span class="api-typath doc">' + esc(m.doc) + '</span>', "api-tyrow open");
  }).join("");


  /* -- guide examples -------------------------------------------------- */
  /* Fetched on demand and shared by every section: opening an item shows the
     guide blocks that name it -- book chapters and runnable example programs,
     never test suites. A literal word match is a fact about the block, so the
     heading says "Example": the pool was written to teach, and an item with
     no block says so rather than borrowing an unrelated one. */
  var USAGE = null;
  var usagePending = null;

  function loadUsage() {
    if (USAGE) return Promise.resolve(USAGE);
    if (!usagePending) {
      usagePending = fetch("api-usage.json").then(function (r) {
        if (!r.ok) throw new Error("api-usage.json responded " + r.status);
        return r.json();
      }).then(function (payload) { USAGE = payload; return USAGE; });
    }
    return usagePending;
  }

  /* Every name the usage index might know one item under, space-separated in
     the button: the catalog wire name and the public method it is reached
     through. `broadcast_as` is why both travel -- no guide block writes the
     bare word `broadcast`, while several write the method. */
  function usageKeysForOp(op) {
    var keys = [op.name];
    var api = (op.catalog && op.catalog.api) || "";
    var method = api.indexOf("::") >= 0 ? api.split("::").pop() : "";
    if (method && keys.indexOf(method) < 0) keys.push(method);
    return keys;
  }

  function fallbackReference(names) {
    /* Rustdoc-style standalone reference for items with no guide block:
       a one-liner the reader can copy, not an empty panel. */
    var primary = names[names.length - 1] || names[0] || "item";
    var codeText = "use incin::prelude::*;\n\n// `" + primary +
      "` — full signature via the documentation link above.\n" +
      "// If you expected a worked example here, it has not been written yet.";
    return '<figure class="api-ex ref-fallback"><figcaption><a href="./#/">Reference — ' + esc(primary) + '</a>' +
      '<span class="api-extag">signature</span></figcaption>' +
      '<pre><code>' + code(codeText) + '</code></pre>' +
      '<p class="api-none" style="padding:8px 14px">No guide block names ' +
        names.map(function (n) { return '<code>' + esc(n) + '</code>'; }).join(" or ") +
        ' yet. <a href="./#/">Start from the book</a>.</p></figure>';
  }

  function usageHtml(names) {
    var seen = {}, ids = [];
    names.forEach(function (n) {
      ((USAGE && USAGE.index[n]) || []).forEach(function (i) {
        if (!seen[i]) { seen[i] = 1; ids.push(i); }
      });
    });
    /* Generic method names (`add`, `mul`, `sum`) match prose incidentally.
       Rank snippets that actually call the item (` .add(`, `::add`,
       `add!`) above ones that merely mention the word, so `.add` shows a
       tensor example instead of "Writing an executor". */
    var primary = names[names.length - 1] || names[0] || "";
    function score(i) {
      if (!primary) return 1;
      var src = (USAGE.snippets[i] && USAGE.snippets[i].code) || "";
      if (src.indexOf("." + primary + "(") >= 0 || src.indexOf("." + primary + "!") >= 0 ||
          src.indexOf("::" + primary) >= 0 || src.indexOf(primary + "!") >= 0) return 0;
      return 1;
    }
    ids.sort(function (a, b) { return score(a) - score(b); });
    ids = ids.slice(0, 3);
    if (!ids.length) {
      return fallbackReference(names);
    }
    return ids.map(function (i) {
      var sn = USAGE.snippets[i];
      var external = sn.origin !== "book";
      return '<figure class="api-ex"><figcaption>' +
        '<a href="' + esc(safeUrl(sn.href, "#")) + '"' +
        (external ? ' target="_blank" rel="noopener noreferrer"' : '') +
        '>' + esc(sn.label) + '</a>' +
        '<span class="api-extag' + (sn.checked ? " ok" : "") + '">' +
        (sn.origin === "book" ? "book" : "example") + ' \u00b7 ' +
        (sn.checked ? "compiled" : "not compiled") + '</span></figcaption>' +
        '<pre><code>' + code(sn.code) + '</code></pre></figure>';
    }).join("");
  }

  function fillUsagePanel(names, panel) {
    panel.hidden = false;
    if (panel.dataset.filled) return;
    panel.innerHTML = '<p class="api-none">Loading example&hellip;</p>';
    loadUsage().then(function () {
      panel.innerHTML = usageHtml(names);
      panel.dataset.filled = "1";
    }).catch(function (error) {
      panel.innerHTML = '<p class="api-none">Example could not be loaded: ' +
        esc(String(error.message || error)) + '</p>';
    });
  }

  document.addEventListener("click", function (e) {
    var item = e.target.closest("[data-usage]");
    if (!item) return;
    var panel = item.nextElementSibling;
    if (!panel || !panel.classList.contains("api-usage")) return;
    var open = item.getAttribute("aria-expanded") === "true";
    item.setAttribute("aria-expanded", open ? "false" : "true");
    if (open) { panel.hidden = true; return; }
    fillUsagePanel(item.dataset.usage.split(" "), panel);
  });

  function expandable(name, inner, cls) {
    return '<button type="button" class="' + esc(cls || "api-item") + '" data-usage="' + esc(name) +
      '" aria-expanded="false">' + inner + '</button>' +
      '<div class="api-usage" hidden></div>';
  }

  /* The book and this page share one highlighter, so a snippet reads the same
     in both places. It escapes as it tokenises, so the raw text goes in. */
  function code(text) {
    return window.incinHighlightRust
      ? window.incinHighlightRust(text)
      : text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  /* -- shape gallery ----------------------------------------------------- */
  /* The shapes chapter opens with the three kinds side by side; the gallery
     shows those opening blocks where the reference lists the types, resolved
     from the chapter at build time so the two cannot drift apart. */
  document.getElementById("shapeGallery").innerHTML = (DATA.shapeGallery || []).map(function (g) {
    return '<figure class="api-ex">' +
      '<figcaption><a href="' + esc(safeUrl(g.href, "#")) + '">' + esc(g.heading) + '</a>' +
      '<span class="api-extag' + (g.checked ? " ok" : "") + '">' +
      (g.checked ? "compiled" : "not compiled") + '</span></figcaption>' +
      '<pre><code>' + code(g.code) + '</code></pre></figure>';
  }).join("");

  /* -- authoring guide ----------------------------------------------------- */
  /* The data-flow tab walks a custom operation from declaration to dispatch,
     through the runnable polar example: contract, kernel, readout, backward.
     Steps resolve from that file at build time; a renamed function fails the
     build instead of silently dropping a step. */
  document.getElementById("flowSteps").innerHTML = (DATA.authoring || []).map(function (a) {
    return '<li class="api-step"><h3>' + esc(a.title) + '</h3>' +
      '<figure class="api-ex">' +
      '<figcaption><a href="' + esc(safeUrl(a.href, "#")) + '" target="_blank" rel="noopener noreferrer">' +
      esc(a.where) + '</a>' +
      '<span class="api-extag' + (a.checked ? " ok" : "") + '">' +
      (a.checked ? "compiled" : "not compiled") + '</span></figcaption>' +
      '<pre><code>' + code(a.code) + '</code></pre></figure></li>';
  }).join("");

  var initial = window.location.hash.slice(1);
  showSection(["types", "dtypes", "backends", "layouts", "shapes", "target", "flow"].indexOf(initial) >= 0 ? initial : "operations");

})();
