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

  document.getElementById("apiMetrics").innerHTML = B.map(function (b) {
    var sup = DATA.operations.filter(function (o) { return o.backends[b].dtypes.length; });
    var native = sup.filter(function (o) { return o.backends[b].impl === "native"; }).length;
    var strided = sup.filter(function (o) { return o.backends[b].layouts.indexOf("strided") >= 0; }).length;
    var pct = Math.round((sup.length / DATA.operations.length) * 100);
    return '<div class="api-metric">' +
      '<div class="b">' + sup.length + '<span> / ' + DATA.operations.length + '</span></div>' +
      '<div class="l">' + LABEL[b] + ' operations</div>' +
      '<div class="api-bar"><i style="width:' + pct + '%"></i></div>' +
      '<div class="s">' + native + ' native &middot; ' + strided + ' strided</div>' +
      '</div>';
  }).join("");

  /* `after` is which list to redraw: the chips are used by both the operation
     table and the type reference, and calling the operations renderer from the
     type chips would filter the wrong list. */
  function chipRow(host, items, bag, after) {
    host.innerHTML = items.map(function (it) {
      return '<button class="api-chip" type="button" aria-pressed="false" data-k="' + it + '">' + it + '</button>';
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
      return '<div class="api-meter none"><span class="lab">' + b + '<b>&mdash;</b></span>' +
        '<span class="api-track"></span></div>';
    }
    var have = e.dtypes.length;
    var pct = (have / DTYPES) * 100;
    // The tick earns its place only where it says something the fill does not:
    // that another backend reaches further on this same operation.
    var tick = best > have
      ? '<u class="api-ref" style="left:' + ((best / DTYPES) * 100).toFixed(2) + '%"></u>'
      : "";
    return '<div class="api-meter"><span class="lab">' + b +
      '<b>' + have + '<em>/' + DTYPES + '</em></b></span>' +
      '<span class="api-track"><i class="api-fill cov-' + step(have) +
      '" style="width:' + pct.toFixed(2) + '%"></i>' + tick + '</span></div>';
  }

  /* Which backends run this operation by composing others rather than with a
     kernel of their own. 29 of the 179 operations are composed somewhere, so
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
    return '<code class="api-sig">' + api + '</code>';
  }

  function describe(op) {
    var c = op.catalog;
    if (c && c.doc) return '<div class="api-desc">' + c.doc + '</div>';
    if (!c) return "";
    var operands = c.arity[0] === c.arity[1]
      ? c.arity[0] + (c.arity[0] === 1 ? " operand" : " operands")
      : c.arity[0] + "\u2013" + c.arity[1] + " operands";
    var bits = [c.kind.toLowerCase(), operands];
    if (c.attrs && c.attrs !== "NoAttributes") bits.push(c.attrs);
    return '<div class="api-desc structural">' + bits.join(" \u00b7 ") + "</div>";
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
     the click. 147 of the 179 operations resolve to a checked item anchor on
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
      '<a class="api-docs" href="' + d.url + '" target="_blank" rel="noopener noreferrer">' +
      label + ' \u2197</a>' +
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
        '<div class="api-kv"><span class="k">family</span><span class="v">' + c.family + '</span></div>' +
        '<div class="api-kv"><span class="k">category</span><span class="v">' + c.kind + '</span></div>' +
        '<div class="api-kv"><span class="k">operands</span><span class="v">' +
          (c.arity[0] === c.arity[1] ? c.arity[0] : c.arity[0] + "\u2013" + c.arity[1]) + '</span></div>' +
        '<div class="api-kv"><span class="k">attributes</span><span class="v">' + c.attrs + '</span></div>' +
        (c.api ? '<div class="api-kv"><span class="k">reached via</span><span class="v">' + c.api + '</span></div>' : "") +
        '</div>';
    }
    return '<div class="api-detail-in">' + docsLink(op) + head + B.map(function (b) {
      var e = op.backends[b];
      if (!e.dtypes.length) {
        return '<div class="api-dcell"><h4>' + b + '</h4>' +
          '<p class="api-none">not advertised &mdash; refused by the registry</p></div>';
      }
      return '<div class="api-dcell on"><h4>' + b + '</h4>' +
        '<div class="api-kv"><span class="k">dtypes</span><span class="v"><span class="api-dts">' +
          e.dtypes.map(function (d) { return '<span class="api-dt">' + d + '</span>'; }).join("") +
        '</span></span></div>' +
        '<div class="api-kv"><span class="k">layouts</span><span class="v">' + (e.layouts.join(", ") || "&mdash;") + '</span></div>' +
        '<div class="api-kv"><span class="k">rank</span><span class="v">' + (e.rank || "&mdash;") + '</span></div>' +
        '<div class="api-kv"><span class="k">training</span><span class="v">' + (e.training ? "yes" : "no") + '</span></div>' +
        '<div class="api-kv"><span class="k">impl</span><span class="v">' + (e.impl || "&mdash;") + '</span></div>' +
        '</div>';
    }).join("") + '</div>';
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
        ? '<span class="api-dot" title="composed on ' + comp.join(", ") +
          ' \u2014 built from other operations rather than a kernel of its own"></span>'
        : "";
      return '<button type="button" class="api-row" aria-expanded="false" data-i="' + i + '">' +
        '<span class="api-id"><span class="api-name">' + op.name + dot +
          (op.catalog ? '<span class="api-fam">' + op.catalog.family + '</span>' : "") +
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
    if (!det.innerHTML) { det.innerHTML = detailFor(rows._list[i]); }
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
    return '<tr><td class="name">' + b + '</td><td class="name">' + r.operation + '</td>' +
      '<td><span class="api-dts">' +
        (r.dtypes.map(function (d) { return '<span class="api-dt">' + d + '</span>'; }).join("") ||
         '<span class="api-none">&mdash;</span>') +
      '</span></td>' +
      '<td class="name">' + (r.layouts.join(", ") || "&mdash;") + '</td>' +
      '<td class="name">' + r.rank + '</td>' +
      '<td class="name">' + (r.training ? "yes" : "no") + '</td>' +
      '<td class="name">' + r.impl + '</td></tr>';
  }).join("");

  document.getElementById("apiSurface").innerHTML = Object.keys(DATA.surface)
    .sort(function (a, b) { return DATA.surface[b] - DATA.surface[a]; })
    .map(function (k) {
      return '<div class="api-scell"><span class="n">' + k + '</span><span class="c">' +
        DATA.surface[k].toLocaleString() + '</span></div>';
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
  document.getElementById("dtypeCards").innerHTML = (DATA.dtypes || []).map(function (d) {
    return '<div class="api-card"><h3><code>' + d.id + '</code></h3>' +
      '<p>' + d.doc + '</p>' +
      '<div class="api-kv"><span class="k">operations</span><span class="v">' +
        d.operations + ' of ' + DATA.operations.length + '</span></div>' +
      '<div class="api-kv"><span class="k">backends</span><span class="v">' +
        (d.backends.length
          ? d.backends.map(function (b) { return '<span class="api-dt">' + b + '</span>'; }).join("")
          : "&mdash;") +
      '</span></div></div>';
  }).join("");

  /* -- backends ---------------------------------------------------------- */
  document.getElementById("backendCards").innerHTML = (DATA.backendDetail || []).map(function (b) {
    var pct = Math.round((b.operations / DATA.operations.length) * 100);
    return '<div class="api-card"><h3>' + (LABEL[b.id] || b.id) + '</h3>' +
      '<div class="api-bar"><i class="cov-' + Math.min(9, Math.max(1, Math.round(pct / 11.2))) +
        '" style="width:' + pct + '%"></i></div>' +
      '<div class="api-kv"><span class="k">operations</span><span class="v">' +
        b.operations + ' of ' + DATA.operations.length + '</span></div>' +
      '<div class="api-kv"><span class="k">own kernel</span><span class="v">' + b.native + '</span></div>' +
      '<div class="api-kv"><span class="k">composed</span><span class="v">' + b.composed + '</span></div>' +
      '<div class="api-kv"><span class="k">accepts strided</span><span class="v">' + b.strided + '</span></div>' +
      '<div class="api-kv"><span class="k">covers training</span><span class="v">' + b.training + '</span></div>' +
      '<div class="api-kv"><span class="k">element types</span><span class="v">' +
        b.dtypes.map(function (d) { return '<span class="api-dt">' + d + '</span>'; }).join("") +
      '</span></div></div>';
  }).join("");

  /* -- data flow --------------------------------------------------------- */
  document.getElementById("flowSteps").innerHTML = (DATA.flow || []).map(function (f) {
    return '<li class="api-step"><h3>' + f.stage + '</h3><p>' + f.detail + '</p>' +
      (f.error ? '<code class="api-err">' + f.error + '</code>' : "") + '</li>';
  }).join("");

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
        String(error.message || error) + '</p>';
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
            ? '<a class="api-tyname" href="' + t.url + '" target="_blank" rel="noopener noreferrer">' +
              t.name + ' ↗</a>'
            : '<span class="api-tyname plain">' + t.name + '</span>';
          return '<div class="api-tyrow"><span class="api-tykind ' + t.kind + '">' + t.kind + '</span>' +
            name + '<code class="api-typath">' + where + '</code></div>';
        }).join("")
      : '<p class="api-empty">No type matches those filters.</p>';
  }

  var tyQ = document.getElementById("tyQ");
  tyQ.addEventListener("input", function () { tyState.q = tyQ.value; renderTypes(); });

  var initial = window.location.hash.slice(1);
  showSection(["types", "dtypes", "backends", "flow"].indexOf(initial) >= 0 ? initial : "operations");

})();
