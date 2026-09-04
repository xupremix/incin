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

  function chipRow(host, items, bag) {
    host.innerHTML = items.map(function (it) {
      return '<button class="api-chip" type="button" aria-pressed="false" data-k="' + it + '">' + it + '</button>';
    }).join("");
    host.addEventListener("click", function (e) {
      var btn = e.target.closest(".api-chip");
      if (!btn) return;
      var on = btn.getAttribute("aria-pressed") === "true";
      btn.setAttribute("aria-pressed", on ? "false" : "true");
      if (on) { bag.delete(btn.dataset.k); } else { bag.add(btn.dataset.k); }
      render();
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

  function meter(b, e, best) {
    if (!e.dtypes.length) {
      return '<div class="api-meter none"><span class="lab">' + b + '<b>&mdash;</b></span>' +
        '<span class="api-track"></span></div>';
    }
    var have = e.dtypes.length;
    // Amber marks a composed implementation, and marks only that.
    var cls = e.impl === "composed" ? "api-meter composed" : "api-meter";
    var pct = (have / DTYPES) * 100;
    // The tick earns its place only where it says something the fill does not:
    // that another backend reaches further on this same operation.
    var tick = best > have
      ? '<u class="api-ref" style="left:' + ((best / DTYPES) * 100).toFixed(2) + '%"></u>'
      : "";
    return '<div class="' + cls + '"><span class="lab">' + b +
      '<b>' + have + '<em>/' + DTYPES + '</em></b></span>' +
      '<span class="api-track"><i class="api-fill" style="width:' + pct.toFixed(2) + '%"></i>' +
      tick + '</span></div>';
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
    if (c.api) bits.push(c.api);
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
    return '<div class="api-detail-in">' + head + B.map(function (b) {
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
      return '<button type="button" class="api-row" aria-expanded="false" data-i="' + i + '">' +
        '<span class="api-id"><span class="api-name">' + op.name +
          (op.catalog ? '<span class="api-fam">' + op.catalog.family + '</span>' : "") +
        '</span>' + describe(op) + '</span>' +
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
})();
