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

  function matches(op) {
    if (state.q && op.name.toLowerCase().indexOf(state.q) < 0) return false;
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

  function chipFor(e) {
    if (!e.dtypes.length) return '<span class="api-sup absent">&mdash;</span>';
    var cls = e.impl === "composed" ? "composed" : "native";
    return '<span class="api-sup ' + cls + '"><i></i>' + e.dtypes.length + '</span>';
  }

  function detailFor(op) {
    return '<div class="api-detail-in">' + B.map(function (b) {
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
      rows.innerHTML = '<tr><td colspan="5" class="api-empty">No operation matches those filters.</td></tr>';
      return;
    }
    rows.innerHTML = list2.map(function (op, i) {
      return '<tr class="op" tabindex="0" role="button" aria-expanded="false" data-i="' + i + '">' +
        '<td class="name">' + op.name + '</td>' +
        B.map(function (b) { return '<td style="text-align:center">' + chipFor(op.backends[b]) + '</td>'; }).join("") +
        '</tr><tr class="api-detail" data-d="' + i + '" hidden><td colspan="5"></td></tr>';
    }).join("");
    rows._list = list2;
  }

  function toggleRow(tr) {
    var i = tr.dataset.i;
    var det = rows.querySelector('tr.api-detail[data-d="' + i + '"]');
    var open = tr.getAttribute("aria-expanded") === "true";
    tr.setAttribute("aria-expanded", open ? "false" : "true");
    if (open) { det.hidden = true; return; }
    if (!det.firstElementChild.innerHTML) {
      det.firstElementChild.innerHTML = detailFor(rows._list[i]);
    }
    det.hidden = false;
  }
  rows.addEventListener("click", function (e) {
    var tr = e.target.closest("tr.op");
    if (tr) toggleRow(tr);
  });
  rows.addEventListener("keydown", function (e) {
    if (e.key !== "Enter" && e.key !== " ") return;
    var tr = e.target.closest("tr.op");
    if (tr) { e.preventDefault(); toggleRow(tr); }
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
