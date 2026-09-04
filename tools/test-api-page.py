#!/usr/bin/env python3
"""Exercise the capability reference page in a real headless browser.

The book test covers the book application. This test covers the one contract
the capability page adds: its meters are a *scale*, and a scale is only
readable if the same length means the same thing everywhere on the page.

That contract was broken twice without anything noticing, because nothing here
ran. The page first encoded two variables on one mark (hue for implementation,
fill for coverage), and then, after that was fixed, still scaled every row
against the best backend on that row -- a denominator that was 1 on 26 rows, 4
on 81 and 8 on 54, so a full bar meant four different amounts of support and no
two rows could be compared. Both were reported by a reader looking at the
rendered page, which is the wrong place to find them.
"""

from __future__ import annotations

import html
import http.server
import importlib.util
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
SITE: Path | None = None

HARNESS = """<!doctype html>
<meta charset="utf-8">
<title>capability page harness</title>
<body>
<pre id="result">API_TEST=PENDING</pre>
<iframe id="frame" style="width:390px;height:900px;border:0"></iframe>
<script>
const frame = document.getElementById("frame");
const result = document.getElementById("result");
async function until(predicate, label) {
  for (let i = 0; i < 200; i += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  throw new Error("timed out waiting for " + label);
}
function check(condition, message) {
  if (!condition) throw new Error(message);
}
function haveOf(meter) {
  return Number(meter.querySelector(".lab b").firstChild.textContent.trim());
}
async function run() {
  frame.src = "/project/api.html";
  await new Promise((resolve) => frame.addEventListener("load", resolve, {once: true}));
  const doc = () => frame.contentDocument;
  await until(() => doc().querySelectorAll(".api-row").length > 0, "operation rows");

  /* One whole per section. The defect this was written for was a denominator
     that changed from row to row inside one table, so bars in the same table
     could not be compared; a different section measuring a different quantity
     -- element types out of nine, operations out of 179 -- is fine, as long as
     it is constant and stated wherever it is drawn. */
  for (const section of doc().querySelectorAll(".api-sec")) {
    const wholes = new Set([...section.querySelectorAll(".api-meter .lab b em")]
      .map((e) => e.textContent.trim()));
    check(wholes.size <= 1,
      "the " + section.id + " section draws meters against " + wholes.size +
      " different wholes: " + [...wholes].join(" "));
  }
  const denoms = new Set([...doc().querySelectorAll("#sec-operations .api-meter .lab b em")]
    .map((e) => e.textContent.trim()));
  check(denoms.size === 1,
    "the operation table is drawn against " + denoms.size + " wholes: " + [...denoms].join(" "));
  const denom = Number([...denoms][0].replace("/", ""));
  check(denom > 0, "the meter denominator did not parse as a number");

  /* Each bar draws its own stated number against that whole. */
  const meters = [...doc().querySelectorAll("#sec-operations .api-meter:not(.none)")];
  check(meters.length > 0, "no meters rendered");
  for (const meter of meters) {
    const have = haveOf(meter);
    const got = parseFloat(meter.querySelector(".api-fill").style.width);
    const want = (have / denom) * 100;
    check(Math.abs(got - want) < 0.05,
      "a bar reading " + have + "/" + denom + " is drawn at " + got + "%, not " + want + "%");
    check(got <= 100.0001, "a bar overflows its track at " + got + "%");
  }

  /* The per-row best is a tick, and only where it says something the fill
     does not: that some other backend reaches further on this operation. */
  for (const strip of doc().querySelectorAll("#sec-operations .api-strip")) {
    const row = [...strip.querySelectorAll(".api-meter:not(.none)")];
    if (!row.length) continue;
    const best = Math.max(...row.map(haveOf));
    for (const meter of row) {
      const tick = meter.querySelector(".api-ref");
      if (haveOf(meter) < best) {
        check(tick, "no reference tick where another backend reaches " + best + "/" + denom);
        const at = parseFloat(tick.style.left);
        check(Math.abs(at - (best / denom) * 100) < 0.05,
          "reference tick at " + at + "% should stand at " + (best / denom) * 100 + "%");
      } else {
        check(!tick, "a reference tick is drawn on the row's own best, where it says nothing");
      }
    }
  }

  /* The summary cards are the same kind of mark and must also be absolute. */
  const cards = [...doc().querySelectorAll(".api-metric")];
  check(cards.length === 4, "expected four summary cards, found " + cards.length);
  for (const card of cards) {
    const stated = card.querySelector(".b").textContent.replace(/\\s/g, "");
    const parts = stated.split("/").map(Number);
    const got = parseFloat(card.querySelector(".api-bar i").style.width);
    const want = (parts[0] / parts[1]) * 100;
    check(Math.abs(got - want) < 1.01,
      "a card reading " + stated + " is drawn at " + got + "%, not " + want + "%");
  }

  const w = doc().defaultView;

  /* Colour encodes the same quantity as length, in whole steps. A mistyped
     custom property renders as no background at all, which is invisible on a
     dark ground and silently removes the encoding. */
  const ramp = [...doc().querySelectorAll(".api-ramp i")];
  check(ramp.length === 9, "the key ramp shows " + ramp.length + " steps, not 9");
  const swatches = ramp.map((el) => w.getComputedStyle(el).backgroundColor);
  swatches.forEach((colour, i) => {
    check(colour && colour !== "rgba(0, 0, 0, 0)" && colour !== "transparent",
      "ramp step " + (i + 1) + " has no colour: a coverage token is missing");
  });
  check(new Set(swatches).size === 9,
    "the ramp repeats a colour, so two coverage steps are indistinguishable");

  /* Every theme, not just the default. The book has five, two of them light,
     and an earlier version of this ramp defined its light palette behind a
     `data-theme="dark"` that no theme here ever sets -- so the two light
     themes would have rendered the dark ramp and nothing would have said so. */
  const themed = doc().documentElement;
  const themeBefore = themed.className;
  for (const theme of ["navy", "coal", "ayu", "light", "rust"]) {
    themed.className = theme;
    const shades = ramp.map((el) => w.getComputedStyle(el).backgroundColor);
    shades.forEach((colour, i) => {
      check(colour && colour !== "rgba(0, 0, 0, 0)" && colour !== "transparent",
        "ramp step " + (i + 1) + " has no colour in the " + theme + " theme");
    });
    check(new Set(shades).size === 9,
      "the ramp repeats a colour in the " + theme + " theme");
    if (theme === "light" || theme === "rust") {
      check(shades.join("|") !== swatches.join("|"),
        "the " + theme + " theme renders the dark ramp unchanged");
    }
  }
  themed.className = themeBefore;

  for (const meter of meters) {
    const have = haveOf(meter);
    const fill = meter.querySelector(".api-fill");
    const stepClass = [...fill.classList].find((c) => c.indexOf("cov-") === 0);
    check(stepClass, "a bar carries no coverage step class");
    check(stepClass === "cov-" + have,
      "a bar reading " + have + "/" + denom + " is painted " + stepClass);
    const painted = w.getComputedStyle(fill).backgroundColor;
    check(painted === swatches[have - 1],
      "a bar reading " + have + " is not painted its own step's colour");
  }

  /* Implementation is marked once, beside the operation, and only where it
     applies. It used to tint the bar, which put two variables on one mark. */
  const rowEls = [...doc().querySelectorAll(".api-row")];
  check(rowEls.length > 0, "no operation rows rendered");
  let dotted = 0;
  for (const row of rowEls) {
    const dot = row.querySelector(".api-name .api-dot");
    const labels = [...row.querySelectorAll(".api-meter:not(.none) .lab")];
    if (dot) dotted += 1;
    check(!row.querySelector(".api-fill.composed"),
      "implementation is being drawn on the bar again");
    void labels;
  }
  check(dotted > 0 && dotted < rowEls.length,
    "the composed dot marks " + dotted + " of " + rowEls.length +
    " rows, which cannot be right");

  /* Every operation must reach its documentation. The link lives in the
     expanded detail, so this opens rows rather than trusting the payload. */
  const opened = rowEls.slice(0, 12);
  for (const row of opened) {
    row.click();
  }
  await until(() => doc().querySelectorAll(".api-detail:not([hidden]) .api-docs").length
    === opened.length, "documentation links in the opened rows");
  const docLinks = [...doc().querySelectorAll(".api-detail:not([hidden]) .api-docs")];
  for (const link of docLinks) {
    const href = link.getAttribute("href") || "";
    check(href.indexOf("https://docs.rs/") === 0,
      "a documentation link does not point at docs.rs: " + href);
    check(link.getAttribute("rel") === "noopener noreferrer",
      "a documentation link opens a new tab without rel=noopener");
    const method = href.indexOf("#method.") >= 0;
    const search = href.indexOf("?search=") >= 0;
    check(method || search,
      "a documentation link names neither an item anchor nor a search: " + href);
    if (method) {
      const item = href.slice(href.indexOf("#method.") + 8);
      check(link.textContent.indexOf(item) >= 0,
        "a documentation link is labelled for a different item than it targets");
    }
  }
  for (const row of opened) {
    row.click();
  }

  /* The key explains the marks, so it has to come before them. */
  const key = doc().querySelector(".api-key");
  const list = doc().querySelector(".api-list");
  check(key && list, "the key or the operation list is missing");
  check(key.compareDocumentPosition(list) & Node.DOCUMENT_POSITION_FOLLOWING,
    "the key is rendered after the table it explains");

  /* Every section: one visible at a time, the right content in each, and no
     sideways scroll in any of them. The overflow has been reported twice by a
     reader, so it is checked per section rather than only on the default. */
  const tabs = [...doc().querySelectorAll(".api-tab")];
  check(tabs.length === 8, "expected eight section tabs, found " + tabs.length);

  /* Counts that come from a source the crate owns, so a section quietly
     rendering nothing -- an extractor that stopped matching, a payload key
     that moved -- fails here rather than shipping an empty tab. */
  const counts = {
    dtypes: [".api-card", 9],
    backends: [".api-card", 4],
    flow: [".api-step", 5],
    layouts: [".api-card", 9],
    target: [".api-tyrow", 21],
  };
  for (const tab of tabs) {
    const id = tab.dataset.sec;
    tab.click();
    const section = doc().getElementById("sec-" + id);
    check(section && !section.hidden, "section " + id + " did not open");
    const others = [...doc().querySelectorAll(".api-sec")].filter((el) => el !== section);
    check(others.every((el) => el.hidden), "opening " + id + " left another section visible");
    check(tab.getAttribute("aria-selected") === "true",
      "the " + id + " tab is not marked selected while its section is open");

    if (counts[id]) {
      const [selector, want] = counts[id];
      const got = section.querySelectorAll(selector).length;
      check(got === want, id + " renders " + got + " entries, not " + want);
    }

    const wide = doc().documentElement;
    check(wide.scrollWidth <= wide.clientWidth + 1,
      "the " + id + " section scrolls sideways: scrollWidth " + wide.scrollWidth +
      " exceeds " + wide.clientWidth);
  }

  /* The type reference is fetched on demand, so it has to actually arrive. */
  doc().querySelector('.api-tab[data-sec="types"]').click();
  await until(() => doc().querySelectorAll("#tyRows .api-tyrow").length > 0,
    "the type reference to load");
  const tyRows = [...doc().querySelectorAll("#tyRows .api-tyrow")];
  check(tyRows.length > 0 && tyRows.length <= 300,
    "the type reference rendered " + tyRows.length + " rows, outside its cap");
  const linked = [...doc().querySelectorAll("#tyRows a.api-tyname")];
  check(linked.length > 0, "no type in the reference links to its documentation");
  for (const link of linked) {
    check((link.getAttribute("href") || "").indexOf("https://docs.rs/") === 0,
      "a type link does not point at docs.rs: " + link.getAttribute("href"));
    check(link.getAttribute("rel") === "noopener noreferrer",
      "a type link opens a new tab without rel=noopener");
  }
  const kindChip = doc().querySelector('#tyKindChips .api-chip[data-k="trait"]');
  check(kindChip, "the type reference has no kind filter");
  kindChip.click();
  await until(() => [...doc().querySelectorAll("#tyRows .api-tykind")]
    .every((el) => el.textContent.trim() === "trait"), "the kind filter to apply");
  kindChip.click();

  /* Every proportional bar on the page, not only the operation meters, is
     painted on the nine-step ramp. The summary and backend bars carried a
     fixed hue while their markup already asked for a step, so they read on a
     different scale from everything else. */
  for (const tab of tabs) {
    tab.click();
    const bars = [...doc().querySelectorAll(".api-sec:not([hidden]) .api-bar i")];
    for (const bar of bars) {
      const stepClass = [...bar.classList].find((c) => c.indexOf("cov-") === 0);
      check(stepClass, "a bar in " + tab.dataset.sec + " carries no coverage step class");
      const pct = parseFloat(bar.style.width);
      const want = Math.min(9, Math.max(1, Math.round((pct / 100) * 9)));
      check(stepClass === "cov-" + want,
        "a bar at " + pct + "% is painted " + stepClass + ", not cov-" + want);
      const painted = w.getComputedStyle(bar).backgroundColor;
      check(painted === swatches[want - 1],
        "a bar at " + pct + "% is not painted its step's colour");
    }
  }
  doc().querySelector('.api-tab[data-sec="operations"]').click();

  /* Examples are the reason to trust the page, so they must actually be
     there, must not be silently truncated, and must say whether the compiler
     checks them -- 20 of the 82 are `ignore` blocks that it does not. */
  for (const tab of tabs) {
    tab.click();
    const id = tab.dataset.sec;
    const figures = [...doc().querySelectorAll(".api-sec:not([hidden]) .api-ex")];
    check(figures.length > 0, "the " + id + " section shows no worked example");
    for (const figure of figures) {
      const code = figure.querySelector("pre code");
      check(code && code.textContent.trim().length > 0,
        "an example in " + id + " renders no code");
      const link = figure.querySelector("figcaption a");
      check(link && (link.getAttribute("href") || "").indexOf("./#/") === 0,
        "an example in " + id + " does not link to its chapter");
      const tag = figure.querySelector(".api-extag");
      check(tag && /^(compiled|not compiled)$/.test(tag.textContent.trim()),
        "an example in " + id + " does not state whether it is compiled");
    }
  }
  doc().querySelector('.api-tab[data-sec="operations"]').click();

  const shapeItems = doc().querySelectorAll("#shapeGroups .api-tyrow").length;
  check(shapeItems > 50,
    "the shape reference rendered " + shapeItems + " entries, which cannot be right");

  doc().querySelector('.api-tab[data-sec="operations"]').click();

  /* The other repeatedly reported defect: sideways scroll on a phone. */
  const root = doc().documentElement;
  const limit = root.clientWidth;
  const wide = [...doc().querySelectorAll("*")]
    .filter((el) => el.getBoundingClientRect().right > limit + 1)
    .map((el) => el.tagName.toLowerCase() +
      (el.className ? "." + String(el.className).trim().split(/\\s+/).join(".") : "") +
      " right=" + Math.round(el.getBoundingClientRect().right));
  check(root.scrollWidth <= limit + 1,
    "the page scrolls sideways at 390px: scrollWidth " + root.scrollWidth +
    " exceeds clientWidth " + limit + " -- offenders: " + wide.slice(0, 8).join(" | "));

  result.textContent = "API_TEST=" + "PASS";
}
run().catch((error) => {
  result.textContent = "API_TEST=FAIL\\n" + error.stack;
});
</script>
"""


def harness_verdict(dumped: str, sentinel: str) -> tuple[bool, str]:
    """Read the verdict from the result element, not from the whole dump.

    `--dump-dom` emits the harness's own <script> source along with the DOM,
    and that source contains the success sentinel as a literal. Substring
    matching the dump therefore reports success unconditionally: this suite
    passed with `check(false)` as the first statement of `run()`, so none of
    its assertions had ever been evaluated. The sentinel is also assembled
    from two pieces below so that the literal cannot reappear in the source.
    """
    match = re.search(r'<pre id="result">(.*?)</pre>', dumped, re.DOTALL)
    if match is None:
        return False, "the harness result element was missing from the dumped DOM"
    verdict = html.unescape(match.group(1)).strip()
    return verdict == sentinel, verdict


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler protocol
        path = urlsplit(self.path).path
        if path == "/api-test.html":
            payload = HARNESS.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path.startswith("/project/"):
            self.path = path.removeprefix("/project")
        else:
            self.send_error(404)
            return
        super().do_GET()

    def log_message(self, *_args) -> None:
        return


def main() -> int:
    global SITE
    browser = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    if browser is None:
        raise SystemExit("a Chromium-compatible browser is required for the capability page test")
    builder_spec = importlib.util.spec_from_file_location("incin_book_builder", ROOT / "docs/book/build_site.py")
    if builder_spec is None or builder_spec.loader is None:
        raise SystemExit("unable to load book site builder")
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)

    with tempfile.TemporaryDirectory(prefix="incin-api-site-") as output:
        builder.SITE = Path(output)
        builder.main()
        SITE = Path(output)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="incin-api-browser-") as profile:
                command = [
                    browser,
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--user-data-dir=" + profile,
                    "--virtual-time-budget=12000",
                    "--dump-dom",
                    f"http://127.0.0.1:{server.server_port}/api-test.html",
                ]
                result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
        finally:
            server.shutdown()
            thread.join(timeout=5)

    output_text = result.stdout + result.stderr
    passed, verdict = harness_verdict(output_text, "API_TEST=" + "PASS")
    if result.returncode != 0 or not passed:
        print(output_text)
        print("harness verdict: " + verdict)
        return 1
    print("capability page checks passed: one scale, bars match their numbers, ticks, no sideways scroll")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
