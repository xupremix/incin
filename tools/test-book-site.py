#!/usr/bin/env python3
"""Exercise the generated book application in a real headless browser.

The static checker verifies that the generated files exist. This test verifies
the application contract that only exists after JavaScript runs: routing below
a repository base path, chapter replacement, browser history, heading routes,
keyboard navigation, search, and theme persistence across navigation.
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


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs/book/site"

HARNESS = r"""<!doctype html>
<meta charset="utf-8">
<title>Book browser contract</title>
<iframe id="book" title="book under test" style="width:1px;height:1px"></iframe>
<pre id="result">BOOK_TEST=RUNNING</pre>
<script>
const frame = document.getElementById("book");
const result = document.getElementById("result");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function until(predicate, label) {
  const deadline = Date.now() + 7000;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await sleep(50);
  }
  throw new Error("timed out waiting for " + label);
}
function check(condition, message) {
  if (!condition) throw new Error(message);
}
/* The application records sidebar state on <body>, and which class carries it
   depends on the viewport: the wide layout marks "hidden", the narrow layout
   marks "open". This previously asserted `#sidebar.open`, a class the
   application has never set, so it could only ever have failed. */
function sidebarOpen(doc, win) {
  return win.innerWidth <= 900
    ? doc.body.classList.contains("sidebar-open")
    : !doc.body.classList.contains("sidebar-hidden");
}
async function run() {
  frame.src = "/project/index.html#/introduction";
  await new Promise((resolve) => frame.addEventListener("load", resolve, {once: true}));
  const win = () => frame.contentWindow;
  const doc = () => frame.contentDocument;
  await until(() => doc().querySelector("#chapter h1"), "initial chapter");
  check(doc().querySelectorAll("#chapter").length === 1, "expected one active chapter mount");
  check(!doc().querySelector("[data-chapter-body]"), "inactive chapter bodies were retained");

  const initialTheme = doc().documentElement.dataset.theme;
  doc().getElementById("theme-toggle").click();
  await until(() => doc().documentElement.dataset.theme !== initialTheme, "theme toggle");
  const selectedTheme = doc().documentElement.dataset.theme;
  const selectedBackground = win().getComputedStyle(doc().documentElement).backgroundColor;

  const quickstart = [...doc().querySelectorAll("a[data-slug]")]
    .find((link) => link.dataset.slug === "quickstart");
  check(quickstart, "quickstart sidebar link is missing");
  quickstart.click();
  await until(() => win().location.hash === "#/quickstart" && doc().querySelector("#chapter h1"), "sidebar route");
  check(doc().title.startsWith("Quickstart"), "chapter title did not update");
  check(doc().documentElement.dataset.theme === selectedTheme, "chapter navigation changed theme");
  check(win().getComputedStyle(doc().documentElement).backgroundColor === selectedBackground,
    "chapter navigation changed the application background");

  const search = doc().getElementById("search");
  search.value = "tensor";
  search.dispatchEvent(new Event("input", {bubbles: true}));
  await until(() => !doc().getElementById("search-results").hidden, "search results");
  check(doc().querySelectorAll("#search-results a").length > 0, "search returned no results");

  const heading = doc().querySelector("#chapter h2[id], #chapter h3[id]");
  check(heading, "quickstart has no addressable heading");
  const headingLink = heading.querySelector("a.header");
  check(headingLink, "heading permalink is missing");
  headingLink.click();
  await until(() => win().location.hash === "#/quickstart#" + heading.id, "heading route");
  check(doc().getElementById(heading.id), "heading was not retained after permalink click");

  win().history.back();
  await until(() => win().location.hash === "#/quickstart", "browser back");
  const next = doc().getElementById("next");
  check(!next.hidden, "next chapter link is missing");
  doc().getElementById("chapter").focus();
  doc().dispatchEvent(new KeyboardEvent("keydown", {key: "ArrowRight", bubbles: true}));
  await until(() => win().location.hash === next.hash, "keyboard next navigation");

  const transformer = [...doc().querySelectorAll("a[data-slug]")]
    .find((link) => link.dataset.slug === "transformer");
  check(transformer, "transformer sidebar link is missing");
  transformer.click();
  /* The hash updates on click; the chapter body arrives a fetch later, and the
     loading indicator clears the old h1 in between -- so an h1 means the new
     chapter has actually landed. Waiting on the hash alone raced, and every
     assertion below ran against the previous chapter. */
  await until(() => win().location.hash === "#/transformer" && doc().querySelector("#chapter h1"),
    "transformer route");
  const sourceLink = doc().querySelector('a[href^="https://github.com/xupremix/incin/blob/"]');
  check(sourceLink, "repository source link was not mapped to GitHub");
  check(!doc().querySelector("#chapter pre > pre"), "nested pre wrapper remains");

  /* The transformer chapter carries the repository source link but has neither
     a runnable block nor a hidden doctest line, so asserting those here was
     asserting features that page never had. They belong on a chapter that has
     both -- 17 chapters render a playground and 12 render hidden lines. */
  const autograd = [...doc().querySelectorAll("a[data-slug]")]
    .find((link) => link.dataset.slug === "autograd");
  check(autograd, "autograd sidebar link is missing");
  autograd.click();
  await until(() => win().location.hash === "#/autograd" && doc().querySelector("#chapter h1"),
    "autograd route");
  check(!doc().querySelector("#chapter pre > pre"), "nested pre wrapper remains");
  check(doc().querySelector("#chapter .playground"), "playground styling hook is missing");
  const boring = doc().querySelector("#chapter .boring");
  check(boring && win().getComputedStyle(boring).display === "none", "hidden doctest line is visible");

  search.value = "tensor";
  search.dispatchEvent(new Event("input", {bubbles: true}));
  search.dispatchEvent(new KeyboardEvent("keydown", {key: "ArrowDown", bubbles: true}));
  check(doc().querySelector('#search-results a[aria-selected="true"]'), "search keyboard selection is missing");

  const toggle = doc().getElementById("sidebar-toggle");
  toggle.click();
  check(toggle.getAttribute("aria-expanded") === "true" && sidebarOpen(doc(), win()),
    "sidebar open state disagrees with aria-expanded");
  /* Sidebar entries are addressed by data-slug; there is no element with this
     id, so this step threw before it could check anything. */
  [...doc().querySelectorAll("a[data-slug]")]
    .find((link) => link.dataset.slug === "quickstart").click();
  await until(() => win().location.hash === "#/quickstart" && doc().querySelector("#chapter h1"),
    "sidebar close after navigation");
  check(toggle.getAttribute("aria-expanded") === "false" && !sidebarOpen(doc(), win()),
    "sidebar close state disagrees with aria-expanded");

  win().location.hash = "#/does-not-exist";
  await until(() => win().location.hash === "#/introduction" && doc().querySelector("#chapter h1"),
    "unknown route normalization");
  check(doc().title.startsWith("Introduction"), "unknown route retained fallback/content disagreement");

  result.textContent = "BOOK_TEST=" + "PASS";
}
run().catch((error) => {
  result.textContent = "BOOK_TEST=FAIL\n" + error.stack;
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
        if path == "/book-test.html":
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
        raise SystemExit("a Chromium-compatible browser is required for the book browser test")
    builder_spec = importlib.util.spec_from_file_location("incin_book_builder", ROOT / "docs/book/build_site.py")
    if builder_spec is None or builder_spec.loader is None:
        raise SystemExit("unable to load book site builder")
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)

    with tempfile.TemporaryDirectory(prefix="incin-book-site-") as output:
        builder.SITE = Path(output)
        builder.main()
        SITE = Path(output)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="incin-book-browser-") as profile:
                command = [
                    browser,
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--user-data-dir=" + profile,
                    "--virtual-time-budget=12000",
                    "--dump-dom",
                    f"http://127.0.0.1:{server.server_port}/book-test.html",
                ]
                result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        finally:
            server.shutdown()
            thread.join(timeout=5)

    output = result.stdout + result.stderr
    passed, verdict = harness_verdict(output, "BOOK_TEST=" + "PASS")
    if result.returncode != 0 or not passed:
        print(output)
        print("harness verdict: " + verdict)
        return 1
    print("book browser checks passed: routing, history, theme, search, keyboard, DOM, base path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
