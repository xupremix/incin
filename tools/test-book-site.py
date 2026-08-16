#!/usr/bin/env python3
"""Exercise the generated book application in a real headless browser.

The static checker verifies that the generated files exist. This test verifies
the application contract that only exists after JavaScript runs: routing below
a repository base path, chapter replacement, browser history, heading routes,
keyboard navigation, search, and theme persistence across navigation.
"""

from __future__ import annotations

import http.server
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
  win().location.hash = "#/quickstart#" + heading.id;
  await until(() => win().location.hash.endsWith("#" + heading.id), "heading route");

  win().history.back();
  await until(() => win().location.hash === "#/quickstart", "browser back");
  const next = doc().getElementById("next");
  check(!next.hidden, "next chapter link is missing");
  doc().getElementById("chapter").focus();
  doc().dispatchEvent(new KeyboardEvent("keydown", {key: "ArrowRight", bubbles: true}));
  await until(() => win().location.hash === next.hash, "keyboard next navigation");

  result.textContent = "BOOK_TEST=PASS";
}
run().catch((error) => {
  result.textContent = "BOOK_TEST=FAIL\n" + error.stack;
});
</script>
"""


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
    browser = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    if browser is None:
        raise SystemExit("a Chromium-compatible browser is required for the book browser test")
    if not (SITE / "index.html").exists():
        raise SystemExit("generated book site is missing; run docs/book/build_site.py first")

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
    if result.returncode != 0 or "BOOK_TEST=PASS" not in output:
        print(output)
        return 1
    print("book browser checks passed: routing, history, theme, search, keyboard, DOM, base path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
