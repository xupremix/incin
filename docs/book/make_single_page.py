#!/usr/bin/env python3
"""Builds a single, self-contained static HTML file from an `mdbook build`
output — no external CSS/JS/font requests, so it can be moved, emailed, or
hosted anywhere by itself.

Starts from mdBook's own built-in `print.html` (every chapter already
concatenated and correctly rendered in reading order) and:
  - inlines the CSS actually needed for the navy theme and code highlighting
  - inlines highlight.js plus a two-line replacement for book.js's syntax-
    highlighting call, since that's the only piece of book.js's behavior a
    single unpaginated page still needs
  - removes the sidebar, search bar, theme picker, and next/previous-chapter
    navigation - all of them are multi-page UI with nothing to point at here
  - drops the FontAwesome icon font and the custom web font declaration
    (their icons/glyphs live in the chrome this strips anyway); text falls
    back to the system font stack already declared in general.css

Usage: python3 docs/book/make_single_page.py
Reads docs/book/book/print.html (built by `mdbook build docs/book` first),
writes docs/book/book/incin-book.html.
"""
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
BUILD = ROOT / "book"
SRC = BUILD / "print.html"
DST = BUILD / "incin-book.html"

CSS_FILES = ["css/variables.css", "css/general.css", "css/chrome.css", "tomorrow-night.css"]
STRIP_IDS = [
    "sidebar", "sidebar-resize-handle", "menu-bar-hover-placeholder", "menu-bar",
    "search-wrapper", "searchbar-outer", "searchresults-outer", "searchresults-header",
    "mdbook-help-container",
]
STRIP_CLASSES = ["nav-chapters", "mobile-nav-chapters", "page-wrapper"]


def main() -> None:
    if not SRC.exists():
        sys.exit(f"{SRC} does not exist - run `mdbook build docs/book` first")

    soup = BeautifulSoup(SRC.read_text(encoding="utf-8"), "html.parser")

    # Map each chapter's filename to the id mdBook gave that chapter's own
    # <h1> (slugified from its title, not its filename - "target_api.html"
    # becomes "the-target-api-and-canonical-dispatch", not "target_api").
    # Read from SUMMARY.md directly rather than the rendered sidebar: content
    # links and sidebar links don't share one relative-path convention (a
    # table cell in pytorch_cheatsheet.md links "./autograd.html", the
    # rendered sidebar's own entries may not carry the "./" at all), so
    # matching against the sidebar DOM silently missed some links in an
    # earlier version of this script. SUMMARY.md's own link targets are the
    # single source both were generated from.
    summary_links = re.findall(r"\]\(\./([\w-]+)\.md\)", (ROOT / "src" / "SUMMARY.md").read_text())
    h1_ids_in_order = [h1.get("id") for h1 in soup.find_all("h1") if h1.get("id")]
    if len(summary_links) != len(h1_ids_in_order):
        sys.exit(
            f"SUMMARY.md lists {len(summary_links)} chapters but print.html has "
            f"{len(h1_ids_in_order)} top-level headings - the positional mapping "
            "below would silently pair the wrong ones. Fix the mismatch first."
        )
    filename_to_anchor = {f"{name}.html": anchor for name, anchor in zip(summary_links, h1_ids_in_order)}

    # Every in-book cross-reference link ("see [Backends](./backends.md)")
    # currently points at a separate page that won't exist once this is the
    # only file - rewrite it to the same-page anchor for that chapter (or,
    # if the original link already carried its own #fragment to a specific
    # heading, keep that fragment and just drop the now-meaningless filename).
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#") or href.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, fragment = href.partition("#")
        path = path.lstrip("./")
        if path in filename_to_anchor:
            a["href"] = f"#{fragment}" if fragment else f"#{filename_to_anchor[path]}"

    # Drop every external <link rel="stylesheet"> and <script src="...">;
    # we replace them with inlined equivalents below.
    for tag in soup.find_all("link", rel="stylesheet"):
        tag.decompose()
    for tag in soup.find_all("link", rel=lambda v: v in ("icon", "shortcut icon")):
        tag.decompose()
    for tag in soup.find_all("script", src=True):
        tag.decompose()
    # print.html's whole purpose is "open straight into the print dialog",
    # which is exactly the one behavior a page meant to be read, not printed,
    # must not inherit from it.
    for tag in soup.find_all("script"):
        if tag.string and "window.print" in tag.string:
            tag.decompose()

    # Strip the multi-page/interactive chrome. `page-wrapper` is unwrapped
    # rather than removed - its *content* (the actual chapters) is what
    # we're here for.
    for id_ in STRIP_IDS:
        for tag in soup.find_all(id=id_):
            tag.decompose()
    for class_ in STRIP_CLASSES:
        for tag in soup.find_all(class_=class_):
            if class_ == "page-wrapper":
                tag.unwrap()
            else:
                tag.decompose()
    for tag in soup.find_all("a", class_="header"):
        # Per-heading "#" permalinks only make sense with a URL to copy.
        tag.unwrap()

    # Inline CSS.
    style = soup.new_tag("style")
    style.string = "\n".join(
        (BUILD / f).read_text(encoding="utf-8") for f in CSS_FILES if (BUILD / f).exists()
    )
    soup.head.append(style)

    # Inline highlight.js and replace book.js's role with the two calls a
    # static page still needs: apply the syntax highlighter, and enable the
    # copy-to-clipboard-free "click a code block, select all" affordance
    # mdBook's own CSS already styles for.
    hljs_src = (BUILD / "highlight.js").read_text(encoding="utf-8")
    # This is highlight.js 10.x (mdBook's bundled version, not the newer API
    # the "highlightAll" name might suggest) - `highlightBlock(el)` per node
    # is the call book.js itself uses; there is no `highlightAll` here.
    script = soup.new_tag("script")
    script.string = (
        hljs_src
        + "\nhljs.configure({tabReplace: '    ', languages: []});"
        + "\ndocument.querySelectorAll('pre code').forEach(hljs.highlightBlock);\n"
    )
    soup.body.append(script)

    # The <html class="navy sidebar-visible" ...> class drove book.js's
    # dynamic sidebar; without a sidebar there's nothing to toggle.
    if soup.html.get("class"):
        soup.html["class"] = [c for c in soup.html["class"] if c != "sidebar-visible"]

    html = str(soup)
    # BeautifulSoup's html.parser occasionally leaves behind now-empty
    # attribute artifacts from decomposed nodes' siblings; harmless, but
    # collapse repeated blank lines left by the strips above for a readable
    # source file.
    html = re.sub(r"\n{3,}", "\n\n", html)

    DST.write_text(html, encoding="utf-8")
    size_kb = DST.stat().st_size / 1024
    print(f"wrote {DST} ({size_kb:.0f} KiB)")


if __name__ == "__main__":
    main()
