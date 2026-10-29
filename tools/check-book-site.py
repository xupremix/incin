#!/usr/bin/env python3
"""Validate the generated chapter site without requiring a browser.

The Pages site is deliberately a small static application. These checks cover
the contract that is easy to regress during template or routing edits: every
SUMMARY chapter has an HTML payload, the shell exposes one chapter mount point,
and the client still contains the hash-routing, base-path, and accessibility
hooks used by the deployed site.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs/book"
SITE = BOOK / "site"
NEXT_SOURCE = ROOT / "docs/book-next/src"
NEXT_SITE = SITE / "next"
PAGES_WORKFLOW = ROOT / ".github/workflows/pages.yml"
REPOSITORY_SOURCE = "https://github.com/xupremix/incin/blob/master/"


def chapters() -> list[str]:
    summary = (BOOK / "src/SUMMARY.md").read_text(encoding="utf-8")
    return re.findall(r"\]\(\./([\w-]+)\.md\)", summary)


def next_chapters() -> list[str]:
    """The Next site carries every stable chapter plus its own in-development
    chapters, so the version switcher can keep the reader on the same page."""
    own = re.findall(r"\]\(\./([\w-]+)\.md\)", (NEXT_SOURCE / "SUMMARY.md").read_text(encoding="utf-8"))
    return chapters() + [slug for slug in own if slug not in chapters()]


def check_site(site: Path, expected: list[str], label: str) -> None:
    index = site / "index.html"
    javascript = site / "book.js"
    if not index.exists() or not javascript.exists():
        sys.exit(f"generated {label} book site is missing index.html or book.js")

    index_text = index.read_text(encoding="utf-8")
    js_text = javascript.read_text(encoding="utf-8")
    required_shell = (
        '<main id="chapter"',
        'id="sidebar-toggle"',
        'aria-label="Toggle chapter navigation"',
        'id="theme-toggle"',
        'aria-label="Switch theme"',
        'data-version-base',
    )
    missing_shell = [marker for marker in required_shell if marker not in index_text]
    if missing_shell:
        sys.exit(f"{label} book site shell is missing: {', '.join(missing_shell)}")

    missing_chapters = [
        name for name in expected if not (site / "chapters" / f"{name}.html").exists()
    ]
    if missing_chapters:
        sys.exit(f"{label} book site is missing chapters: {', '.join(missing_chapters)}")

    chapter_links = []
    for chapter in expected:
        text = (site / "chapters" / f"{chapter}.html").read_text(encoding="utf-8")
        if "<pre><pre" in text:
            sys.exit(f"{label} chapter retains nested pre wrappers: {chapter}")
        chapter_links.extend(re.findall(r'href="([^"]+)"', text))
    bad_links = [link for link in chapter_links if link.startswith(("../", "./")) or re.match(r"^[\w-]+\.html", link)]
    # Cross-book reader links (`../#/<slug>` from Next into stable) are the
    # one sanctioned `../` form; everything else relative is a rewrite miss.
    bad_links = [link for link in bad_links if not re.match(r"^\.\./#/", link)]
    if bad_links:
        sys.exit(f"{label} chapter-relative links remain: {', '.join(bad_links[:5])}")
    source_links = [link for link in chapter_links if "crates/" in link or "docs/" in link or "tools/" in link]
    if source_links and any(not link.startswith(REPOSITORY_SOURCE) for link in source_links):
        sys.exit(f"{label} repository source links are not mapped to GitHub")

    required_client_hooks = (
        "window.location.hash",
        "history.pushState",
        'fetch(basePath() + "chapters/"',
        'localStorage.getItem("incin-book-theme")',
        "aria-current",
        "aria-selected",
        "replaceState",
        "data-version-base",
    )
    missing_hooks = [hook for hook in required_client_hooks if hook not in js_text]
    if missing_hooks:
        sys.exit(f"{label} book client is missing: {', '.join(missing_hooks)}")

    print(f"{label} site checks passed: {len(expected)} chapters and static shell")


def main() -> None:
    workflow = PAGES_WORKFLOW.read_text(encoding="utf-8")
    if 'branches: ["master"]' not in workflow:
        sys.exit("Pages workflow does not build from master")
    if "github.ref == 'refs/heads/master'" not in workflow:
        sys.exit("Pages deployment is not restricted to master")

    index = SITE / "index.html"
    javascript = SITE / "book.js"
    if not index.exists() or not javascript.exists():
        sys.exit("generated book site is missing index.html or book.js")

    check_site(SITE, chapters(), "stable")
    check_site(NEXT_SITE, next_chapters(), "next")


if __name__ == "__main__":
    main()
