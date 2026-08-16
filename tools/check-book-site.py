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


def chapters() -> list[str]:
    summary = (BOOK / "src/SUMMARY.md").read_text(encoding="utf-8")
    return re.findall(r"\]\(\./([\w-]+)\.md\)", summary)


def main() -> None:
    index = SITE / "index.html"
    javascript = SITE / "book.js"
    if not index.exists() or not javascript.exists():
        sys.exit("generated book site is missing index.html or book.js")

    index_text = index.read_text(encoding="utf-8")
    js_text = javascript.read_text(encoding="utf-8")
    required_shell = (
        '<main id="chapter"',
        'id="sidebar-toggle"',
        'aria-label="Toggle chapter navigation"',
        'id="theme-toggle"',
        'aria-label="Switch theme"',
    )
    missing_shell = [marker for marker in required_shell if marker not in index_text]
    if missing_shell:
        sys.exit(f"book site shell is missing: {', '.join(missing_shell)}")

    missing_chapters = [
        name for name in chapters() if not (SITE / "chapters" / f"{name}.html").exists()
    ]
    if missing_chapters:
        sys.exit(f"book site is missing chapters: {', '.join(missing_chapters)}")

    required_client_hooks = (
        "window.location.hash",
        "history.pushState",
        'fetch(basePath() + "chapters/"',
        'localStorage.getItem("incin-book-theme")',
        "aria-current",
    )
    missing_hooks = [hook for hook in required_client_hooks if hook not in js_text]
    if missing_hooks:
        sys.exit(f"book client is missing: {', '.join(missing_hooks)}")

    print(f"book site checks passed: {len(chapters())} chapters and static shell")


if __name__ == "__main__":
    main()
