#!/usr/bin/env python3
"""Build the chaptered documentation site from mdBook's rendered chapters.

mdBook remains the Markdown renderer and SUMMARY.md remains authoritative for
ordering and grouping. This script turns the rendered, individual chapter
pages into a small static application suitable for GitHub Pages.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "book"
SOURCE = ROOT / "src"
SITE = ROOT / "site"
REPOSITORY_SOURCE = "https://github.com/xupremix/incin/blob/develop/"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def summary() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    sections: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    chapters: list[dict[str, str]] = []
    text = (SOURCE / "SUMMARY.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        heading = re.match(r"^#\s+(.+)$", line)
        if heading:
            current = {"title": heading.group(1), "chapters": []}
            sections.append(current)
            continue
        link = re.match(r"^\s*-?\s*\[(.+)\]\(\./([\w-]+)\.md\)\s*$", line)
        if link:
            title, slug = link.groups()
            item = {"title": title, "slug": slug}
            if current is None:
                current = {"title": "Contents", "chapters": []}
                sections.append(current)
            current["chapters"].append(item)  # type: ignore[index]
            chapters.append(item)
    if not chapters:
        raise SystemExit("SUMMARY.md contains no chapter links")
    return sections, chapters


def chapter_body(path: Path, slug: str) -> str:
    rendered = path.read_text(encoding="utf-8")
    match = re.search(r"<main>(.*?)</main>", rendered, re.DOTALL)
    if not match:
        raise SystemExit(f"{path} does not contain an mdBook <main> element")
    body = match.group(1)
    body = re.sub(r"\s*<nav class=\"nav-wrapper\".*?</nav>\s*", "", body, flags=re.DOTALL)

    # Some mdBook playground renderings contain an accidental second pre
    # wrapper. Keep the playground itself, including its hidden doctest spans.
    while re.search(r"<pre>\s*<pre\b", body):
        body = re.sub(r"<pre>\s*(<pre\b[^>]*>.*?</pre>)\s*</pre>", r"\1", body, count=1, flags=re.DOTALL)

    def rewrite(match: re.Match[str]) -> str:
        target, fragment = match.group(1), match.group(2) or ""
        if target in {f["slug"] + ".html" for f in chapters_global}:
            return f'href="#/{target[:-5]}{fragment}"'
        return match.group(0)

    body = re.sub(r'href="(?:\./)?([\w-]+\.html)(#[^"]*)?"', rewrite, body)
    body = re.sub(
        r'href="(#[^/"][^"]*)"',
        lambda match: f'href="#/{slug}{match.group(1)}"',
        body,
    )
    body = re.sub(
        r'href="(?:\.\./)+((?:crates|docs|tools|examples)/[^"#]+)(#[^"]*)?"',
        lambda match: f'href="{REPOSITORY_SOURCE}{match.group(1)}{match.group(2) or ""}"',
        body,
    )
    body = body.replace('href="print.html"', 'href="/"')
    return body.strip()


def render_sidebar(sections: list[dict[str, object]]) -> str:
    groups: list[str] = []
    for index, section in enumerate(sections):
        items = section["chapters"]
        links = "".join(
            f'<li><a href="#/{item["slug"]}" data-slug="{item["slug"]}">'
            f'{html.escape(item["title"])}</a></li>'
            for item in items  # type: ignore[union-attr]
        )
        groups.append(
            f'<details class="chapter-group" data-group="{index}" open>'
            f'<summary>{html.escape(str(section["title"]))}</summary><ul>{links}</ul></details>'
        )
    return "".join(groups)


def main() -> None:
    global chapters_global
    sections, chapters_global = summary()
    SITE.mkdir(parents=True, exist_ok=True)
    chapter_dir = SITE / "chapters"
    if chapter_dir.exists():
        shutil.rmtree(chapter_dir)
    chapter_dir.mkdir()

    search: list[dict[str, str]] = []
    for chapter in chapters_global:
        slug = chapter["slug"]
        body = chapter_body(SRC / f"{slug}.html", slug)
        (chapter_dir / f"{slug}.html").write_text(body + "\n", encoding="utf-8")
        parser = TextExtractor()
        parser.feed(body)
        search.append({"slug": slug, "title": chapter["title"], "text": " ".join(parser.parts)})

    manifest = {"title": "The Incin Book", "sections": sections, "chapters": chapters_global}
    (SITE / "chapters.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (SITE / "search-index.json").write_text(json.dumps(search, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy(ROOT / "web" / "book.css", SITE / "book.css")
    shutil.copy(ROOT / "web" / "book.js", SITE / "book.js")
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    index = index.replace("__SIDEBAR__", render_sidebar(sections))
    (SITE / "index.html").write_text(index, encoding="utf-8")
    print(f"wrote {len(chapters_global)} chapters to {SITE}")


if __name__ == "__main__":
    chapters_global: list[dict[str, str]] = []
    main()
