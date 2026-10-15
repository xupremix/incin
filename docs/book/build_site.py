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
NEXT_ROOT = ROOT.parent / "book-next"
NEXT_SRC = NEXT_ROOT / "book"
NEXT_SOURCE = NEXT_ROOT / "src"
REPOSITORY_SOURCE = "https://github.com/xupremix/incin/blob/master/"

STABLE_TITLE = "The Incin Book"
NEXT_TITLE = "The Incin Book \u2014 Next"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def parse_summary(path: Path) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Parse one SUMMARY.md into sidebar sections and an ordered chapter list."""
    sections: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    chapters: list[dict[str, str]] = []
    text = path.read_text(encoding="utf-8")
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


def chapter_body(path: Path, slug: str, chapters: list[dict[str, str]]) -> str:
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

    chapter_pages = {entry["slug"] + ".html" for entry in chapters}

    def rewrite(match: re.Match[str]) -> str:
        target, fragment = match.group(1), match.group(2) or ""
        if target in chapter_pages:
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


def version_switcher(version: str) -> str:
    """Header switcher between the stable book and the Next book.

    The link's plain href is the no-JavaScript fallback (the other book's
    introduction); the client upgrades the click to keep the current chapter
    when the other book carries the same slug.
    """
    if version == "next":
        return (
            '<span class="version-current" aria-current="page">Next</span>'
            ' <a class="version-link" data-version-base="../" href="../">'
            "&larr;&nbsp;Stable</a>"
        )
    return (
        '<span class="version-current" aria-current="page">Stable</span>'
        ' <a class="version-link" data-version-base="next/" href="next/">'
        "Next&nbsp;&rarr;</a>"
    )


def collect_chapters(version: str) -> tuple[list[dict[str, object]], list[dict[str, str]], dict[str, Path]]:
    """Resolve one site's table of contents to mdBook-rendered pages.

    The stable site renders its own SUMMARY. The Next site renders the stable
    chapters unchanged plus its own in-development chapters: each slug
    resolves to the Next book's rendered page when it has its own source
    file, and to the stable book's rendered page otherwise. Slugs must not
    collide across the two sources.
    """
    sections, chapters = parse_summary(SOURCE / "SUMMARY.md")
    rendered: dict[str, Path] = {chapter["slug"]: SRC / f'{chapter["slug"]}.html' for chapter in chapters}
    if version == "next":
        next_sections, next_chapters = parse_summary(NEXT_SOURCE / "SUMMARY.md")
        for chapter in next_chapters:
            slug = chapter["slug"]
            if slug in rendered:
                raise SystemExit(f"Next chapter slug collides with a stable chapter: {slug}")
            own = NEXT_SOURCE / f"{slug}.md"
            if not own.exists():
                raise SystemExit(f"Next SUMMARY links to missing chapter {own}")
            rendered[slug] = NEXT_SRC / f"{slug}.html"
        sections = sections + next_sections
        chapters = chapters + next_chapters
    return sections, chapters, rendered


def build(version: str, out_dir: Path) -> int:
    sections, chapters, rendered = collect_chapters(version)
    out_dir.mkdir(parents=True, exist_ok=True)
    chapter_dir = out_dir / "chapters"
    if chapter_dir.exists():
        shutil.rmtree(chapter_dir)
    chapter_dir.mkdir()

    if version == "stable":
        asset_source = SOURCE / "assets"
        asset_target = out_dir / "assets"
        if asset_target.exists():
            shutil.rmtree(asset_target)
        if asset_source.exists():
            shutil.copytree(asset_source, asset_target)

    search: list[dict[str, str]] = []
    for chapter in chapters:
        slug = chapter["slug"]
        body = chapter_body(rendered[slug], slug, chapters)
        (chapter_dir / f"{slug}.html").write_text(body + "\n", encoding="utf-8")
        parser = TextExtractor()
        parser.feed(body)
        search.append({"slug": slug, "title": chapter["title"], "text": " ".join(parser.parts)})

    title = NEXT_TITLE if version == "next" else STABLE_TITLE
    manifest = {"title": title, "sections": sections, "chapters": chapters}
    (out_dir / "chapters.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "search-index.json").write_text(json.dumps(search, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy(ROOT / "web" / "book.css", out_dir / "book.css")
    shutil.copy(ROOT / "web" / "book.js", out_dir / "book.js")
    shutil.copy(ROOT / "web" / "highlight.js", out_dir / "highlight.js")
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    index = index.replace("__SIDEBAR__", render_sidebar(sections))
    index = index.replace("__VERSION_SWITCHER__", version_switcher(version))
    index = index.replace("__PAGE_TITLE__", title)
    index = index.replace("__MENU_TITLE__", title)
    index = index.replace("__API_HREF__", "../api.html" if version == "next" else "api.html")
    (out_dir / "index.html").write_text(index, encoding="utf-8")

    if version == "stable":
        write_api_reference(out_dir)
    print(f"wrote {len(chapters)} {version} chapters to {out_dir}")
    return len(chapters)


def main(argv: list[str] | None = None) -> None:
    """Build the book site. Bare invocation keeps its historic contract (the
    stable book into `docs/book/site`, which the browser test relies on);
    `--next` builds only the Next book into `docs/book/site/next`, and
    `--all` builds both. `mdbook build docs/book` (and `docs/book-next` for
    the Next book) must run first."""
    import sys as _sys

    args = _sys.argv[1:] if argv is None else argv
    versions = ["stable", "next"] if "--all" in args else (["next"] if "--next" in args else ["stable"])
    for version in versions:
        build(version, SITE / "next" if version == "next" else SITE)


def write_api_reference(out_dir: Path) -> None:
    """Emit the capability reference beside the book.

    The payload is inlined into a JSON script tag rather than fetched, so the
    page has no load-failure path and works from a `file://` checkout. It is a
    separate document rather than a chapter because it is a table to operate,
    not prose to read in order -- SUMMARY.md stays the book's contents, and
    `check-docs.py` keeps every chapter there matched to a doctest include,
    which a generated page has nothing to offer.
    """
    payload = ROOT.parent / "api-site-data.json"
    if not payload.exists():
        raise SystemExit(
            "docs/api-site-data.json is missing; run tools/build-api-site-data.py"
        )
    shutil.copy(ROOT / "web" / "api.css", out_dir / "api.css")
    shutil.copy(ROOT / "web" / "api.js", out_dir / "api.js")
    page = (ROOT / "web" / "api.html").read_text(encoding="utf-8")
    # A closing tag inside the JSON would end the script element early.
    data = payload.read_text(encoding="utf-8").replace("</", "<\\/")
    (out_dir / "api.html").write_text(page.replace("__API_DATA__", data), encoding="utf-8")
    # The type reference is fetched on demand rather than inlined: it is three
    # times the size of everything else on the page, and most readers never
    # open it.
    shutil.copy(ROOT.parent / "api-types.json", out_dir / "api-types.json")
    shutil.copy(ROOT.parent / "api-usage.json", out_dir / "api-usage.json")


if __name__ == "__main__":
    main()
