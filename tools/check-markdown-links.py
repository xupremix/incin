#!/usr/bin/env python3
"""Validate repository-local Markdown links without requiring network access."""

from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"target", "graphify-out", ".git", ".vscode-test", "node_modules"}


def markdown_files() -> list[Path]:
    return [path for path in ROOT.rglob("*.md") if not SKIP.intersection(path.parts)]


def main() -> int:
    errors: list[str] = []
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for source in markdown_files():
        for raw in pattern.findall(source.read_text(encoding="utf-8", errors="replace")):
            target = raw.strip().split()[0].strip("<>")
            if not target or target.startswith(("#", "/", "http:", "https:", "mailto:", "data:")):
                continue
            path = unquote(target.split("#", 1)[0])
            resolved = (source.parent / path).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"{source.relative_to(ROOT)}: link escapes repository: {target}")
                continue
            if not resolved.exists():
                errors.append(f"{source.relative_to(ROOT)}: missing link target: {target}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"markdown link check passed: {len(markdown_files())} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
