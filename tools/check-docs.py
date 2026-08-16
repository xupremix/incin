#!/usr/bin/env python3
"""Check documentation inventories that must not drift from source metadata."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/book/src/SUMMARY.md"
FEATURES = ROOT / "crates/incin/Cargo.toml"
FEATURE_DOC = ROOT / "docs/book/src/feature_flags.md"
FACADE_LIB = ROOT / "crates/incin/src/lib.rs"


def summary_chapters() -> list[str]:
    return re.findall(r"\]\(\./([\w-]+)\.md\)", SUMMARY.read_text())


def facade_features() -> list[str]:
    text = FEATURES.read_text()
    table = text.split("[features]", 1)[1].split("[dependencies]", 1)[0]
    return [name for name in re.findall(r"^([A-Za-z0-9_-]+)\s*=", table, re.MULTILINE) if name != "default"]


def main() -> int:
    errors: list[str] = []
    chapters = summary_chapters()
    if len(chapters) != len(set(chapters)):
        errors.append("SUMMARY.md contains duplicate chapter links")
    for slug in chapters:
        if not (SUMMARY.parent / f"{slug}.md").exists():
            errors.append(f"SUMMARY.md links to missing chapter {slug}.md")

    facade = FACADE_LIB.read_text()
    included = set(re.findall(r'include_str!\("\.\./\.\./\.\./docs/book/src/([\w-]+)\.md"\)', facade))
    for slug in chapters:
        if slug not in included:
            errors.append(f"book chapter `{slug}.md` is missing from the Cargo-backed doctest aggregation")
    for slug in sorted(included - set(chapters)):
        errors.append(f"Cargo-backed doctest aggregation includes chapter not listed in SUMMARY.md: {slug}.md")

    doc = FEATURE_DOC.read_text()
    for feature in facade_features():
        count = len(re.findall(rf"^\| `{re.escape(feature)}` \|", doc, re.MULTILINE))
        if count != 1:
            errors.append(f"feature_flags.md documents `{feature}` {count} times; expected once")

    if errors:
        for error in errors:
            print(f"docs check: {error}", file=sys.stderr)
        return 1
    print(f"docs checks passed: {len(chapters)} book chapters and {len(facade_features())} facade features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
