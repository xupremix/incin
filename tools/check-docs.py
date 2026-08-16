#!/usr/bin/env python3
"""Check documentation inventories that must not drift from source metadata."""

from pathlib import Path
import re
import sys
import tomllib

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


def workspace_features() -> dict[str, list[str]]:
    features: dict[str, list[str]] = {}
    for manifest in sorted((ROOT / "crates").rglob("Cargo.toml")):
        package = tomllib.loads(manifest.read_text(encoding="utf-8")).get("package", {})
        name = package.get("name")
        if not name:
            continue
        for feature in tomllib.loads(manifest.read_text(encoding="utf-8")).get("features", {}):
            if feature != "default":
                features.setdefault(feature, []).append(name)
    return features


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
    for feature, crates in workspace_features().items():
        count = len(re.findall(rf"^\| `{re.escape(feature)}` \|", doc, re.MULTILINE))
        if count != 1:
            owners = ", ".join(crates)
            errors.append(
                f"feature_flags.md documents workspace feature `{feature}` {count} times; "
                f"expected once (defined by {owners})"
            )

    if errors:
        for error in errors:
            print(f"docs check: {error}", file=sys.stderr)
        return 1
    print(
        f"docs checks passed: {len(chapters)} book chapters, "
        f"{len(facade_features())} facade features, and {len(workspace_features())} workspace features"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
