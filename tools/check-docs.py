#!/usr/bin/env python3
"""Check documentation inventories that must not drift from source metadata."""

from pathlib import Path
import binascii
import re
import struct
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/book/src/SUMMARY.md"
FEATURES = ROOT / "crates/incin/Cargo.toml"
FEATURE_DOC = ROOT / "docs/book/src/feature_flags.md"
FEATURE_MATRIX = ROOT / "docs/FEATURE_MATRIX.md"
FACADE_LIB = ROOT / "crates/incin/src/lib.rs"
EDITOR_ASSETS = ROOT / "docs/assets/editors"
BOOK_EDITOR_ASSETS = ROOT / "docs/book/src/assets/editors"
EDITOR_SCREENSHOTS = (
    "vscode-shape-diagnostic.png",
    "neovim-shape-diagnostic.png",
)


def summary_chapters() -> list[str]:
    return re.findall(r"\]\(\./([\w-]+)\.md\)", SUMMARY.read_text())


def facade_features() -> list[str]:
    table = tomllib.loads(FEATURES.read_text(encoding="utf-8")).get("features", {})
    return [name for name in table if name != "default"]


def workspace_features() -> dict[str, list[str]]:
    features: dict[str, list[str]] = {}
    for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        package = tomllib.loads(manifest.read_text(encoding="utf-8")).get("package", {})
        name = package.get("name")
        if not name:
            continue
        for feature in tomllib.loads(manifest.read_text(encoding="utf-8")).get("features", {}):
            if feature != "default":
                features.setdefault(feature, []).append(name)
    return features


def feature_contract_rows(path: Path) -> set[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    begin = "<!-- BEGIN GENERATED: feature-inventory -->"
    end = "<!-- END GENERATED: feature-inventory -->"
    if begin not in text or end not in text:
        raise ValueError(f"{path.relative_to(ROOT)} is missing feature-inventory markers")
    body = text.split(begin, 1)[1].split(end, 1)[0]
    return set(re.findall(r"^\| `([^`]+)` \| `([^`]+)` \|", body, re.MULTILINE))


def check_editor_screenshot(path: Path) -> list[str]:
    """Check structure and metadata; visual privacy is reviewed separately."""
    errors: list[str] = []
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return [f"{path.relative_to(ROOT)} is not a PNG"]

    offset = 8
    saw_header = False
    saw_end = False
    forbidden = {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"caBX"}
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            errors.append(f"{path.relative_to(ROOT)} has a truncated PNG chunk")
            break
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        actual_crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            errors.append(f"{path.relative_to(ROOT)} has an invalid {kind!r} chunk checksum")
        if kind in forbidden:
            errors.append(
                f"{path.relative_to(ROOT)} contains forbidden text or EXIF metadata ({kind.decode()})"
            )
        if kind == b"IHDR":
            saw_header = True
            width, height = struct.unpack(">II", payload[:8])
            if width < 1000 or height < 600:
                errors.append(
                    f"{path.relative_to(ROOT)} is too small for legible editor evidence ({width}x{height})"
                )
        if kind == b"IEND":
            saw_end = True
            if end != len(data):
                errors.append(f"{path.relative_to(ROOT)} has bytes after its IEND chunk")
            break
        offset = end
    if not saw_header or not saw_end:
        errors.append(f"{path.relative_to(ROOT)} is missing IHDR or IEND")
    return errors


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

    expected = {
        (package, feature)
        for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml"))
        for package in [tomllib.loads(manifest.read_text(encoding="utf-8")).get("package", {}).get("name")]
        if package
        for feature in tomllib.loads(manifest.read_text(encoding="utf-8")).get("features", {})
        if feature != "default"
    }
    for path in (FEATURE_DOC, FEATURE_MATRIX):
        try:
            actual = feature_contract_rows(path)
        except ValueError as error:
            errors.append(str(error))
            continue
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            errors.append(f"{path.relative_to(ROOT)} feature inventory drifted: missing={missing}, extra={extra}")

    for name in EDITOR_SCREENSHOTS:
        canonical = EDITOR_ASSETS / name
        book_copy = BOOK_EDITOR_ASSETS / name
        if not canonical.exists() or not book_copy.exists():
            errors.append(f"editor screenshot or Book copy is missing: {name}")
            continue
        errors.extend(check_editor_screenshot(canonical))
        if canonical.read_bytes() != book_copy.read_bytes():
            errors.append(f"Book editor screenshot drifted from docs/assets/editors/{name}")

    if errors:
        for error in errors:
            print(f"docs check: {error}", file=sys.stderr)
        return 1
    print(
        f"docs checks passed: {len(chapters)} book chapters, "
        f"{len(facade_features())} facade features, and {len(expected)} crate feature declarations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
