#!/usr/bin/env python3
"""Keep the reviewed inventory of hidden public items complete."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    actual_files = set()
    actual_items = set()
    for path in (ROOT / "crates").rglob("*.rs"):
        if "/target/" in path.as_posix():
            continue
        relative = path.relative_to(ROOT).as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "#[doc(hidden)]" in line:
                actual_files.add(relative)
                actual_items.add(f"{relative}:{line_number}")
    text = (ROOT / "docs/public-api/hidden-items.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"`(crates/[^`]+\.rs)`", text))
    reviewed = set(re.findall(r"`(crates/[^`]+\.rs:[0-9]+)`", text))
    review_rows = re.findall(
        r"^\| `(crates/[^`]+\.rs:[0-9]+)` \| ([ABC]) \| .+ \|$",
        text,
        re.MULTILINE,
    )
    reviewed_with_classification = {location for location, _ in review_rows}
    missing = sorted(actual_files - listed)
    stale = sorted(listed - actual_files)
    missing_items = sorted(actual_items - reviewed)
    stale_items = sorted(reviewed - actual_items)
    unclassified_items = sorted(actual_items - reviewed_with_classification)
    if missing or stale or missing_items or stale_items or unclassified_items:
        if missing:
            print("hidden-item files missing from inventory:", *missing, sep="\n  ")
        if stale:
            print("stale hidden-item inventory entries:", *stale, sep="\n  ")
        if missing_items:
            print("hidden-item occurrences missing from review:", *missing_items, sep="\n  ")
        if stale_items:
            print("stale hidden-item occurrence reviews:", *stale_items, sep="\n  ")
        if unclassified_items:
            print(
                "hidden-item occurrences missing A/B/C classification:",
                *unclassified_items,
                sep="\n  ",
            )
        raise SystemExit(1)
    print(
        f"hidden-item inventory passed: {len(actual_files)} source files, "
        f"{len(actual_items)} occurrences"
    )


if __name__ == "__main__":
    main()
