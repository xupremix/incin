#!/usr/bin/env python3
"""Keep the reviewed source-file inventory of hidden public items complete."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "crates").rglob("*.rs")
        if "#[doc(hidden)]" in path.read_text(encoding="utf-8")
        and "/target/" not in path.as_posix()
    }
    text = (ROOT / "docs/public-api/hidden-items.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"`(crates/[^`]+\.rs)`", text))
    missing = sorted(actual - listed)
    stale = sorted(listed - actual)
    if missing or stale:
        if missing:
            print("hidden-item files missing from inventory:", *missing, sep="\n  ")
        if stale:
            print("stale hidden-item inventory entries:", *stale, sep="\n  ")
        raise SystemExit(1)
    print(f"hidden-item inventory passed: {len(actual)} source files")


if __name__ == "__main__":
    main()
