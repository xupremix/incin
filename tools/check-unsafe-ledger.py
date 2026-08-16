#!/usr/bin/env python3
"""Ensure every production source file containing unsafe is in the ledger."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/security/unsafe-ledger.md"
SOURCE = ROOT / "crates"


def main() -> None:
    source_files = {
        path.relative_to(ROOT).as_posix()
        for path in SOURCE.rglob("*.rs")
        if re.search(r"\bunsafe\b", path.read_text(encoding="utf-8"))
        and "/tests/" not in path.as_posix()
    }
    ledger_text = LEDGER.read_text(encoding="utf-8")
    ledger_files = set(re.findall(r"`(crates/[^`]+\.rs)`", ledger_text))
    missing = sorted(source_files - ledger_files)
    stale = sorted(ledger_files - source_files)
    if missing or stale:
        if missing:
            print("unsafe files missing from ledger:", *missing, sep="\n  ")
        if stale:
            print("stale unsafe ledger entries:", *stale, sep="\n  ")
        raise SystemExit(1)
    print(f"unsafe ledger passed: {len(source_files)} production source files")


if __name__ == "__main__":
    main()
