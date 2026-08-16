#!/usr/bin/env python3
"""Compare the reviewed CPU facade API against its checked-in baseline."""

from pathlib import Path
import difflib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/public-api/incin-cpu.txt"


def main() -> int:
    command = [
        "cargo",
        "public-api",
        "-sss",
        "--color",
        "never",
        "-p",
        "incin",
        "--no-default-features",
        "--features",
        "cpu",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode

    actual = sorted(set(result.stdout.splitlines()))
    expected = BASELINE.read_text(encoding="utf-8").splitlines()
    if actual != expected:
        diff = difflib.unified_diff(expected, actual, fromfile=str(BASELINE), tofile="current")
        sys.stderr.write("public API baseline mismatch; review the change and update the baseline deliberately:\n")
        sys.stderr.writelines(diff)
        return 1

    print(f"public API baseline passed: {len(actual)} reviewed facade entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
