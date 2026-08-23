#!/usr/bin/env python3
"""Compare every shipped crate's public API against its checked-in baseline.

Each entry pins the package, the feature set the baseline was generated
with, and the baseline file. A mismatch fails with a unified diff; the
resolution is always a deliberate review followed by regenerating that
one baseline in the same commit.
"""

from pathlib import Path
import difflib
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "docs/public-api"

# rustdoc's JSON rendering of generic bounds changes across nightlies, so
# baseline generation pins one dated nightly. Override with
# PUBLIC_API_TOOLCHAIN only while regenerating after a deliberate pin bump.
PINNED_TOOLCHAIN = os.environ.get("PUBLIC_API_TOOLCHAIN", "nightly-2026-07-28")

# (crate, baseline file, cargo feature arguments)
BASELINES = [
    ("incin", "incin-cpu.txt", ["--no-default-features", "--features", "cpu"]),
    ("incin-core", "incin-core-std.txt", ["--no-default-features", "--features", "std"]),
    ("incin-backends", "incin-backends-cpu.txt", ["--no-default-features", "--features", "std,cpu"]),
    ("incin-data", "incin-data.txt", []),
    ("incin-diagnostics", "incin-diagnostics.txt", []),
    ("incin-telemetry", "incin-telemetry.txt", []),
    ("incin-macros", "incin-macros.txt", []),
    ("incin-lsp", "incin-lsp.txt", []),
    ("incin-viz-plugin-api", "incin-viz-plugin-api.txt", []),
    ("incin-viz", "incin-viz.txt", []),
]


def generate(crate: str, features: list[str]) -> list[str]:
    command = ["cargo", f"+{PINNED_TOOLCHAIN}", "public-api", "-sss", "--color", "never", "-p", crate]
    command.extend(features)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return sorted(set(result.stdout.splitlines()))


def main() -> int:
    update = "--update" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--update"]
    only = args[0] if args else None
    failures = 0
    checked = 0
    for crate, baseline_name, features in BASELINES:
        if only and crate != only:
            continue
        baseline = API_DIR / baseline_name
        actual = generate(crate, features)
        if update:
            baseline.write_text("\n".join(actual) + "\n", encoding="utf-8")
            print(f"public API baseline updated: {crate} ({len(actual)} entries)")
            checked += 1
            continue
        expected = baseline.read_text(encoding="utf-8").splitlines()
        if actual != expected:
            diff = difflib.unified_diff(
                expected, actual, fromfile=str(baseline), tofile="current"
            )
            sys.stderr.write(
                f"public API baseline mismatch for {crate}; review the change "
                "and update the baseline deliberately:\n"
            )
            sys.stderr.writelines(diff)
            failures += 1
        else:
            print(f"public API baseline passed: {crate} ({len(actual)} entries)")
        checked += 1
    if update:
        return 0
    if failures:
        sys.stderr.write(f"{failures} baseline(s) drifted\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
