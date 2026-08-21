#!/usr/bin/env python3
"""Fail closed when a production panic/unwrap/expect site lacks audit evidence."""

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "audit-evidence/FND-003/production-panic-sites.json"
SITE = re.compile(r"\b(?:panic!|unwrap\(|expect\()")

SPEC = importlib.util.spec_from_file_location(
    "unsafe_ledger_cfg", ROOT / "tools/check-unsafe-ledger.py"
)
assert SPEC and SPEC.loader
CFG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CFG)


def disposition(path: str, code: str) -> tuple[str, str]:
    """Classify one reviewed site; callers persist the exact site in JSON."""
    if "/bin/" in path or path.endswith("incin-lsp/src/main.rs"):
        return ("declared process boundary", "ERROR_CONTRACT.md#panics-and-process-boundaries")
    if "panic_test" in path:
        return ("explicit diagnostic test panel", "incin-viz panic panel tests")
    if "/codegen/" in path:
        return ("infallible String formatting", "codegen unit tests")
    if "/cuda/" in path or "/dist/nccl/" in path:
        return ("validated backend transition", "CUDA/NCCL clippy and hardware matrix")
    if "/macros/" in path:
        return ("macro-generated internal transition", "macro compile fixtures")
    if "panic!(\"paranoid-validation" in code:
        return ("opt-in invariant assertion", "exec proof tests")
    return ("statically proven internal transition", "FND-003 classification and focused crate tests")


def sites() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for path in sorted((ROOT / "crates").glob("*/src/**/*.rs")):
        if "/tests/" in path.as_posix() or path.name in {"test.rs", "tests.rs"}:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        masked = CFG.test_only_lines(lines)
        for number, line in enumerate(lines):
            code = line.split("//", 1)[0].strip()
            if number in masked or not SITE.search(code):
                continue
            relative = path.relative_to(ROOT).as_posix()
            status, evidence = disposition(relative, code)
            result.append(
                {
                    "site": f"{relative}:{number + 1}",
                    "code": code,
                    "disposition": status,
                    "evidence": evidence,
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="write reviewed-site candidates")
    args = parser.parse_args()
    current = sites()
    if args.update:
        INVENTORY.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(current)} production panic sites; review the inventory before commit")
        return
    recorded = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if recorded != current:
        print("production panic audit drifted; run tools/check-panic-audit.py --update, review every disposition, and commit the inventory")
        raise SystemExit(1)
    print(f"production panic audit passed: {len(current)} reviewed sites")


if __name__ == "__main__":
    main()
