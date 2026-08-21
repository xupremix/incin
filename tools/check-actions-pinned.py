#!/usr/bin/env python3
"""Require immutable, readable references for external workflow actions."""
from __future__ import annotations

import re
import sys
from pathlib import Path


USES = re.compile(r"^\s*uses:\s*([^\s#]+)(?:\s+#\s*(.*))?\s*$", re.MULTILINE)
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def check_workflow_text(text: str, source: str = "workflow") -> list[str]:
    errors: list[str] = []
    for match in USES.finditer(text):
        reference = match.group(1)
        if reference.startswith("./"):
            continue
        if "@" not in reference:
            errors.append(f"{source}:{text.count(chr(10), 0, match.start()) + 1}: action has no ref: {reference}")
            continue
        action, ref = reference.rsplit("@", 1)
        if not COMMIT.fullmatch(ref):
            errors.append(f"{source}:{text.count(chr(10), 0, match.start()) + 1}: {action} is not pinned to a lowercase commit SHA")
        if not (match.group(2) or "").strip():
            errors.append(f"{source}:{text.count(chr(10), 0, match.start()) + 1}: {action} pin needs a readable version comment")
    return errors


def main() -> int:
    errors: list[str] = []
    workflows = sorted(Path(".github/workflows").glob("*.y*ml"))
    for workflow in workflows:
        errors.extend(check_workflow_text(workflow.read_text(encoding="utf-8"), str(workflow)))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"action pin checks passed ({len(workflows)} workflows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
