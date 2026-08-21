#!/usr/bin/env python3
"""Keep every production unsafe block locally explained and audit-mapped.

This is deliberately a source audit, not a count of ``unsafe`` strings. It
recognises ``#[cfg(test)]`` / ``#[test]`` bodies, then checks each remaining
unsafe block for an immediately adjacent ``SAFETY:`` explanation. Blocks whose
shared proof is recorded at invariant-family level are still forced into the
ledger, where their test or sanitizer gate is named.
"""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/security/unsafe-ledger.md"
SOURCE = ROOT / "crates"
UNSAFE_BLOCK = re.compile(r"\bunsafe\s*\{")
CFG_ATTRIBUTE = re.compile(r"#\[\s*cfg\s*\((.*)\)\s*\]")


def split_cfg_arguments(text: str) -> list[str]:
    """Split a cfg combinator's arguments without confusing nested commas."""
    arguments: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(text[start:index].strip())
            start = index + 1
    arguments.append(text[start:].strip())
    return arguments


def cfg_guarantees_test(predicate: str) -> bool:
    """Whether every configuration satisfying ``predicate`` is test-only.

    This deliberately rejects predicates which merely *mention* test. For
    example, ``cfg(not(test))`` and ``cfg(any(feature = \"cpu\", test))`` can
    build production code and must stay in the audit; ``cfg(all(test, unix))``
    cannot.
    """
    predicate = predicate.strip()
    if predicate == "test":
        return True
    match = re.fullmatch(r"(all|any|not)\((.*)\)", predicate)
    if not match:
        return False
    operator, inner = match.groups()
    arguments = split_cfg_arguments(inner)
    if operator == "all":
        return any(cfg_guarantees_test(argument) for argument in arguments)
    if operator == "any":
        return bool(arguments) and all(cfg_guarantees_test(argument) for argument in arguments)
    return False


def line_starts_test_item(line: str) -> bool:
    """Recognise only attributes that guarantee the following item is test-only."""
    if re.fullmatch(r"\s*#\[\s*test\s*\]\s*", line):
        return True
    match = CFG_ATTRIBUTE.fullmatch(line.strip())
    return bool(match and cfg_guarantees_test(match.group(1)))


def test_only_lines(lines: list[str]) -> set[int]:
    """Return lines in test-only items using the following item's braces."""
    masked: set[int] = set()
    pending: int | None = None
    active = False
    depth = 0
    for number, line in enumerate(lines):
        if not active and pending is None and line_starts_test_item(line):
            pending = number
            continue
        if not active and pending is not None:
            if "{" not in line:
                continue
            depth = line.count("{") - line.count("}")
            if depth <= 0:
                pending = None
                continue
            masked.update(range(pending, number + 1))
            pending = None
            active = True
            continue
        if active:
            masked.add(number)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                active = False
    return masked


def has_local_safety_comment(lines: list[str], line_number: int) -> bool:
    """Accept the contiguous comment immediately preceding an unsafe block."""
    cursor = line_number - 1
    comments: list[str] = []
    while cursor >= 0:
        text = lines[cursor].strip()
        if not text:
            cursor -= 1
            continue
        if text.startswith("//"):
            comments.append(text)
            cursor -= 1
            continue
        break
    return any("SAFETY:" in text for text in comments)


def production_unsafe_blocks(path: Path) -> list[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    masked = test_only_lines(lines)
    return [
        number
        for number, line in enumerate(lines)
        if number not in masked
        and UNSAFE_BLOCK.search(line.split("//", 1)[0])
    ]


def main() -> None:
    source_sites: dict[str, list[int]] = {}
    local_comments = 0
    for path in SOURCE.rglob("*.rs"):
        if "/tests/" in path.as_posix():
            continue
        sites = production_unsafe_blocks(path)
        if not sites:
            continue
        relative = path.relative_to(ROOT).as_posix()
        source_sites[relative] = sites
        lines = path.read_text(encoding="utf-8").splitlines()
        for number in sites:
            if has_local_safety_comment(lines, number):
                local_comments += 1

    source_files = set(source_sites)
    ledger_text = LEDGER.read_text(encoding="utf-8").split("## Non-production unsafe", 1)[0]
    ledger_files = set(re.findall(r"`(crates/[^`]+\.rs)`", ledger_text))
    missing = sorted(source_files - ledger_files)
    stale = sorted(ledger_files - source_files)
    if missing or stale:
        if missing:
            print("unsafe files missing from ledger:", *missing, sep="\n  ")
        if stale:
            print("stale unsafe ledger entries:", *stale, sep="\n  ")
        raise SystemExit(1)
    print(
        "unsafe ledger passed: "
        f"{len(source_files)} production files, "
        f"{sum(map(len, source_sites.values()))} unsafe blocks "
        f"({local_comments} locally annotated; remaining blocks have a shared family proof)"
    )


if __name__ == "__main__":
    main()
