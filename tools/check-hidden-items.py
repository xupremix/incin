#!/usr/bin/env python3
"""Keep the reviewed inventory of hidden public items complete."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def extract_hidden_items(path: Path) -> list[str]:
    relative = path.relative_to(ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    items = []

    container_stack: list[tuple[str, str, int]] = []
    current_brace_depth = 0

    for i, line in enumerate(lines):
        cleaned = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
        cleaned = cleaned.split("//")[0]

        m_enum = re.search(r"\b(?:pub(?:\([^)]+\))?\s+)?enum\s+([A-Za-z0-9_]+)", cleaned)
        m_struct = re.search(r"\b(?:pub(?:\([^)]+\))?\s+)?struct\s+([A-Za-z0-9_]+)", cleaned)
        m_trait = re.search(r"\b(?:pub(?:\([^)]+\))?\s+)?trait\s+([A-Za-z0-9_]+)", cleaned)
        m_impl = re.search(
            r"\bimpl(?:<[^>]+>)?\s+(?:([A-Za-z0-9_]+)\s+for\s+)?([A-Za-z0-9_]+)",
            cleaned,
        )

        container_to_push = None
        if m_enum:
            container_to_push = ("enum", m_enum.group(1))
        elif m_trait:
            container_to_push = ("trait", m_trait.group(1))
        elif m_impl:
            target = m_impl.group(1) if m_impl.group(1) else m_impl.group(2)
            container_to_push = ("impl", target)
        elif m_struct:
            container_to_push = ("struct", m_struct.group(1))

        for char in cleaned:
            if char == "{":
                current_brace_depth += 1
                if container_to_push:
                    container_stack.append(
                        (container_to_push[0], container_to_push[1], current_brace_depth)
                    )
                    container_to_push = None
            elif char == "}":
                if container_stack and container_stack[-1][2] == current_brace_depth:
                    container_stack.pop()
                current_brace_depth -= 1

        if "#[doc(hidden)]" in line:
            j = i + 1
            while j < len(lines) and (
                lines[j].strip().startswith("#[")
                or not lines[j].strip()
                or lines[j].strip().startswith("//")
            ):
                j += 1
            if j < len(lines):
                target = lines[j].split("//")[0].strip()
                m_fn = re.search(
                    r"\b(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?(?:const\s+)?fn\s+([A-Za-z0-9_]+)",
                    target,
                )
                m_st = re.search(r"\b(?:pub(?:\([^)]+\))?\s+)?struct\s+([A-Za-z0-9_]+)", target)
                m_en = re.search(r"\b(?:pub(?:\([^)]+\))?\s+)?enum\s+([A-Za-z0-9_]+)", target)
                m_tr = re.search(r"\b(?:pub(?:\([^)]+\))?\s+)?trait\s+([A-Za-z0-9_]+)", target)
                m_ty = re.search(r"\b(?:pub(?:\([^)]+\))?\s+)?type\s+([A-Za-z0-9_]+)", target)
                m_co = re.search(r"\b(?:pub(?:\([^)]+\))?\s+)?const\s+([A-Za-z0-9_]+)", target)
                m_mo = re.search(r"\b(?:pub(?:\([^)]+\))?\s+)?mod\s+([A-Za-z0-9_]+)", target)
                m_ma = re.search(r"\bmacro_rules!\s+([A-Za-z0-9_]+)", target)
                m_va = re.match(r"^([A-Za-z0-9_]+)(?:\s*\(|\s*\{|\s*,|\s*$)", target)

                name = None
                if m_fn:
                    name = m_fn.group(1)
                elif m_st:
                    name = m_st.group(1)
                elif m_en:
                    name = m_en.group(1)
                elif m_tr:
                    name = m_tr.group(1)
                elif m_ty:
                    name = m_ty.group(1)
                elif m_co:
                    name = m_co.group(1)
                elif m_mo:
                    name = m_mo.group(1)
                elif m_ma:
                    name = m_ma.group(1)
                elif m_va:
                    name = m_va.group(1)
                else:
                    name = f"unknown_{j+1}"

                if (
                    container_stack
                    and container_stack[-1][0] == "enum"
                    and m_va
                    and not (m_fn or m_st or m_en or m_tr or m_ty or m_co or m_mo or m_ma)
                ):
                    name = f"{container_stack[-1][1]}::{name}"

                full_id = f"{relative}::{name}"
                items.append(full_id)
    return items


def main() -> None:
    actual_files = set()
    actual_items = set()
    for path in (ROOT / "crates").rglob("*.rs"):
        if "/target/" in path.as_posix():
            continue
        relative = path.relative_to(ROOT).as_posix()
        for item in extract_hidden_items(path):
            actual_files.add(relative)
            actual_items.add(item)
    text = (ROOT / "docs/public-api/hidden-items.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^-\s+`(crates/[^`]+\.rs)`", text, re.MULTILINE))
    reviewed = set(re.findall(r"`(crates/[^`]+\.rs::[^`]+)`", text))
    review_rows = re.findall(
        r"^\|\s+`(crates/[^`]+\.rs::[^`]+)`\s+\|\s+([ABC])\s+\|\s+.+\s+\|$",
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
