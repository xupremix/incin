#!/usr/bin/env python3
"""Keep `docs/plan/roadmap.md`'s completion table equal to `docs/plan/ledger.toml`.

The roadmap used to maintain its own counts. They drifted until the two
documents disagreed about the same task IDs — the roadmap said 47 of 100 tasks
were done while the ledger recorded 101 of 101, and nothing in the repository
noticed. A planning document that contradicts the record it summarises is worse
than no summary, because a reader cannot tell which one to act on.

This does not check *statuses*: the ledger owns those, along with the
`completed_on` and `deviations` evidence a generator cannot reproduce. It
checks that the roadmap's per-theme row counts, its total, and its deviation
count are the ones the ledger actually contains.

    python3 tools/check-roadmap-counts.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

LEDGER = Path("docs/plan/ledger.toml")
ROADMAP = Path("docs/plan/roadmap.md")

# The roadmap names themes with a prefix and an em-dash gloss, e.g.
# "`dist` — distributed". Only the backtick-quoted key is compared.
THEME_ROW = re.compile(r"^\|\s*`(\w+)`[^|]*\|\s*(\d+)\s*\|", re.M)
TOTAL_ROW = re.compile(r"^\|\s*\*\*Total\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|(.*)\|", re.M)
DEVIATION_COUNT = re.compile(r"(\d+)\s+rows carry")

# The roadmap abbreviates two theme keys for readability.
ALIASES = {"shp": "shape", "grd": "grad", "rel": "release"}


def ledger_counts() -> tuple[Counter, int]:
    text = LEDGER.read_text(encoding="utf8")
    blocks = text.split("[[task]]")[1:]
    themes: Counter = Counter()
    deviations = 0
    for block in blocks:
        theme = re.search(r'theme = "(\w+)"', block)
        if not theme:
            print(f"a [[task]] block has no theme:\n{block[:200]}", file=sys.stderr)
            sys.exit(1)
        themes[theme.group(1)] += 1
        recorded = re.search(r"deviations = \[(.*?)\]\n", block, re.S)
        if recorded and recorded.group(1).strip():
            deviations += 1
    return themes, deviations


def main() -> int:
    if not LEDGER.is_file() or not ROADMAP.is_file():
        print("run this from the workspace root", file=sys.stderr)
        return 1

    themes, deviations = ledger_counts()
    roadmap = ROADMAP.read_text(encoding="utf8")

    claimed = {
        ALIASES.get(key, key): int(count) for key, count in THEME_ROW.findall(roadmap)
    }
    problems: list[str] = []

    for theme, count in sorted(themes.items()):
        if theme not in claimed:
            problems.append(f"roadmap.md has no row for theme `{theme}` ({count} tasks)")
        elif claimed[theme] != count:
            problems.append(
                f"theme `{theme}`: roadmap.md says {claimed[theme]}, ledger.toml has {count}"
            )
    for theme in sorted(set(claimed) - set(themes)):
        problems.append(f"roadmap.md has a row for `{theme}`, which the ledger does not")

    total = TOTAL_ROW.search(roadmap)
    if not total:
        problems.append("roadmap.md has no **Total** row")
    elif int(total.group(1)) != sum(themes.values()):
        problems.append(
            f"total: roadmap.md says {total.group(1)}, ledger.toml has {sum(themes.values())}"
        )

    stated = DEVIATION_COUNT.search(roadmap)
    if not stated:
        problems.append("roadmap.md does not state how many rows carry deviations")
    elif int(stated.group(1)) != deviations:
        problems.append(
            f"deviations: roadmap.md says {stated.group(1)}, ledger.toml has {deviations}"
        )

    if problems:
        print("docs/plan/roadmap.md disagrees with docs/plan/ledger.toml:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(
        f"roadmap.md matches ledger.toml: {sum(themes.values())} tasks across "
        f"{len(themes)} themes, {deviations} with deviations"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
