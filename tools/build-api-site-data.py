#!/usr/bin/env python3
"""Build the API-reference site's data payload from the generated capability doc.

`docs/capabilities.md` is itself generated from the registrations in
`crates/incin-backends/src/capability/tables.rs`, so this script never restates
a capability -- it reshapes the generated document into JSON. Run it after
`INCIN_DOCS=overwrite cargo test -p incin-backends --test generated_docs`.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKENDS = ["cpu", "cuda", "wgpu", "metal"]


def parse_capabilities(text: str):
    """Return (dtypes-by-operation, rules-by-backend) from the generated doc."""
    dtypes: dict[str, dict[str, list[str]]] = {}
    rules: dict[str, list[dict]] = {b: [] for b in BACKENDS}
    section = None
    backend = None

    for line in text.splitlines():
        heading = re.match(r"^## `(\w+)`$", line)
        if heading:
            backend, section = heading.group(1), "backend"
            continue
        if line.startswith("## Element types"):
            section, backend = "dtypes", None
            continue
        if line.startswith("## Reading this"):
            section = None
            continue
        if not line.startswith("| `"):
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if section == "dtypes":
            dtypes[cells[0].strip("`")] = {
                b: ([] if c == "—" else [x.strip().strip("`") for x in c.split(",")])
                for b, c in zip(BACKENDS, cells[1:])
            }
        elif section == "backend" and backend and len(cells) >= 6:
            rules[backend].append(
                {
                    "operation": cells[0].strip("`"),
                    "dtypes": []
                    if cells[1] == "—"
                    else [x.strip().strip("`") for x in cells[1].split(",")],
                    "layouts": []
                    if cells[2] == "—"
                    else [x.strip().strip("`") for x in cells[2].split(",")],
                    "rank": cells[3],
                    "training": cells[4] == "yes",
                    "impl": cells[5],
                }
            )
    return dtypes, rules


def main() -> int:
    doc = ROOT / "docs/capabilities.md"
    dtypes, rules = parse_capabilities(doc.read_text(encoding="utf-8"))

    # Join: most rule rows name an operation, a few name a rule *category*
    # (`pointwise`, `reduction`, ...). Category rules are kept separately rather
    # than guessed onto members, because guessing which operations a category
    # covers would invent a claim the generated document does not make.
    operations = []
    for name in sorted(dtypes):
        entry = {"name": name, "backends": {}}
        for b in BACKENDS:
            supported = dtypes[name][b]
            rule = next((r for r in rules[b] if r["operation"] == name), None)
            entry["backends"][b] = {
                "dtypes": supported,
                "layouts": rule["layouts"] if rule else [],
                "rank": rule["rank"] if rule else None,
                "training": rule["training"] if rule else None,
                "impl": rule["impl"] if rule else None,
            }
        operations.append(entry)

    categories = {
        b: [r for r in rules[b] if r["operation"] not in dtypes] for b in BACKENDS
    }

    surface = {}
    for f in sorted((ROOT / "docs/public-api").glob("*.txt")):
        surface[f.stem] = sum(1 for line in f.read_text().splitlines() if line.strip())

    payload = {
        "operations": operations,
        "categories": categories,
        "surface": surface,
        "backends": BACKENDS,
    }
    out = ROOT / "docs/api-site-data.json"
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}: {len(operations)} operations, "
          f"{sum(len(v) for v in categories.values())} category rules, "
          f"{len(surface)} public-API baselines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
