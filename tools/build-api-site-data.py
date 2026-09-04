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

# Doc sentences are read only from the user-facing surface. Backend internals
# define same-named helpers whose docs describe a kernel rather than the API,
# and letting them compete is how an earlier draft captioned `dropout` with
# `MSELoss::forward`'s sentence.
DOC_ROOTS = ("tensor", "nn", "distributions")


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


def parse_catalog(text: str) -> dict[str, dict]:
    """Rows of the single operation declaration in `operation_catalog.rs`.

    Each row carries the wire name, the shape-error category, the family, the
    attribute type, the operand arity and the public API path the operation is
    reached through -- everything a reference needs except prose.
    """
    rows = re.findall(
        r'\(\s*(\w+)\s*,\s*"([^"]+)"\s*,\s*(\w+)\s*,\s*(\w+)\s*,'
        r'\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*"([^"]*)"\s*\)',
        text,
    )
    return {
        wire: {
            "variant": variant,
            "kind": kind,
            "family": family,
            "attrs": attrs,
            "arity": [int(lo), int(hi)],
            "api": api,
        }
        for variant, wire, kind, family, attrs, lo, hi, api in rows
    }


def _first_sentence(block: str) -> str | None:
    lines: list[str] = []
    for raw in block.splitlines():
        line = raw.strip().lstrip("/").strip()
        if not line or line.startswith(("#", "```", "[")):
            break
        lines.append(line)
    if not lines:
        return None
    text = re.sub(r"\s+", " ", " ".join(lines))
    end = re.search(r"\.(\s|$)", text)
    if end:
        text = text[: end.start() + 1]
    # A truncated clause reads worse than no description at all.
    if text.count("(") != text.count(")"):
        return None
    if text.endswith((",", "see", "the", "a", "of", "and")):
        return None
    return text if len(text) > 12 else None


def attach_docs(catalog: dict[str, dict]) -> int:
    """Attach the doc sentence above each operation's public entry point.

    A method resolving to two different sentences is left undescribed rather
    than captioned with whichever file was walked first: a wrong description
    is worse than none.
    """
    blobs = {
        f: f.read_text(encoding="utf-8", errors="ignore")
        for root in DOC_ROOTS
        for f in (ROOT / "crates/incin-core/src" / root).rglob("*.rs")
    }
    resolved = 0
    for entry in catalog.values():
        entry["doc"] = None
        api = entry.get("api") or ""
        if "::" not in api:
            continue
        owner, method = api.rsplit("::", 1)
        owner = owner.strip(":")
        pattern = re.compile(
            r"((?:^[ \t]*///.*\n)+)[ \t]*(?:#\[[^\]]*\]\s*\n[ \t]*)*pub fn "
            + re.escape(method)
            + r"\b",
            re.M,
        )
        found = []
        for path, blob in blobs.items():
            for match in pattern.finditer(blob):
                sentence = _first_sentence(match.group(1))
                if sentence:
                    found.append((path, sentence))
        if not found:
            continue

        # An owner that names a *type* must resolve inside that type's own
        # file. Without this, `Adam::step`, `AdamW::step` and `SGD::step` all
        # matched the single `pub fn step` in the activation module and were
        # captioned "Applies the Step function element-wise" -- three
        # confidently wrong descriptions, which is the outcome this function
        # exists to avoid. Uniqueness is not enough: one match can still be the
        # wrong one when nothing checks whose method it is.
        if owner and owner != "Tensor":
            snake = re.sub(r"(?<!^)(?=[A-Z])", "_", owner).lower()
            scoped = {s for path, s in found if path.stem == snake}
            if len(scoped) == 1:
                entry["doc"] = scoped.pop()
                resolved += 1
            continue

        # No owner, or the tensor surface itself, where the method name is the
        # identity. A method resolving to two different sentences is left
        # undescribed rather than captioned with whichever file was walked
        # first.
        unique = {s for _, s in found}
        if len(unique) == 1:
            entry["doc"] = found[0][1]
            resolved += 1
    return resolved


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

    catalog = parse_catalog(
        (ROOT / "crates/incin-core/src/operation_catalog.rs").read_text(encoding="utf-8")
    )
    described = attach_docs(catalog)
    for op in operations:
        op["catalog"] = catalog.get(op["name"])

    # Resolved and checked out of band by tools/build-docsrs-links.py, and read
    # here from the committed file so this generator stays offline and its
    # output stays reproducible for the drift check CI runs against it.
    links_file = ROOT / "docs/api-docsrs-links.json"
    links = json.loads(links_file.read_text(encoding="utf-8"))["links"]
    missing = [op["name"] for op in operations if op["name"] not in links]
    if missing:
        raise SystemExit(
            "no docs.rs link for: " + ", ".join(missing) +
            "\nrun: python3 tools/build-docsrs-links.py"
        )
    for op in operations:
        op["docs"] = links[op["name"]]

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
    print(
        f"wrote {out.relative_to(ROOT)}: {len(operations)} operations, "
        f"{sum(1 for o in operations if o['catalog'])} with catalog metadata, "
        f"{described} with a doc sentence, "
        f"{sum(len(v) for v in categories.values())} category rules, "
        f"{len(surface)} public-API baselines"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
