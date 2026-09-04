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



# Baselines to read the current public surface from, and the crate each one
# documents. These are the reviewed `cargo public-api` dumps the repository
# already gates on, so the type reference is checkable rather than narrated.
TYPE_BASELINES = {
    "incin-cpu": "incin",
    "incin-core-std": "incin_core",
    "incin-backends-cpu": "incin_backends",
    "incin-data": "incin_data",
    "incin-telemetry": "incin_telemetry",
    "incin-viz": "incin_viz",
    "incin-viz-plugin-api": "incin_viz_plugin_api",
    "incin-diagnostics": "incin_diagnostics",
    "incin-lsp": "incin_lsp",
    "incin-macros": "incin_macros",
}
CRATE_OF_MODULE = {
    "incin": "incin", "incin_core": "incin-core", "incin_backends": "incin-backends",
    "incin_data": "incin-data", "incin_telemetry": "incin-telemetry",
    "incin_viz": "incin-viz", "incin_viz_plugin_api": "incin-viz-plugin-api",
    "incin_diagnostics": "incin-diagnostics", "incin_lsp": "incin-lsp",
    "incin_macros": "incin-macros",
}
TYPE_DECL = re.compile(r"^pub (struct|trait|enum|type) ([A-Za-z0-9_:]+)")



# The stages canonical dispatch runs, in order, with the failure class each
# one owns. Taken from the lowering chapter and exec/dispatch.rs, which keep
# these apart deliberately: the class says who is at fault.
DISPATCH_FLOW = [
    {
        "stage": "logical metadata",
        "detail": "Shape, dtype and device are read off the storage that will "
                  "actually run, so validation cannot be satisfied by metadata "
                  "describing some other tensor.",
        "error": None,
    },
    {
        "stage": "output inference and cross-check",
        "detail": "Outputs are computed from the operation and its inputs, never "
                  "dictated by the caller; the caller's predicted shape must agree "
                  "with what was inferred.",
        "error": "DescriptorError",
    },
    {
        "stage": "payload validation",
        "detail": "The descriptor's own invariants are checked before anything is "
                  "asked of a backend.",
        "error": "DescriptorError",
    },
    {
        "stage": "capability admission",
        "detail": "The backend's registry is asked whether it accepts this "
                  "operation over these operands, and the context's fallback "
                  "policy filters the answer, so no route reaches a kernel with a "
                  "composed or transfer fallback it was not granted.",
        "error": "PolicyViolation",
    },
    {
        "stage": "backend launch",
        "detail": "The kernel runs. A failure here means a legal request failed at "
                  "or after launch, which is the device's fault rather than the "
                  "caller's.",
        "error": "BackendError",
    },
]




# Which chapters carry the worked examples for each reference section. The
# mapping is stated rather than guessed from the prose: an example is shown
# under a section because that chapter is about that subject, not because a
# keyword matched.
EXAMPLE_CHAPTERS = {
    "operations": ["tensors", "quickstart", "building_models", "pytorch_cheatsheet"],
    "dtypes": ["quantization", "tensors"],
    "layouts": ["layout"],
    "shapes": ["shapes", "advanced_shapes", "macros"],
    "target": ["target_api", "backend_authoring"],
    "flow": ["deep_lowering", "proofs_to_execution", "custom_operations"],
    "backends": ["backends", "backend_conformance"],
    "types": ["deep_type_semantics", "invariants"],
}
FENCE = re.compile(r"^```(rust[\w,]*)$")



REPO_BLOB = "https://github.com/xupremix/incin/blob/master/"
FN_START = re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+([a-z0-9_]+)", re.M)


def _fn_body(text: str, start: int):
    """The whole function beginning at `start`, brace-matched.

    A test function is short and self-contained, which makes it a better
    snippet than either the whole file or an arbitrary window around a match.
    """
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth, index = 0, brace
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        index += 1
    if index >= len(text):
        return None
    body = text[start:index + 1]
    lines = body.count("\n")
    return body if 2 <= lines <= 45 else None


def collect_snippets() -> list:
    """A pool of real, compiled usages: book examples, then test functions.

    Book blocks come first because they were written to be read. Test
    functions fill in what the book does not reach -- they are compiled and
    run, so they cannot describe an API that no longer exists.
    """
    pool = []
    for path in sorted((ROOT / "docs/book/src").glob("*.md")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        heading, index = None, 0
        while index < len(lines):
            if lines[index].startswith("#"):
                heading = lines[index].lstrip("#").strip()
            match = FENCE.match(lines[index].strip())
            if match:
                end = index + 1
                while end < len(lines) and lines[end].strip() != "```":
                    end += 1
                body = "\n".join(
                    l for l in lines[index + 1:end]
                    if not l.lstrip().startswith("# ") and l.strip() != "#"
                ).strip()
                if body:
                    pool.append({
                        "origin": "book",
                        "label": heading or path.stem,
                        "where": path.stem,
                        "href": "./#/" + path.stem,
                        "checked": "ignore" not in match.group(1),
                        "code": body,
                    })
                index = end
            index += 1

    # The runnable custom-operation examples live beside the backend, not in
    # crates/incin/examples/, so the second pattern misses them: without the
    # third, opening CpuStorage or f64 shows test snippets but never the
    # calibration/polar programs that use them end to end.
    for pattern in (
        "crates/*/tests/*.rs",
        "crates/incin/examples/*/*.rs",
        "crates/incin-backends/examples/*.rs",
    ):
        for path in sorted(ROOT.glob(pattern)):
            rel = path.relative_to(ROOT).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in FN_START.finditer(text):
                body = _fn_body(text, match.start())
                if body is None:
                    continue
                name = match.group(1)
                pool.append({
                    "origin": "test",
                    "label": name,
                    "where": rel,
                    "href": REPO_BLOB + rel,
                    "checked": True,
                    "code": body.strip(),
                })
    # Identical bodies appear in more than one place; keep the first.
    seen, unique = set(), []
    for snippet in pool:
        key = snippet["code"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(snippet)
    return unique


def index_usage(pool: list, names: list, limit: int = 3) -> dict:
    """Which snippets literally use each name.

    A literal word match is a fact about the snippet, not a guess about its
    meaning: if `RowMajor` appears in the code, that code uses `RowMajor`. The
    page says "used in", never "the example for", because a name can appear
    incidentally. Book snippets are offered first.
    """
    order = sorted(range(len(pool)), key=lambda i: (pool[i]["origin"] != "book", len(pool[i]["code"])))
    out = {}
    for name in names:
        if len(name) < 3:
            continue
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        hits = []
        for i in order:
            if pattern.search(pool[i]["code"]):
                hits.append(i)
                if len(hits) == limit:
                    break
        if hits:
            out[name] = hits
    return out


def collect_examples() -> list:
    """Every worked example in the book, with whether the compiler checks it.

    The chapters are `include_str!`d into a doctest-only module in the facade,
    so a `no_run` or `compile_fail` block is compiled by CI. An `ignore` block
    is not, and is labelled that way here rather than presented as if it were
    checked -- 25 of the 96 are in that state.
    """
    # A chapter can serve more than one section: `tensors` carries both the
    # operation examples and the element-type ones.
    sections_of = {}
    for section, chapters in EXAMPLE_CHAPTERS.items():
        for chapter in chapters:
            sections_of.setdefault(chapter, []).append(section)

    out = []
    for path in sorted((ROOT / "docs/book/src").glob("*.md")):
        chapter = path.stem
        if chapter not in sections_of:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        heading, index = None, 0
        while index < len(lines):
            line = lines[index]
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
            match = FENCE.match(line.strip())
            if match:
                end = index + 1
                while end < len(lines) and lines[end].strip() != "```":
                    end += 1
                body = "\n".join(lines[index + 1:end])
                # Hidden doctest scaffolding is not part of the example.
                shown = "\n".join(
                    l for l in body.splitlines()
                    if not l.lstrip().startswith("# ") and l.strip() != "#"
                ).strip()
                tags = match.group(1)
                for section in sections_of[chapter] if shown else []:
                    out.append({
                        "section": section,
                        "chapter": chapter,
                        "heading": heading or chapter,
                        "checked": "ignore" not in tags,
                        "tags": tags,
                        "code": shown,
                    })
                index = end
            index += 1
    return out


DECL = re.compile(
    r"((?:^[ \t]*///.*\n)+)(?:^[ \t]*#\[[^\]]*\]\n)*^[ \t]*pub (struct|trait|enum) ([A-Za-z0-9_]+)",
    re.M,
)
METHOD = re.compile(
    r"((?:^[ \t]*///.*\n)+)(?:^[ \t]*#\[[^\]]*\]\n)*^[ \t]*fn ([a-z0-9_]+)",
    re.M,
)


def first_sentence(doc_block: str) -> str:
    """The first documented sentence, with rustdoc's markup left alone.

    The reference quotes what the source says rather than paraphrasing it, so
    a claim on the page and the claim in the crate cannot drift apart.
    """
    lines = []
    for raw in doc_block.splitlines():
        text = raw.strip()
        text = text[3:].strip() if text.startswith("///") else text
        if not text:
            break
        lines.append(text)
    joined = " ".join(lines)
    cut = joined.find(". ")
    if cut > 0:
        joined = joined[: cut + 1]
    return joined.strip()


def declared_items(path: pathlib.Path) -> list:
    source = path.read_text(encoding="utf-8")
    return [
        {"kind": m.group(2), "name": m.group(3), "doc": first_sentence(m.group(1))}
        for m in DECL.finditer(source)
    ]


def trait_methods(path: pathlib.Path) -> list:
    source = path.read_text(encoding="utf-8")
    return [
        {"name": m.group(2), "doc": first_sentence(m.group(1))}
        for m in METHOD.finditer(source)
    ]


def parse_encodings(source: str) -> dict:
    """Each dtype's storage encoding, read from the descriptors that set it."""
    out = {}
    pattern = re.compile(
        r"DTypeId::([A-Za-z0-9_]+) => DTypeDescriptor::builtin\((?:[^()]|\([^()]*\))*?"
        r"DTypeKind::([A-Za-z]+),\s*(?:// [^\n]*\n\s*)?StorageEncoding::(scalar|block)\(([^)]*)\)",
        re.S,
    )
    for match in pattern.finditer(source):
        name, kind, form, args = match.groups()
        nums = [int(a.strip()) for a in args.split(",")]
        if form == "scalar":
            per_block, block_bytes, align = 1, nums[0], nums[1]
        else:
            per_block, block_bytes, align = nums[0], nums[1], nums[2]
        out[name.lower()] = {
            "kind": kind,
            "elementsPerBlock": per_block,
            "bytesPerBlock": block_bytes,
            "alignment": align,
            "bitsPerElement": round(block_bytes * 8 / per_block, 2),
        }
    return out


def parse_dtypes(source: str) -> dict:
    """Read the element types and their doc sentences out of `DTypeId`."""
    body = source.split("pub enum DTypeId {", 1)[1].split("\n}", 1)[0]
    out, pending = {}, []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("///"):
            pending.append(line[3:].strip())
        elif line.startswith("#["):
            continue
        elif line.endswith(","):
            name = line[:-1].strip()
            if name:
                out[name.lower()] = " ".join(p for p in pending if p).strip()
            pending = []
    return out


def collect_types(links: dict) -> list:
    """The public type surface, each entry linked where docs.rs publishes it.

    The canonical path a baseline records is often not the path rustdoc
    documents an item at, because re-exports move it, so the exact path is
    tried first and a bare name accepted only where it is unambiguous in that
    crate. An item with neither match carries no link: the released version
    predates it, and a link that lands nowhere is worse than none.
    """
    out, seen = [], set()
    for baseline, module in TYPE_BASELINES.items():
        path = ROOT / "docs/public-api" / f"{baseline}.txt"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = TYPE_DECL.match(line)
            if not match:
                continue
            kind, full = match.group(1), match.group(2)
            parts = full.split("::")
            if len(parts) < 2:
                continue
            crate = CRATE_OF_MODULE.get(parts[0])
            if crate is None:
                continue
            name, mods = parts[-1], parts[1:-1]
            key = (crate, kind, tuple(mods), name)
            if key in seen:
                continue
            seen.add(key)
            entry = links.get("crates", {}).get(crate)
            url = None
            if entry:
                rel = entry["by_path"].get(f"{kind}|{'/'.join(mods)}|{name}")
                if rel is None:
                    rel = entry["by_name"].get(f"{kind}|{name}")
                if rel:
                    url = entry["base"] + rel
            out.append({
                "crate": crate,
                "kind": kind,
                "name": name,
                "module": "::".join(mods),
                "url": url,
            })
    out.sort(key=lambda t: (t["crate"], t["name"].lower(), t["kind"]))
    return out


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

    # The element types, read from the enum that defines them rather than
    # restated here, so the list cannot drift from the crate.
    dtype_doc = parse_dtypes(
        (ROOT / "crates/incin-core/src/tensor/dtype/registry.rs").read_text(encoding="utf-8")
    )
    used = {d: set() for d in dtype_doc}
    for op in operations:
        for b in BACKENDS:
            for d in op["backends"][b]["dtypes"]:
                used.setdefault(d, set()).add(b)
    encodings = parse_encodings(
        (ROOT / "crates/incin-core/src/tensor/dtype/registry.rs").read_text(encoding="utf-8")
    )
    dtypes_out = [
        {
            "id": key,
            "doc": doc,
            "encoding": encodings.get(key),
            "backends": sorted(used.get(key, ())),
            "operations": sum(
                1 for op in operations
                if any(key in op["backends"][b]["dtypes"] for b in BACKENDS)
            ),
        }
        for key, doc in dtype_doc.items()
    ]

    shapes_dir = ROOT / "crates/incin-core/src/shapes"
    shape_groups = []
    for module in sorted(shapes_dir.glob("*.rs")):
        if module.stem in {"mod", "layout"}:
            continue
        items = declared_items(module)
        if items:
            shape_groups.append({"module": module.stem, "items": items})

    layout_out = {
        # `SealedFresh` is public only so a sealed trait can name it; it is not
        # part of the surface a reader can use, so it is not listed as if it were.
        "items": [i for i in declared_items(shapes_dir / "layout.rs")
                  if not i["name"].startswith("Sealed")]
                 + [i for i in declared_items(shapes_dir / "dynamic.rs") if i["name"] == "Dyn"],
        "byBackend": [
            {
                "id": b,
                "contiguous": sum(1 for op in operations
                                  if "contiguous" in op["backends"][b]["layouts"]),
                "strided": sum(1 for op in operations
                               if "strided" in op["backends"][b]["layouts"]),
            }
            for b in BACKENDS
        ],
    }

    target_out = trait_methods(ROOT / "crates/incin-backends/src/target/ext.rs")

    backends_out = []
    for b in BACKENDS:
        advertised = [op for op in operations if op["backends"][b]["dtypes"]]
        backends_out.append({
            "id": b,
            "operations": len(advertised),
            "native": sum(1 for op in advertised if op["backends"][b]["impl"] == "native"),
            "composed": sum(1 for op in advertised if op["backends"][b]["impl"] == "composed"),
            "strided": sum(1 for op in advertised
                           if "strided" in op["backends"][b]["layouts"]),
            "training": sum(1 for op in advertised if op["backends"][b]["training"]),
            "dtypes": sorted({d for op in advertised for d in op["backends"][b]["dtypes"]}),
        })

    type_links = json.loads(
        (ROOT / "docs/api-type-links.json").read_text(encoding="utf-8")
    )
    types = collect_types(type_links)
    types_out = ROOT / "docs/api-types.json"
    types_out.write_text(json.dumps({"types": types}, separators=(",", ":")), encoding="utf-8")

    # Real usage for as much of the surface as the compiled sources reach.
    # Only the snippets the index actually points at are shipped.
    pool = collect_snippets()
    names = set()
    for op in operations:
        names.add(op["name"])
        api = (op.get("catalog") or {}).get("api") or ""
        if api.startswith("::"):
            names.add(api[2:])
        elif "::" in api:
            names.add(api.split("::", 1)[1])
    names.update(i["name"] for i in layout_out["items"])
    names.update(i["name"] for group in shape_groups for i in group["items"])
    names.update(m["name"] for m in target_out)
    names.update(t["name"] for t in types)
    names.update(d["id"] for d in dtypes_out)
    usage = index_usage(pool, sorted(n for n in names if n))
    kept = sorted({i for hits in usage.values() for i in hits})
    remap = {old_index: new_index for new_index, old_index in enumerate(kept)}
    usage_out = ROOT / "docs/api-usage.json"
    usage_out.write_text(json.dumps({
        "snippets": [pool[i] for i in kept],
        "index": {name: [remap[i] for i in hits] for name, hits in usage.items()},
    }, separators=(",", ":")), encoding="utf-8")

    payload = {
        "operations": operations,
        "categories": categories,
        "surface": surface,
        "backends": BACKENDS,
        "dtypes": dtypes_out,
        "backendDetail": backends_out,
        "flow": DISPATCH_FLOW,
        "shapes": shape_groups,
        "layouts": layout_out,
        "targetApi": target_out,
        "examples": collect_examples(),
        "typeCount": len(types),
        "typeLinked": sum(1 for t in types if t["url"]),
        "usageNames": len(usage),
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
