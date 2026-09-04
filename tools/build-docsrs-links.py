#!/usr/bin/env python3
"""Resolve a docs.rs target for every catalogued operation, by checking.

Operation names in the catalog are not method names. `cmp_eq` is reached
through `eq`, the axis reductions through `sum`/`mean`/`max`/`min`, and the
owner-qualified entries (`Loss::mse_loss`, `Adam::step`) live on different
pages. Guessing that mapping from the name is what produced a wrong
test-coverage metric earlier in this project, so nothing is guessed here: each
candidate is checked against the anchors docs.rs actually publishes, and an
operation with no verified anchor gets a search link rather than a dead one.

Network-facing on purpose, and run by hand. The result is committed as
`docs/api-docsrs-links.json` so that `build-api-site-data.py` -- which CI runs
and drift-checks -- stays deterministic and offline.

    python3 tools/build-docsrs-links.py [--refresh]
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = ROOT / "docs/api-site-data.json"
OUTPUT = ROOT / "docs/api-docsrs-links.json"
CACHE = ROOT / "target/docsrs-cache"

TENSOR = "https://docs.rs/incin-core/latest/incin_core/tensor/base/struct.Tensor.html"
SEARCH = "https://docs.rs/incin-core/latest/incin_core/?search="
TYPE_OUTPUT = ROOT / "docs/api-type-links.json"

# Every published crate, and the module name docs.rs serves it under.
CRATES = {
    "incin": "incin",
    "incin-core": "incin_core",
    "incin-backends": "incin_backends",
    "incin-data": "incin_data",
    "incin-telemetry": "incin_telemetry",
    "incin-viz": "incin_viz",
    "incin-viz-plugin-api": "incin_viz_plugin_api",
    "incin-diagnostics": "incin_diagnostics",
    "incin-lsp": "incin_lsp",
    "incin-macros": "incin_macros",
}
ITEM = re.compile(
    r"^(.*?)(?:^|/)?(struct|trait|enum|type|fn|macro|constant)\.([A-Za-z0-9_]+)\.html$"
)

# Pages that own the methods the catalog names. `Tensor` is the implementation
# surface; the rest are the layers and optimizers that own a `forward`/`step`.
PAGES = {
    "Tensor": TENSOR,
    "Adam": "https://docs.rs/incin/latest/incin/optim/struct.Adam.html",
    "AdamW": "https://docs.rs/incin/latest/incin/optim/struct.AdamW.html",
    "SGD": "https://docs.rs/incin/latest/incin/optim/struct.SGD.html",
    "Dropout": "https://docs.rs/incin/latest/incin/nn/struct.Dropout.html",
    "LSTM": "https://docs.rs/incin/latest/incin/nn/struct.LSTM.html",
    "Linear": "https://docs.rs/incin/latest/incin/nn/struct.Linear.html",
    "RMSNorm": "https://docs.rs/incin/latest/incin/nn/struct.RMSNorm.html",
    "RNN": "https://docs.rs/incin/latest/incin/nn/struct.RNN.html",
}


def fetch(url: str, refresh: bool) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", url)[:120] + ".html")
    if key.exists() and not refresh:
        return key.read_text(encoding="utf-8", errors="replace")
    request = urllib.request.Request(url, headers={"User-Agent": "incin-docs-linker"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed hosts
        body = response.read().decode("utf-8", errors="replace")
    key.write_text(body, encoding="utf-8")
    return body


def anchors(body: str) -> set[str]:
    return set(re.findall(r'id="method\.([A-Za-z0-9_]+)"', body))


def candidates(name: str, api: str | None):
    """Every place this operation could reasonably be documented, best first."""
    if api and api.startswith("::"):
        yield "Tensor", api[2:]
    elif api and "::" in api:
        owner, _, method = api.partition("::")
        if method:
            if owner in PAGES:
                yield owner, method
            yield "Tensor", method
    yield "Tensor", name
    # Structural forms: the axis and keepdim variants share one method, the
    # comparison operators drop their `cmp_` prefix, and the view/exact
    # spellings document the base operation.
    for suffix in ("_dim", "_keepdim", "_all", "_view", "_exact"):
        if name.endswith(suffix):
            yield "Tensor", name[: -len(suffix)]
    if name.startswith("cmp_"):
        yield "Tensor", name[4:]


def type_index(refresh: bool) -> dict:
    """Index every item docs.rs actually publishes, per crate.

    `all.html` lists the crate's whole public surface in one page, so the whole
    index costs one request per crate instead of a probe per item. Items are
    recorded under both their published module path and their bare name: the
    committed API baselines record an item's canonical path, which is often not
    the path rustdoc documents it at once re-exports are followed, so the path
    match is tried first and the name used only when it is unambiguous.
    """
    index = {}
    for crate, module in CRATES.items():
        url = f"https://docs.rs/{crate}/latest/{module}/all.html"
        try:
            body = fetch(url, refresh)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            print(f"  ! {crate}: {error}")
            continue
        by_path, by_name = {}, {}
        for href in re.findall(r'href="([^"]+)"', body):
            match = ITEM.match(href)
            if not match:
                continue
            mod, kind, name = match.group(1).strip("/"), match.group(2), match.group(3)
            by_path[f"{kind}|{mod}|{name}"] = href
            by_name.setdefault(f"{kind}|{name}", []).append(href)
        index[crate] = {
            "base": f"https://docs.rs/{crate}/latest/{module}/",
            "by_path": by_path,
            "by_name": {k: v[0] for k, v in by_name.items() if len(v) == 1},
            "ambiguous": sorted(k for k, v in by_name.items() if len(v) > 1),
        }
        print(f"  {crate}: {len(by_path)} published items")
    return index


def main() -> int:
    refresh = "--refresh" in sys.argv
    payload = json.loads(PAYLOAD.read_text())
    published = {page: anchors(fetch(url, refresh)) for page, url in PAGES.items()}

    links, verified, searched = {}, 0, []
    for op in payload["operations"]:
        name = op["name"]
        api = (op.get("catalog") or {}).get("api")
        hit = None
        for page, method in candidates(name, api):
            if method in published.get(page, ()):
                hit = (page, method)
                break
        if hit:
            page, method = hit
            links[name] = {
                "url": f"{PAGES[page]}#method.{method}",
                "kind": "method",
                "item": f"{page}::{method}",
            }
            verified += 1
        else:
            links[name] = {"url": SEARCH + name, "kind": "search", "item": name}
            searched.append(name)

    OUTPUT.write_text(json.dumps(
        {
            "note": "Generated by tools/build-docsrs-links.py; every 'method' "
                    "entry was checked against the anchors docs.rs publishes.",
            "links": dict(sorted(links.items())),
        },
        indent=1,
    ) + "\n", encoding="utf-8")

    print(f"resolved {len(links)} operations: {verified} to a published item, "
          f"{len(searched)} to a search")

    print("indexing the published type surface")
    TYPE_OUTPUT.write_text(json.dumps(
        {
            "note": "Generated by tools/build-docsrs-links.py. Every entry is an "
                    "item docs.rs publishes; consumed offline by "
                    "build-api-site-data.py to link the type reference.",
            "crates": type_index(refresh),
        },
        indent=1,
    ) + "\n", encoding="utf-8")
    print(f"wrote {TYPE_OUTPUT.relative_to(ROOT)}")
    if searched:
        print("  search fallback: " + ", ".join(sorted(searched)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
