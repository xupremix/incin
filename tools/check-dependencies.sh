#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:---report}

case "$mode" in
  --report|--check) ;;
  *)
    echo "usage: $0 [--report|--check]" >&2
    exit 2
    ;;
esac

python3 - "$repo_root" "$mode" <<'PY'
import collections
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]) / "crates" / "incin-core" / "src"
mode = sys.argv[2]

# These are the first structural edges HND-002 makes non-negotiable. The
# report remains useful while the ownership moves are being implemented; the
# check becomes a hard gate once all listed edges have been removed.
forbidden = {
    ("shapes", "tensor"),
    ("shapes", "io"),
    ("shapes", "exec"),
    ("tensor", "optim"),
    ("tensor", "distributions"),
    ("tensor", "graph"),
}

edges = collections.Counter()
locations = collections.defaultdict(list)
pattern = re.compile(r"\bcrate::([A-Za-z_][A-Za-z0-9_]*)")

for path in sorted(root.rglob("*.rs")):
    rel = path.relative_to(root)
    source = rel.parts[0] if len(rel.parts) > 1 else rel.stem
    if source == "lib":
        source = "crate"
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        for match in pattern.finditer(line):
            target = match.group(1)
            if target in {"crate", source}:
                continue
            key = (source, target)
            edges[key] += 1
            if key in forbidden and len(locations[key]) < 5:
                locations[key].append(f"{rel}:{line_no}")

print("Incin core dependency edges (crate::<module> references)")
for (source, target), count in sorted(edges.items()):
    marker = " FORBIDDEN" if (source, target) in forbidden else ""
    print(f"{source:16} -> {target:16} {count:4}{marker}")

violations = [(key, count) for key, count in sorted(edges.items()) if key in forbidden]
if violations:
    print("\nForbidden edge locations:")
    for key, _ in violations:
        print(f"{key[0]} -> {key[1]}: {', '.join(locations[key])}")

if mode == "--check" and violations:
    print("\ndependency check failed", file=sys.stderr)
    sys.exit(1)
print("\ndependency check passed" if mode == "--check" else "\ndependency report complete")
PY
