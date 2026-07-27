#!/usr/bin/env bash
# Shape/dtype/backend/device proof-stage audit (SHP-001).
#
#   tools/audit-shapes.sh            print the current inventory
#   tools/audit-shapes.sh --check    fail if docs/audit/shape-proof-inventory.md is stale
#   tools/audit-shapes.sh --update   rewrite the generated block in that document
#
# The document is the human-readable half of this audit; the generated block is
# the machine-readable half. Edit both, or neither -- `--check` is what keeps the
# two from drifting, the same round-trip contract `cargo xtask ledger` enforces
# for docs/plan/ledger.toml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/docs/audit/shape-proof-inventory.md"
BEGIN='<!-- BEGIN GENERATED: audit-shapes -->'
END='<!-- END GENERATED: audit-shapes -->'

MODE="${1:-report}"
case "$MODE" in
  report | --report) MODE=report ;;
  --check) MODE=check ;;
  --update) MODE=update ;;
  -h | --help)
    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit 0
    ;;
  *)
    echo "audit-shapes: unknown argument '$MODE' (want --check, --update, or no argument)" >&2
    exit 2
    ;;
esac

report() {
  python3 - "$ROOT" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])

# Source roots that carry shape, dtype, backend, or device *rules*. Test files
# and generated bindings are excluded: this audit measures the obligations the
# library imposes on its callers, not the ones its tests impose on themselves.
GROUPS = {
    "shapes":  ["crates/incin-core/src/shapes"],
    "tensor":  ["crates/incin-core/src/tensor"],
    "backend": ["crates/incin-backends/src/dispatch.rs",
                "crates/incin-backends/src/dtype_policy.rs"],
}
EXCLUDE_NAMES = {"onnx_pb.rs"}

# Panic-class categories. Each is a way a shape rule can fail at runtime that
# the proof-carrying design in PROPOSALS.md sec. 1.2 intends to make either
# impossible or an explicit `Result`.
CATEGORIES = [
    ("unwrap",      re.compile(r"\.unwrap\(\)")),
    ("expect",      re.compile(r"\.expect\(")),
    ("panic",       re.compile(r"\b(panic!|unreachable!|todo!|unimplemented!)")),
    ("assert",      re.compile(r"\bassert(_eq|_ne)?!")),
]

# Named chains that specific ledger tasks are required to drive to zero.
#
# The call-plus-`.unwrap()` chains are counted by balancing parentheses, not by
# regex. `from_dyn(&broadcast_dims::<Self, (A, B)>(lhs, rhs)).unwrap()` nests
# two levels of parentheses inside the argument, and a `\([^)]*\)` pattern stops
# at the first inner `)` and misses the site entirely -- which hid 14 of the 42
# `from_dyn` sites from the SHP-001 baseline.
CHAINS = [
    ("from_dyn().unwrap()",  "SHP-004", "from_dyn"),
    ("from_size().unwrap()", "SHP-005", "from_size"),
    ("Default::default()",   "SHP-005", None),
]
DEFAULT_RX = re.compile(r"Default::default\(\)")


def count_unwrapped_calls(code, name):
    """Count `name( ... ).unwrap()` occurrences in `code`, nesting and all."""
    total, start = 0, 0
    while True:
        at = code.find(name + "(", start)
        if at < 0:
            return total
        depth, i = 0, at + len(name)
        while i < len(code):
            depth += (code[i] == "(") - (code[i] == ")")
            if depth == 0:
                break
            i += 1
        if re.match(r"\s*\.unwrap\(\)", code[i + 1:]):
            total += 1
        start = at + len(name) + 1
CHAIN_SCOPE = {
    "from_dyn().unwrap()":  ["crates/incin-core/src"],
    "from_size().unwrap()": ["crates/incin-core/src/shapes"],
    "Default::default()":   ["crates/incin-core/src/shapes/spatial.rs"],
}


def rs_files(specs):
    out = []
    for spec in specs:
        p = root / spec
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(q for q in p.rglob("*.rs")
                              if q.name not in EXCLUDE_NAMES))
    return out


def live_lines(path):
    """Yield (lineno, code) for lines outside #[cfg(test)] modules and comments."""
    lines = path.read_text().splitlines()
    masked = set()
    i = 0
    while i < len(lines):
        if re.search(r"#\[cfg\(test\)\]", lines[i]):
            depth, started, j = 0, False, i
            while j < len(lines):
                depth += lines[j].count("{") - lines[j].count("}")
                if "{" in lines[j]:
                    started = True
                masked.add(j)
                if started and depth <= 0:
                    break
                j += 1
            i = j + 1
            continue
        i += 1
    for n, line in enumerate(lines):
        if n in masked:
            continue
        code = line.split("//")[0]
        if code.strip():
            yield n + 1, code


def panic_counts():
    rows = []
    for group, specs in GROUPS.items():
        tally = {name: 0 for name, _ in CATEGORIES}
        for path in rs_files(specs):
            for _, code in live_lines(path):
                for name, rx in CATEGORIES:
                    if rx.search(code):
                        tally[name] += 1
        rows.append((group, tally))
    return rows


def chain_counts():
    rows = []
    for label, owner, call in CHAINS:
        n = 0
        for path in rs_files(CHAIN_SCOPE[label]):
            for _, code in live_lines(path):
                n += (len(DEFAULT_RX.findall(code)) if call is None
                      else count_unwrapped_calls(code, call))
        rows.append((label, owner, n))
    return rows


# An impl counts toward a rank ceiling only when its target is a tuple of macro
# parameters. `impl_shape_for_tuple!` also emits `impl EndsWith<usize> for
# [usize; N]`; those fixed-size-array impls belong to the fully dynamic shape
# family and would otherwise credit a typed tuple rule with a rank it does not
# implement. Requiring `for (` excludes them, and requiring a `$` in the target
# excludes concrete tuples that the second, non-macro scan already counts.
TUPLE_IMPL_HEAD = re.compile(
    r"\bimpl\s*(?:<[^>]*>)?\s*([A-Z][A-Za-z0-9_]*)\s*(?:<[^>]*>)?\s*for\s*\(")


def tuple_impls(flat):
    """Yield (trait, target) for each `impl Trait for ( ... )` in `flat`.

    The target is read by balancing parentheses rather than by regex. A
    repetition like `( $($name,)* )` contains a nested `)`, so a
    `\\(([^)]*)\\)` pattern truncates it to `$($name,` -- which then reads as
    one fixed element and inflates every variadic rule's rank by one.
    """
    for m in TUPLE_IMPL_HEAD.finditer(flat):
        depth, start = 0, m.end() - 1
        for i in range(start, len(flat)):
            depth += (flat[i] == "(") - (flat[i] == ")")
            if depth == 0:
                target = flat[start + 1:i]
                if "$" in target:
                    yield m.group(1), target
                break


def macro_traits(text):
    """Map each `impl_*` macro to the tuple rules it implements.

    Two macro styles appear in this tree and they set their ceiling
    differently. A *variadic* macro expands one `( $($name,)* )` impl, so its
    ceiling is the largest arity it is invoked at. An *enumerated* macro spells
    out one impl per arity (`($n1, $last)`, `($n1, $n2, $last)`, ...), so its
    ceiling is the widest tuple in its own body regardless of invocation.
    """
    out = {}
    for m in re.finditer(r"macro_rules!\s+(impl_[a-z0-9_]+)\s*\{", text):
        name, start, depth, body = m.group(1), m.end() - 1, 0, ""
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                body = text[start:i + 1]
                break
        flat = re.sub(r"\s+", " ", body)
        traits = {}
        for trait, target in tuple_impls(flat):
            variadic = "$(" in target
            if variadic:
                # A variadic target may also carry fixed trailing elements:
                # `impl_conv2d_shape!` expands `for ( $($B,)* CIn, HIn, WIn )`,
                # so its rank is the batch count *plus three*. Counting only
                # the repetition would understate every conv and pool rule.
                # Both `$(..)*` and `$(..)+` appear in this tree; missing the
                # `+` form leaves the repetition in place, where its own comma
                # then reads as two fixed elements.
                # A fixed element counts toward the rank only when it comes
                # from *outside* the swept name list. `impl_conv2d_shape!`
                # expands `( $($B,)* CIn, HIn, WIn )`, where the three trailing
                # axes are plain identifiers the macro supplies itself -- real
                # extra axes. But `impl_concat_shape!` expands
                # `( $($pre,)* $ax, $($post,)* )`, where `$ax` is one of the N
                # names the sweep passed in and is already inside the count.
                # The `$` sigil is exactly that distinction.
                fixed = len(
                    [a for a in re.sub(r"\$\([^)]*\)\s*[*+?]", "", target).split(",")
                     if a.strip() and not a.strip().startswith("$")])
                arity = 0
            else:
                fixed, arity = 0, len([a for a in target.split(",") if a.strip()])
            was_variadic, prev_arity, prev_fixed = traits.get(trait, (False, 0, 0))
            traits[trait] = (was_variadic or variadic,
                             max(prev_arity, arity),
                             max(prev_fixed, fixed))
        if traits:
            out[name] = traits
    return out


MAX_RANK_RX = re.compile(r"pub\(crate\)\s+const\s+MAX_RANK:\s*usize\s*=\s*(\d+)")
SWEEP_RX = re.compile(
    r"rank_sweep!\(\s*\w+\s*=>\s*(impl_[a-z0-9_]+)((?:\s*,\s*\w+\s*=\s*\d+)*)\s*\)")


def max_rank():
    """The single ceiling `SHP-006` put in `incin-macros/src/rank.rs`."""
    source = (root / "crates/incin-macros/src/rank.rs")
    if not source.is_file():
        return None
    found = MAX_RANK_RX.search(source.read_text())
    return int(found.group(1)) if found else None


def rank_ceilings():
    """Highest tuple rank each shape rule is actually implemented at.

    `Shape` reaches rank 8. Every rule below that ceiling names a rank at which
    a tensor type is expressible but the rule cannot resolve -- the typed
    frontend accepts the shape and then has no proof to offer for the operation.
    Counts merge macro-generated impls with hand-written tuple impls.
    """
    out = {}

    def bump(trait, arity):
        out[trait] = max(out.get(trait, 0), arity)

    ceiling = max_rank()

    for path in sorted((root / "crates/incin-core/src/shapes").glob("*.rs")):
        text = path.read_text()
        m2t = macro_traits(text)

        # A ladder generated by `rank_sweep!` has no literal `impl_x!(N)` lines
        # to count. Its rank is `MAX_RANK`, or the `max =` the sweep declares --
        # which is a real ceiling, not a gap: a rule whose `Output` gains an
        # axis (`AppendDim`, `StackShape`) must stop one short, or the tuple it
        # produces has no `Shape` impl.
        for name, options in SWEEP_RX.findall(text):
            declared = dict(
                (k.strip(), int(v))
                for k, v in (opt.split("=") for opt in options.split(",") if opt.strip())
            )
            arity = declared.get("max", ceiling)
            if arity is None:
                continue
            for trait, (variadic, body_max, fixed) in m2t.get(name, {}).items():
                bump(trait, arity + fixed if variadic else min(body_max, arity))
        for m in re.finditer(r"^\s*(impl_[a-z0-9_]+)!\(([^)]*)\);", text, re.M):
            name, args = m.group(1), m.group(2)
            # A `;` separates fixed macro arguments from the repeated list --
            # `impl_conv2d_shape!(5, 6; B0: 0, ...)` passes the two spatial
            # tuple indices up front. Only the repeated tail sets the rank.
            args = args.rsplit(";", 1)[-1]
            groups = [a for a in args.split(",") if a.strip()]
            # `impl_shape_for_tuple!(8, D0 0, ...)` leads with the rank itself.
            arity = int(groups[0]) if groups and groups[0].strip().isdigit() else len(groups)
            for trait, (variadic, body_max, fixed) in m2t.get(name, {}).items():
                bump(trait, arity + fixed if variadic else body_max)
        for m in re.finditer(
                r"^impl\s*(?:<[^>]*>)?\s*([A-Z][A-Za-z0-9_]*)\s*(?:<[^>]*>)?\s*for\s*\(([^)]*)\)",
                text, re.M):
            bump(m.group(1), len([a for a in m.group(2).split(",") if a.strip()]))
    return out


w = sys.stdout.write
w("### Panic-class sites by rule surface\n\n")
w("| Rule surface | `unwrap` | `expect` | `panic!`-class | `assert!` |\n")
w("|---|---:|---:|---:|---:|\n")
for group, t in panic_counts():
    w(f"| `{group}` | {t['unwrap']} | {t['expect']} | {t['panic']} | {t['assert']} |\n")

w("\n### Named chains with a required terminal count of zero\n\n")
w("| Chain | Owner | Sites |\n|---|---|---:|\n")
for label, owner, n in chain_counts():
    w(f"| `{label}` | {owner} | {n} |\n")

w("\n### Rank ceiling by shape rule\n\n")
w("| Rule | Max rank | vs `Shape` |\n|---|---:|---|\n")
ceilings = rank_ceilings()
ceiling = ceilings["Shape"]
for name, arity in sorted(ceilings.items(), key=lambda kv: (-kv[1], kv[0])):
    d = arity - ceiling
    verdict = "aligned" if d == 0 else (f"{d} over" if d > 0 else f"{-d} short")
    w(f"| `{name}` | {arity} | {verdict} |\n")
PY
}

generated="$(report)"

case "$MODE" in
report)
  printf '%s\n' "$generated"
  ;;
check)
  if [[ ! -f $DOC ]]; then
    echo "audit-shapes: missing $DOC" >&2
    exit 1
  fi
  embedded="$(awk -v b="$BEGIN" -v e="$END" \
    'index($0,b){f=1;next} index($0,e){f=0} f' "$DOC" | sed '/^$/d')"
  if [[ -z $embedded ]]; then
    echo "audit-shapes: no generated block found in $DOC" >&2
    exit 1
  fi
  if diff -u <(printf '%s\n' "$embedded") <(printf '%s\n' "$generated" | sed '/^$/d') > /tmp/audit-shapes.diff; then
    echo "audit-shapes ok: inventory matches $(basename "$DOC")"
  else
    echo "audit-shapes: inventory is stale -- rerun 'tools/audit-shapes.sh --update'" >&2
    echo "  (-) recorded in the document, (+) measured in the tree" >&2
    cat /tmp/audit-shapes.diff >&2
    exit 1
  fi
  ;;
update)
  # The body travels via the environment, not heredoc interpolation: it is
  # markdown full of backticks and quotes, and inlining it into the script text
  # would expose it to a second round of shell and Python parsing.
  AUDIT_BODY="$generated" python3 - "$DOC" "$BEGIN" "$END" <<'PY'
import os, pathlib, sys
doc, begin, end = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
body = os.environ["AUDIT_BODY"].strip()
text = doc.read_text()
head, sep, rest = text.partition(begin)
if not sep:
    sys.exit(f"audit-shapes: no '{begin}' marker in {doc}")
_, sep, tail = rest.partition(end)
if not sep:
    sys.exit(f"audit-shapes: no '{end}' marker in {doc}")
doc.write_text(f"{head}{begin}\n\n{body}\n\n{end}{tail}")
print("audit-shapes: updated", doc.name)
PY
  ;;
esac
