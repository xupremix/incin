#!/usr/bin/env bash
set -euo pipefail

# Export and validate the repository handoff artifact.  The archive is built
# from HEAD so ignored files and an author's untracked build output cannot
# silently become part of the handoff.

if [[ $# -ne 1 ]]; then
    echo "usage: tools/export-snapshot.sh <output.zip>" >&2
    exit 2
fi

output=$1
repo_dir=$(git rev-parse --show-toplevel)
cd "$repo_dir"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "export requires a clean tracked checkout" >&2
    exit 1
fi

mkdir -p "$(dirname "$output")"
output=$(realpath -m "$output")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

prefix=incin-snapshot/
git archive --format=zip --prefix="$prefix" HEAD -o "$output"

tracked="$tmp/tracked"
archived="$tmp/archived"
git ls-files | sort >"$tracked"
unzip -Z1 "$output" \
    | sed "s#^$prefix##" \
    | awk 'NF && substr($0, length($0), 1) != "/"' \
    | sort >"$archived"
if ! diff -u "$tracked" "$archived"; then
    echo "export does not contain exactly the tracked source set" >&2
    exit 1
fi

unzip -q "$output" -d "$tmp/unpacked"
snapshot="$tmp/unpacked/$prefix"

required=(
    crates/incin-core/src/lib.rs
    crates/incin-core/src/dist/mod.rs
    crates/incin-backends/src/lib.rs
    crates/incin-backends/src/dist/mod.rs
    tools/check-architecture.sh
    tools/check-large-files.sh
    tools/check-public-api.sh
)
for path in "${required[@]}"; do
    [[ -f "$snapshot/$path" ]] || {
        echo "required handoff file is missing: $path" >&2
        exit 1
    }
done

(cd "$snapshot" && bash tools/check-architecture.sh)
(cd "$snapshot" && bash tools/check-large-files.sh)
(cd "$snapshot" && bash tools/check-public-api.sh)

if command -v cargo >/dev/null 2>&1; then
    (cd "$snapshot" && cargo check -p incin-core --no-default-features)
fi

echo "validated handoff snapshot: $output"
