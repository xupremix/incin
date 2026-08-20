#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# Every source file above the handoff threshold must have a named reason in
# docs/HANDOFF.md. This is an ownership ledger, not a claim that each file is
# already ideally split; staged extraction targets are allowed when documented.
declare -A explained=(
    [crates/incin-core/src/exec/catalog/tests.rs]=1
    [crates/incin-backends/src/dist/nccl.rs]=1
    [crates/incin-core/src/tensor/ops/manipulation.rs]=1
    [crates/incin-backends/src/kernel.rs]=1
    [crates/incin-diagnostics/src/lib.rs]=1
    [crates/incin-core/src/generated/onnx.rs]=1
    [crates/incin-macros/src/generated/onnx.rs]=1
    [crates/incin-backends/src/dist/tuning.rs]=1
    [crates/incin-backends/src/tuning/identity.rs]=1
    [crates/incin-backends/src/cpu/ops/conv.rs]=1
    [crates/incin-backends/src/tuning/service.rs]=1
    [crates/incin-core/src/optim/mod.rs]=1
)

mapfile -t actual < <(
    find crates -path '*/src/*' -name '*.rs' -print0 |
        xargs -0 wc -l |
        awk '$2 != "total" && $1 > 1200 { print $2 }' |
        sort
)

failures=0
for path in "${actual[@]}"; do
    if [[ -z "${explained[$path]+yes}" ]]; then
        echo "large-file check failed: $path is over 1200 lines but not inventoried" >&2
        failures=$((failures + 1))
    fi
done

for path in "${!explained[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "large-file check failed: inventoried file is missing: $path" >&2
        failures=$((failures + 1))
    elif (( $(wc -l < "$path") <= 1200 )); then
        echo "large-file check failed: $path is inventoried but no longer exceeds 1200 lines" >&2
        failures=$((failures + 1))
    fi
done

if (( failures != 0 )); then
    exit 1
fi
echo "large-file inventory checks passed (${#actual[@]} files)"
