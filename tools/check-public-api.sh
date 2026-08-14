#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# The ordinary facade prelude is the stable user tier. Backend operation
# families remain available only through the explicit authoring surface while
# descriptor execution is the canonical public backend contract.
if rg -n 'pub use incin_core::prelude::[^;]*(FloatOps|NumericOps|TensorOps|CreationOps|ReductionOps|ModuleOps|LossOps|QuantizedOps|OptimizerOps)' \
    crates/incin/src/lib.rs; then
    echo "public API check failed: legacy operation family leaked into facade prelude" >&2
    exit 1
fi

for capability in HostInterop VariableBackend AutogradBackend TransferBackend; do
    if ! rg -q "\\b${capability}\\b" crates/incin-core/src/lib.rs crates/incin/src/lib.rs; then
        echo "public API check failed: missing named capability ${capability}" >&2
        exit 1
    fi
done

echo "public API checks passed"
