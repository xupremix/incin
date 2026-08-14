#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# The ordinary facade prelude is the stable user tier. Descriptor execution is
# the canonical public backend contract; legacy adapters are implementation-only.
if rg -n 'pub use incin_core::prelude::[^;]*(FloatOps|NumericOps|TensorOps|CreationOps|ReductionOps|ModuleOps|LossOps|QuantizedOps|OptimizerOps)' \
    crates/incin/src/lib.rs; then
    echo "public API check failed: legacy operation family leaked into facade prelude" >&2
    exit 1
fi

if rg -n 'pub use crate::tensor::backend::[^;]*(FloatOps|NumericOps|TensorOps|CreationOps|ReductionOps|ModuleOps|LossOps|QuantizedOps|OptimizerOps)' \
    crates/incin-core/src/lib.rs; then
    echo "public API check failed: legacy operation family leaked into core exports" >&2
    exit 1
fi

if rg -n 'backend_authoring::legacy' crates/incin-core/src/lib.rs crates/incin/src/lib.rs; then
    echo "public API check failed: legacy operation family remains in the authoring surface" >&2
    exit 1
fi

# Loss and optimizer compatibility helpers are backend-local. Formatting is a
# host-interoperability concern, not part of the backend identity contract.
if rg -n '^pub trait (LossOps|OptimizerOps)|fn format_tensor_(display|debug)' \
    crates/incin-core/src/tensor crates/incin-core/src/lib.rs --glob '*.rs'; then
    echo "public API check failed: removed transitional backend surface reappeared" >&2
    exit 1
fi

if ! rg -q '^pub\(crate\) mod backend;' crates/incin-core/src/tensor/mod.rs; then
    echo "public API check failed: tensor backend module is public" >&2
    exit 1
fi

if rg -n 'incin_core::tensor::backend' crates/incin-backends crates/incin --glob '*.rs'; then
    echo "public API check failed: external crate uses private tensor backend path" >&2
    exit 1
fi

for capability in HostInterop VariableBackend AutogradBackend TransferBackend; do
    if ! rg -q "\\b${capability}\\b" crates/incin-core/src/lib.rs crates/incin/src/lib.rs; then
        echo "public API check failed: missing named capability ${capability}" >&2
        exit 1
    fi
done

echo "public API checks passed"
