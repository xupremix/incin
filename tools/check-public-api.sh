#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python3 tools/check-public-api-baseline.py

# Stable facades must name every exported item.  Internal module aggregation
# may still use local globs, but a wildcard in either consumer-facing facade
# would silently expand the supported API.
if grep -nE '^[[:space:]]*pub use .*::\*' crates/incin/src/lib.rs crates/incin-core/src/lib.rs; then
    echo "public API check failed: wildcard export leaked into a stable facade" >&2
    exit 1
fi

# The ordinary facade prelude is the stable user tier. Descriptor execution is
# the canonical public backend contract; legacy adapters are implementation-only.
if grep -nE 'pub use incin_core::prelude::[^;]*(FloatOps|NumericOps|TensorOps|CreationOps|ReductionOps|ModuleOps|LossOps|QuantizedOps|OptimizerOps)' \
    crates/incin/src/lib.rs; then
    echo "public API check failed: legacy operation family leaked into facade prelude" >&2
    exit 1
fi

if grep -nE 'pub use crate::tensor::backend::[^;]*(FloatOps|NumericOps|TensorOps|CreationOps|ReductionOps|ModuleOps|LossOps|QuantizedOps|OptimizerOps)' \
    crates/incin-core/src/lib.rs; then
    echo "public API check failed: legacy operation family leaked into core exports" >&2
    exit 1
fi

if grep -nE 'backend_authoring::legacy' crates/incin-core/src/lib.rs crates/incin/src/lib.rs; then
    echo "public API check failed: legacy operation family remains in the authoring surface" >&2
    exit 1
fi

# Loss and optimizer compatibility helpers are backend-local. Formatting is a
# host-interoperability concern, not part of the backend identity contract.
if grep -rnE --include='*.rs' '^pub trait (LossOps|OptimizerOps)|fn format_tensor_(display|debug)' \
    crates/incin-core/src/tensor crates/incin-core/src/lib.rs; then
    echo "public API check failed: removed transitional backend surface reappeared" >&2
    exit 1
fi

if ! grep -qE '^pub\(crate\) mod backend;' crates/incin-core/src/tensor/mod.rs; then
    echo "public API check failed: tensor backend module is public" >&2
    exit 1
fi

if grep -rnE --include='*.rs' 'incin_core::tensor::backend' crates/incin-backends crates/incin; then
    echo "public API check failed: external crate uses private tensor backend path" >&2
    exit 1
fi

# Kernel iteration plumbing takes runtime dimensions and sits below the
# descriptor contract, so it is not part of the backend-authoring surface.
if ! grep -qE '^pub\(crate\) mod iteration;' crates/incin-backends/src/lib.rs; then
    echo "public API check failed: kernel iteration plumbing is public" >&2
    exit 1
fi

# Keep implementation-level shape, storage, graph, and state-staging names out
# of the ordinary facade prelude.  They remain available through named expert
# modules or macro support where appropriate.
prelude=$(sed -n '/^pub mod prelude {/,/^#\[cfg(test)\]/p' crates/incin/src/lib.rs)
for symbol in \
    Graph ConcreteStaticExtent DimCons Nil ProductDims ReplaceAt \
    StructuralConcatShape StorageBackend SupportsDType StorageEncoding \
    StateDict StateLoadPlan ShapeInfo ComputeStats LayerNode NamedLayers \
    ParameterVisitor VisitParameters StateVisitor StateMutVisitor VisitState VisitStateMut \
    VariableBackend tracing_mark_input extract_graph; do
    if printf '%s\n' "$prelude" | grep -qE "\\b${symbol}\\b"; then
        echo "public API check failed: ${symbol} leaked into the ordinary facade prelude" >&2
        exit 1
    fi
done

for capability in HostInterop VariableBackend AutogradBackend TransferBackend; do
    if ! grep -qE "\\b${capability}\\b" crates/incin-core/src/lib.rs crates/incin/src/lib.rs; then
        echo "public API check failed: missing named capability ${capability}" >&2
        exit 1
    fi
done

echo "public API checks passed"
