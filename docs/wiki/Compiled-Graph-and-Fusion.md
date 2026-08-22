# Compiled Graph Execution & Kernel Fusion

Incin includes an experimental graph compiler under `incin::experimental::compiled` (available with the `--features compiled` flag).

---

## 1. Graph Compilation Architecture

Graph execution transforms declarative tensor computation into an optimized linear execution plan:

1. **Tracing**: Records node computations into an intermediate representation (`GraphIR`).
2. **Constant Folding**: Evaluates constant tensor operations (such as pre-computed positional embeddings or fixed weights) at compile time.
3. **Dead Code Elimination (DCE)**: Prunes unused intermediate graph branches.
4. **Kernel Fusion**: Combines sequential elementwise kernels (e.g. `Conv2d + Bias + ReLU`) into a single fused compute kernel, eliminating redundant memory roundtrips between DRAM and registers.

---

## 2. Compiling and Executing a Plan

```rust
use incin::prelude::*;
use incin::experimental::compiled::{CompiledModel, CompileOptions};

// 1. Define model
let model = MyConvNet::new(&Cpu)?;

// 2. Trace and compile with optimization passes
let options = CompileOptions::default().with_fusion(true);
let compiled = CompiledModel::compile(&model, options)?;

// 3. Fast execution loop
for batch in dataloader {
    let output = compiled.execute(&batch)?;
}
```

---

## 3. Plan Inspection & Telemetry

Use the CLI to inspect and profile execution plans:

```bash
cargo incin plan --json
```

This outputs a breakdown of fused vs unfused operators, estimated memory bandwidth, and kernel launch latency.
