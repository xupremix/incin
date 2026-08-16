# Incin Performance Roadmap

## Where we are

### Completion state (as of 2026-07-30)

| Theme | Done | Active | Planned | Total |
|-------|:----:|:------:|:-------:|:-----:|
| `gov` — governance & baselines | 7 | 0 | 0 | 7 |
| `exec` — execution contract | 8 | 1 | 0 | 9 |  
| `shp` — shape & dim system | 6 | 0 | 2 | 8 |
| `grd` — autograd | 5 | 0 | 2 | 7 |
| `dist` — distributed | 5 | 5 | 6 | 16 |
| `tune` — autotune | 3 | 0 | 6 | 9 |
| `perf` — kernel performance | 2 | 0 | 2 | 4 |
| `ux` — user experience | 2 | 0 | 13 | 15 |
| `ci` — CI & gates | 3 | 0 | 5 | 8 |
| `compile` — compile-time | 0 | 0 | 6 | 6 |
| `metal` — Metal backend | 0 | 0 | 6 | 6 |
| `rel` — release | 0 | 0 | 4 | 4 |
| **Total** | **47** | **7** | **46** | **100** |

### What's already solid ✅

- **Compile-time shape verification** — `s![]`, `dim!`, `idx![]`, typenum
  arithmetic, compile-fail test suite. Ahead of every other Rust ML framework.
- **Typed multi-backend dispatch** — `Backend` trait, `DispatchBackend<T,D>`,
  CPU/CUDA/WGPU/Candle bridges, typed `Tensor<S,B,K,G>`.
- **Autograd** — reverse-mode tape, `GRD-001..005` complete, grad parity tests.
- **Autotune infrastructure** — UUID/topology identities, atomic persistent cache,
  profile-guided + coordinated-warmup services (`TUN-000..003`). The backend is
  done; the call-site API isn't exposed yet (→ TUN-004).
- **Distributed** — typed meshes, physical binding, placement proofs, NCCL
  transport, 5/16 tasks complete and 5 active.
- **Governance** — Criterion baselines, budget gates, ledger validator, decision
  log.

### What's missing or weak ⚠️

- **CPU kernel performance** — ~9 000 lines of op code with no SIMD: reduce,
  norm, pool, conv, and all non-f32 elementwise. Every unary op widens to f64.
- **Compiled/fused execution** — eager only; no op fusion, no graph-level
  optimization (`EXE-009` active).
- **Metal backend** — 0/6 tasks started.
- **UX** — 13 planned tasks around ergonomics, error messages, Python bindings.
- **Higher-order grad** — `GRD-006/007` planned (Jacobian, Hessian).

---

## New tasks: Mojo-inspired performance sprint

> Each task has a **benchmark before / benchmark after** workflow built in.
> The rule: measure first, implement, measure again. If the Criterion comparison
> doesn't show the expected gain the task was wrong about something — either the
> bottleneck was somewhere else or the fix didn't land cleanly.

### Dependency chain

```
PRF-003 (simd_lanes const fn)
    └── PRF-004 (TypedKernel — eliminate f64 widening)
    └── PRF-005 (tile_2d + autotune integration)  ←── TUN-003
            └── PRF-006 (vectorize! combinator)
                        └── TUN-004 (#[autotune] macro)  ←── TUN-003
```

### Task summaries

| Task | What | Expected gain | Effort |
|------|------|---------------|--------|
| [PRF-003](tasks/PRF-003.md) | `simd_lanes<T>()` compile-time lane constant | Indirect enabler; 10–30% from eliminated branch | Low |
| [PRF-004](tasks/PRF-004.md) | `TypedKernel<T>` — kill f64 widening in unary ops | **4–8× on bf16/f16; 2–3× on f32** | Medium |
| [PRF-005](tasks/PRF-005.md) | `tile_2d` cache-blocking + autotune tile selection | **3–8× matmul/conv off BLAS path** | Low–Medium |
| [PRF-006](tasks/PRF-006.md) | `vectorize!` combinator for reduce/norm/pool/conv | **3–8× on all currently-scalar ops** | Medium |
| [TUN-004](tasks/TUN-004.md) | `#[autotune(...)]` proc-macro to expose tuning infra | **15–40% mean improvement** | Medium |

### Benchmark workflow (applies to all tasks)

1. **Extend** `crates/incin/benches/baselines.rs` with series for the ops the
   task touches (see individual task files for the exact series to add).
2. **Record before baseline:**
   ```bash
   cargo bench -p incin --bench baselines -- '<pattern>' \
     --save-baseline <task>-before
   ```
3. **Implement** the task.
4. **Record after baseline and compare:**
   ```bash
   cargo bench -p incin --bench baselines -- '<pattern>' \
     --save-baseline <task>-after
   cargo bench -p incin --bench baselines -- '<pattern>' \
     --load-baseline <task>-before --baseline <task>-after
   ```
5. **Gate:** if the comparison shows a regression anywhere, do not merge.
   If the expected gain is < 50% of what the task estimated, open a follow-up
   investigation item before closing.
6. **Update** `docs/plan/baselines/main.toml` and `docs/plan/budgets.toml`
   in the same commit as the implementation (as existing governance requires).

---

## Broader inspiration: what other frameworks and languages do that incin doesn't yet

### JAX / XLA

| JAX feature | What it gives | Incin status | Task |
|-------------|---------------|--------------|------|
| `@jax.jit` — compile a pure function to XLA | Global op fusion, constant folding, layout optimization across the whole forward pass | **Already scoped** — this is the `model.compile()` story in incin; the full chain is EXE-009 → CMP-001..006 | See chain below |
| `jax.vmap` — auto-vectorise over a batch axis | Write a single-sample function; `vmap` generates the batched version with no explicit batch loop | **Not yet scoped** — Incin requires the user to pass a batched tensor; no axis-lifting transform | New `CMP-007` |
| `jax.lax.scan` — JIT-compilable loop | Replaces Python `for` loops over time steps so the compiler sees the whole sequence | **Not yet scoped** — RNN/LSTM loop currently re-enters Rust per step | New `EXE-011` |
| `jax.grad` of `jax.grad` | Higher-order derivatives (Jacobian, Hessian) composable with `jit` and `vmap` | **Planned** — `GRD-006/007` exist in the ledger but are not yet scoped | `GRD-006` |

#### The `model.compile()` chain (already in the roadmap)

```
EXE-009  (active)  — remove monolithic adapter; clean descriptor surface
    └── CMP-001    — capture the eager graph into validated IR
        └── CMP-002 — immutable compiled plans + dynamic shape guards
            ├── CMP-003 — liveness & allocation planner (peak-memory reduction)
            └── CMP-004 — constant folding, weight prepacking, shape buckets
                └── CMP-005 — op fusion + backward hooks (← the key one)
                    └── CMP-006 — versioned compiled artifacts
```

`CMP-005` is the direct analogue of PyTorch's TorchInductor / Triton fusion
pass: *"Safe fusion and backward hooks; gradient parity and launch-count
reduction."* When it lands, `incin_model.compile()` (or the Rust equivalent)
will merge adjacent pointwise ops into a single kernel, eliminating intermediate
allocations across the entire forward pass — the **2–4× memory-traffic win**
described in the PyTorch section below.

**Biggest JAX idea not yet in incin:** `vmap`. The static shape system is
uniquely positioned to implement it correctly — if a function maps
`Tensor<s![N], B>` to `Tensor<s![M], B>`, a `vmap` transform over a new batch
axis produces `Tensor<s![Batch, N], B>` → `Tensor<s![Batch, M], B>` with the
shape proven at compile time. No other Rust framework can do this safely.

### PyTorch 2.x / TorchInductor / Triton

| Feature | What it gives | Incin gap |
|---------|---------------|-----------|
| `torch.compile` / TorchDynamo | Captures a dynamic Python computation graph and lowers it to optimised Triton kernels via Inductor | `EXE-009` is the analogue; incin has the `OperationSpec` IR but no fusion pass |
| **Op fusion** (Inductor/Triton) | Runs `relu(norm(x))` as a single kernel; eliminates all intermediate allocations | Every incin op allocates an output buffer today; fusion would be a 2–4× memory-traffic win for transformer layers |
| **FlexAttention** | Custom attention variants (sliding window, ALiBi, RoPE mask) expressed as a Python closure; the compiler generates a FlashAttention-style fused kernel | `scaled_dot_product_attention` exists; no custom mask API |
| **`torch.compile` with `max-autotune`** | Systematic tile/split search across the whole computation graph, not just individual ops | Incin autotune is per-op; a graph-level search is `TUN-008` |

**Biggest PyTorch idea for incin:** **op fusion**. The `OperationSpec` enum
already gives a description of every op. A simple fusion pass that merges
adjacent pointwise ops into a single CPU loop (or single GPU kernel) would
eliminate the intermediate allocation per op — likely a **2–4×** memory-traffic
reduction for a transformer forward pass with no change to kernel performance.

### Mojo (already documented in detail)

See the task files above for the concrete proposals.
The short list: `simd_lanes` (PRF-003), typed kernels (PRF-004), tiling
(PRF-005), `vectorize!` (PRF-006), `#[autotune]` (TUN-004).

### Swift (Swift for TensorFlow / MLIR)

| Feature | What it gives | Incin applicability |
|---------|---------------|-------------------|
| `@differentiable` function attribute | Makes any function differentiable by annotation; the compiler generates the vjp/jvp automatically | Incin's tape is manual; a macro `#[differentiable]` that instruments a fn for the tape would reduce boilerplate |
| `@_semantics("tensorflow....")` | Compiler intrinsics for op recognition and fusion | Analogous to incin's `OperationSpec`; incin is already here |

### Chapel / Zig (systems languages with ML relevance)

| Language | Relevant idea | Incin applicability |
|----------|--------------|-------------------|
| **Chapel** | `forall` loop with `reduce` intent — the compiler parallelises the loop and the reduction in one statement, no rayon | incin's `rayon::par_chunks_mut` is explicit; a higher-level `parallel_reduce!` macro would be cleaner |
| **Zig** | `comptime` — arbitrary computation at compile time including array size calculation | Rust `const fn` is catching up; the `simd_lanes` const fn (PRF-003) is the exact analogue |

---

## Recommended execution order

```
1. PRF-003  — simd_lanes (1–2 days, unlocks everything)
2. PRF-004  — TypedKernel / no f64 widening (3–4 days, highest perf per effort)
3. PRF-005  — tile_2d + autotune tile (2–3 days, highest raw throughput)
4. PRF-006  — vectorize! combinator (4–5 days, broad coverage)
5. TUN-004  — #[autotune] macro (2–3 days, exposes existing infra)
──── benchmark gate: compare all before/after baselines ────
6. EXE-009  — compiled execution foundation (then JAX/PyTorch fusion ideas)
7. CMP-007  — vmap axis-lifting transform (uniquely enabled by static shapes)
```

---

## What would make incin genuinely great

Beyond kernel performance, these are the things that would turn incin from "an
impressive Rust ML library" into "the reason I use Rust for ML."

### 1. 🧠 FlexAttention — user-defined attention masks as compiled kernels

**What:** PyTorch 2.x's killer feature: you write a Python closure that describes
your attention mask variant (sliding window, ALiBi, causal, RoPE, document mask)
and the compiler generates a fused FlashAttention-style kernel from it.

**Where incin is:** `scaled_dot_product_attention` exists in `tensor/matmul.rs`
but only supports standard causal masking. No user-defined mask API.

**Why it matters:** Every modern architecture uses a custom attention variant.
Without this, users rewrite the attention loop by hand for every model, losing
both the fusion win and the memory-efficient tiling that FlashAttention provides.

**What to build:** A `FlexAttention` trait or closure-taking API:
```rust
tensor.flex_attention(&key, &value, |q_idx, kv_idx| {
    // sliding window: only attend to last 512 tokens
    q_idx - kv_idx < 512
})
```
The closure gets compiled into a mask that's applied inside the tiled attention
kernel (CMP-005's fusion pass handles this naturally). Incin's compile-time
shapes can statically verify the mask dimensions.

---

### 2. 📦 Sharded model loading and the model hub (docs/growth/09)

**What:** Every large open-weights release (Llama 405B, DeepSeek-V3, Kimi K2)
ships as `model-00001-of-000NN.safetensors` shards. None of them load in incin
today.

**Where incin is:** `hub.rs` downloads a single `model.safetensors` file.
`import_model!` takes a single `LitStr` path. No shard index support.

**Why it matters:** This is the "can I actually use this" gate. A user who types
`model!("meta-llama/Llama-4-Scout-17B-16E", ...)` gets a compile error because
the checkpoint is sharded. Growth doc 09 fully specifies the fix across 4
workstreams.

**What to build:**
- Workstream A: read `model.safetensors.index.json`, resolve shard paths
- Workstream B: `cargo incin fetch <repo_id>` CLI and a manifest cache
- Workstream C: streaming shard reassembly into typed structs
- Workstream D: MoE router and expert gating (for frontier models)

---

### 3. ⚡ Mixed precision training with AMP-style auto-casting

**What:** PyTorch's `torch.cuda.amp.autocast` and JAX's `jax.numpy.bfloat16`
let you run forward passes in bf16/f16 while keeping master weights in f32,
with automatic loss scaling to prevent underflow.

**Where incin is:** DType is part of the tensor type (`Tensor<S, B, K, G>`).
Typed kernels work for any dtype but the *training loop* has no auto-cast or
loss-scaling machinery. The `Trainer` (UX-001) runs everything at the storage
dtype.

**Why it matters:** Training in bf16 halves memory and doubles throughput on
modern GPUs. Without AMP, users must manually manage casts everywhere.

**What to build:**
- A `MixedPrecisionPolicy { compute: bf16, master: f32, loss_scale: dynamic }`
  type that the `Trainer` consumes
- Auto-cast wrappers on `forward()` that insert dtype casts at module boundaries
- Dynamic loss scaling with inf/nan detection in the backward pass
- Static type assertions that the master weights stay f32

---

### 4. 🐍 PyO3 Python bindings — `pip install incin`

**What:** MLX (Apple), candle (Hugging Face), and burn all ship Python wheels.
The Python ML ecosystem is non-negotiable for adoption.

**Where incin is:** Pure Rust. No Python bindings exist.

**Why it matters:** 95% of ML practitioners use Python. Even if incin is better
in every technical dimension, a Rust-only API limits adoption to Rust developers.
The `Trainer`, model hub, and diagnostic tooling should all be callable from
Python.

**What to build:**
- `incin-python` crate using PyO3, wrapping `Tensor`, `Module`, `Trainer`
- `pip install incin` wheel (maturin build)
- Numpy interop: `tensor.numpy()` / `Tensor.from_numpy(arr)`
- Match the PyTorch API surface just enough that existing code ports trivially

---

### 5. 📖 The Book — `docs/growth/07-the-book.md`

**What:** Rust has *The Book*. JAX has its excellent tutorial sequence. Incin has
an API reference and a README.

**Why it matters:** The diagnostics crate (`incin-diagnostics`) already
humanises compile errors beautifully — but a new user doesn't know the shape
system, the `s![]` macro, or the Backend trait well enough to hit those errors
productively. A progressive tutorial ("your first tensor → your first model →
your first training run → your first GPU run") is the highest-leverage docs
investment.

**Where incin is:** Growth doc 07 exists and specifies "the book" but no content
has been written.

---

### 6. 🚀 Inference serving — `incin serve`

**What:** vLLM, TensorRT-LLM, and ONNX Runtime Server dominate the inference
serving space. They all have continuous batching, KV-cache management, and
streaming token generation.

**Where incin is:** No serving infrastructure. A user who trains a model in
incin must export it to ONNX and serve it with something else.

**Why it matters:** If incin can train a model but can't serve it, the user has
to leave the ecosystem at exactly the moment they need reliability most.

**What to build (incrementally):**
1. A simple `Model::generate()` API with greedy/sampling decoding
2. A KV cache that reuses allocated memory across generation steps
3. An HTTP endpoint (`cargo incin serve --model <path> --port 8080`)
4. Continuous batching for throughput

---

### 7. 🔄 `#[differentiable]` — tape instrumentation by annotation

**What:** Swift for TensorFlow's `@differentiable` attribute made any function
automatically differentiable. The compiler inserted the vjp/jvp.

**Where incin is:** The tape is manual — users call `.backward()` and the tape
records ops. This works but is boilerplate-heavy when building custom autograd
functions.

**What to build:** A `#[differentiable]` proc-macro in `incin-macros` that:
- Instruments a `fn(&Tensor<..>) -> Tensor<..>` to record on the tape
- Generates the backward closure automatically from the function body
- Errors at compile time if the function uses non-differentiable operations

---

### 8. 📊 Memory profiler — `cargo incin profile`

**What:** PyTorch's `torch.cuda.memory_summary()` and JAX's `jax.profiler` show
exactly where memory goes: which tensors are alive, which are leaked, where peak
usage occurs.

**Where incin is:** `incin-viz` has a `MemoryPanel` and `incin-telemetry` has
events, but there's no one-command memory profiler.

**What to build:**
- A `cargo incin profile <script>` command that runs a training step and dumps:
  - Peak memory by backend (CPU / CUDA / WGPU)
  - Tensor allocation timeline (live tensors vs. time)
  - Top-N tensors by size at peak
- Integrate with the `incin-viz` panels for graphical display

---

### 9. 🏆 End-to-end showcase: train and serve a real model

**What:** The ultimate proof that a framework works is a complete example: load
a model, fine-tune it on a dataset, export it, serve it.

**Where incin is:** `test_models/` exists but contains only test fixtures. No
end-to-end example that a user can copy and run.

**What to build:**
- `examples/train_gpt2.rs` — fine-tune GPT-2 on a small dataset
- `examples/inference_llama.rs` — load Llama weights, generate text
- `examples/train_cifar.rs` — image classification from scratch
- Each example runs in under 5 minutes on a consumer GPU and produces a
  checkpoint that can be loaded back

---

### Priority matrix

| # | Idea | Impact on adoption | Technical difficulty | Depends on |
|---|------|--------------------|---------------------|------------|
| 1 | Sharded model loading | 🔴 Critical — blocks all frontier models | Medium | Growth doc 09 |
| 2 | End-to-end examples | 🔴 Critical — first thing a new user looks for | Low | Sharded loading |
| 3 | The Book | 🟠 High — retains users after first contact | Low | Nothing |
| 4 | Mixed precision / AMP | 🟠 High — halves memory, doubles GPU throughput | Medium | PRF-004 |
| 5 | FlexAttention | 🟠 High — unlocks custom architectures | High | CMP-005 |
| 6 | PyO3 Python bindings | 🟡 Medium — opens to 95% of ML practitioners | Medium | Stable API (REL-001) |
| 7 | Memory profiler | 🟡 Medium — saves debugging hours | Low | Telemetry |
| 8 | `#[differentiable]` macro | 🟡 Medium — reduces boilerplate | Medium | GRD-005 |
| 9 | Inference serving | 🟢 Future — needs the above first | High | Sharded loading, AMP |

---

## PyTorch-to-incin friction map

A PyTorch user switching to incin will reach for these things instinctively.
Every one that doesn't work is a reason to close the tab. The items below are
ordered from "most jarring absence" to "nice-to-have polish."

### 1. `torch.tensor([1, 2, 3])` — one-line tensor creation from data

**PyTorch:**
```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.randn(3, 4)
z = torch.zeros_like(x)
```

**Incin today:**
```rust
// Must pick a Backend, a Shape, write ::from_raw, pass bytes, specify DTypeId…
let x = Tensor::<Dyn, Backend>::from_raw(
    Backend::from_bytes::<f32>(bytes, &[3], DTypeId::F32, &DeviceId::cpu())?,
    vec![3]
)?;
```

**What to build:** A `tensor!` macro and convenience constructors:
```rust
let x = tensor![1.0, 2.0, 3.0];                   // infers f32, CPU
let y = Tensor::randn([3, 4]);                     // shape from array
let z = x.zeros_like();                            // same shape/device/dtype
let w = Tensor::from_slice(&[1.0, 2.0], [2]);      // no unsafe, no bytes
let v = Tensor::full([3, 3], 42.0);                // torch.full equivalent
let eye = Tensor::eye(4);                          // identity matrix
```

This is the single highest-impact ergonomic change. Look at the MNIST example's
`MnistCollate` — 40 lines of unsafe byte manipulation for what PyTorch does in
`torch.tensor(images).reshape(B, 1, 28, 28)`.

---

### 2. `+`, `-`, `*`, `/` — operator overloading

**PyTorch:** `z = x + y * 2.0`

**Incin today:** `let z = x.add(&y.mul_scalar(2.0)?)?;`

**What to build:** `impl Add/Sub/Mul/Div for &Tensor` and `for Tensor` (owned),
plus scalar variants. The `?` stays (Rust requires it), but the operator syntax
makes expressions readable:
```rust
let z = &x + &(&y * 2.0)?;
// or with owned tensors:
let z = x + y * 2.0;
```

---

### 3. `.to(device)` — move tensors and models between devices

**PyTorch:**
```python
model = model.to("cuda:0")
x = x.to("cuda:0")
```

**Incin today:** The backend is a *type parameter*. Moving between backends
requires type-level gymnastics.

**What to build:** A `.to(device)` method that returns
`Tensor<S, DispatchBackend<T, D>>` (the runtime-dispatch backend) so device
movement is a runtime call, not a type change. A `Model::to(device)` that moves
all parameters. For the common case where someone just wants "use my GPU":
```rust
let x = tensor![1.0, 2.0, 3.0].to(Device::cuda(0))?;
let model = model.to(Device::cuda(0))?;
```

---

### 4. `print(tensor)` — human-readable tensor display

**PyTorch:**
```python
>>> x = torch.randn(2, 3)
>>> print(x)
tensor([[ 0.3171, -0.9524,  0.1331],
        [-0.6189,  0.4829, -0.2168]])
```

**Incin today:** `Debug` output shows the internal struct fields, not the data.

**What to build:** Implement `Display` for `Tensor` that shows values in
PyTorch-style formatting:
```
Tensor([[ 0.3171, -0.9524,  0.1331],
        [-0.6189,  0.4829, -0.2168]], shape=[2, 3], dtype=f32, device=cpu)
```
With smart truncation for large tensors (show corners, elide middle).

---

### 5. A training loop that fits on one screen

**PyTorch:**
```python
for epoch in range(10):
    for x, y in dataloader:
        loss = model(x).cross_entropy(y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Incin today:** The MNIST example is 127 lines and includes 40 lines of unsafe
byte collation boilerplate. The training loop itself (lines 86–113) is clean,
but users never see it because they bounce off the data loading.

**What to build:**
- `Tensor::from_slice` / `tensor!` (item 1) to kill the byte manipulation
- A built-in `Collate` impl for common cases (`Vec<(Tensor, Tensor)>`)
- `optimizer.zero_grad()` integrated into `step()` (PyTorch convention)
- A `no_grad` context (see item 8 below)

Target: the MNIST example should be 30 lines, not 127.

---

### 6. `model.summary()` / `torchinfo`-style model inspection

**Incin today:** `ComputeStats` trait exists on `Linear`, but there's no
one-call summary that shows the full model architecture with a table of layers,
output shapes, param counts, and MACs.

---

### 7. `state_dict` round-trip — unified checkpoint save/load

**Incin today:** `model.save(Format::Safetensors, path)` exists. `AdamW` has
`state_dict()` / `load_state_dict()`. But there's no unified `Checkpoint` that
saves model + optimizer + epoch + scheduler state in one file and restores it
with one call.

---

### 8. `torch.no_grad()` — disable gradient tracking

**Incin today:** Gradient tracking is part of the type `G` parameter.

**What to build:** A runtime `no_grad` scope:
```rust
let output = incin::no_grad(|| model.forward(x))?;
let x_detached = x.detach();
```

---

### 9. DataLoader from tensors — zero boilerplate

**Incin today:** Requires implementing `Collate` trait with unsafe byte casting.

**What to build:**
```rust
let loader = DataLoader::from_tensors(&x_train, &y_train)
    .batch_size(32)
    .shuffle(true);
```

---

### 10. `.numpy()` / `from_numpy()` — ndarray interop

Behind an `ndarray` feature gate:
```rust
let arr: Array2<f32> = tensor.to_ndarray()?;
let tensor = Tensor::from_ndarray(arr)?;  // zero-copy if contiguous
```

---

### 11. Migration guide — "PyTorch to incin" cheat sheet

A single markdown page with side-by-side translations:
```
torch.tensor(...)        →  tensor![...]
nn.Linear(784, 128)      →  Linear::<Dyn, B>::build((784, 128))?
model.to("cuda")         →  model.to(Device::cuda(0))?
loss.backward()          →  loss.backward()?
optimizer.zero_grad()    →  (implicit in step)
torch.save(...)          →  Checkpoint::new().model(&m).save(path)?
torch.no_grad()          →  incin::no_grad(|| { ... })
x.shape                  →  x.shape_dims()
```

---

### 12. Error messages that name the PyTorch equivalent

Incin's diagnostics already humanise shape errors. Add hints that reference the
PyTorch pattern:
```
error: matmul dimension mismatch: [3, 4] × [5, 6]
  hint: in PyTorch this is torch.matmul(a, b) which requires a.shape[-1] == b.shape[-2]
```

---

### 13. Gradient checkpointing — trade compute for memory

```rust
let output = incin::checkpoint(|| {
    let x = model.layer1.forward(input)?;
    model.layer2.forward(x)
})?;
// forward intermediates freed; recomputed during backward
```

---

### 14. ONNX export

```rust
model.export_onnx("model.onnx", &dummy_input)?;
```

Export the `OperationSpec` graph to ONNX protobuf. Deploys incin models on
ONNX Runtime, TensorRT, CoreML, and every other inference engine.

---

### Impact heat-map

```
                    Easy                          Hard
              ┌──────────────────────────────────────────┐
  Critical    │  ① tensor!         ⑤ training loop      │
  adoption    │  ④ print(tensor)   ② operators           │
  impact      │  ⑪ migration guide ③ .to(device)         │
              ├──────────────────────────────────────────┤
  High        │  ⑧ no_grad         ⑨ DataLoader          │
  quality     │  ⑫ error messages  ⑥ model.summary()     │
  of life     │  ⑩ .numpy()        ⑦ Checkpoint          │
              ├──────────────────────────────────────────┤
  Future      │                    ⑬ grad checkpoint      │
  complete    │                    ⑭ ONNX export          │
              └──────────────────────────────────────────┘
```

**Recommended first sprint (1–2 weeks, transforms the first-use experience):**
1. `tensor!` macro + `Tensor::from_slice` (kills the unsafe boilerplate)
2. `Display` for `Tensor` (makes debugging possible)
3. Operator overloading (`+`, `-`, `*`, `/`)
4. Migration guide (markdown page, half a day)
5. Rewrite MNIST example in 30 lines using the above
