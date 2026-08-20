# 09 - Real checkpoints: sharded loading · the model hub · Mixture of Experts

> **Status:** NOT STARTED (specified 2026-07-29). **Effort:** Low (A) → High (D).
> **Priority:** the "can I load the model I actually want" close. Today a user
> who types `model!("kimi-k2", …)` gets a compile error, and there is no
> workaround.

## Goal

Three things a user expects to work and which currently do not:

1. **Load a checkpoint that ships as shards.** Kimi K2/K3, DeepSeek-V3,
   Llama-405B and every other large open-weights release ship as
   `model-00001-of-000NN.safetensors` plus a `model.safetensors.index.json`.
   None of them can be named by any API in this repo.
2. **Fetch a model by repo id and get a typed struct out of it**, the way
   `model!` gives you one from a local file - without making `cargo check`
   depend on the network.
3. **Build and run Mixture-of-Experts models**, which is what most of the
   frontier open-weights releases now are.

## Grounding - what exists today, cited

Verified 2026-07-29 against the working tree. Confirm before acting.

**Sharding: absent.** Nothing in the repo reads `model.safetensors.index.json`
(no occurrence of the string). [`hub.rs:56`](../../crates/incin-data/src/hub.rs)
is `let file = filename.unwrap_or("model.safetensors");` - one file, by name.
The `model!` / `import_model!` macro takes a single `LitStr` path
([`safetensors.rs:14`](../../crates/incin-macros/src/safetensors.rs)) and
rejects anything that is not `.safetensors`/`.onnx`.

**The loader reads whole files into RAM.**
[`serialize.rs:129`](../../crates/incin-core/src/serialize.rs) is
`std::fs::read(self.path)` - no mmap - and the deserializer then materializes
*every* tensor onto the device and returns a `BTreeMap` holding all of them.
Peak host residency is therefore ≈ the full checkpoint, on top of device
memory. This is fine for ResNet-18 and impossible for anything worth sharding,
so **an index reader alone does not make large models loadable.**

**Import is hard-bound to f32.** Both
[`hub.rs:51`](../../crates/incin-data/src/hub.rs) and
[`hub.rs:68`](../../crates/incin-data/src/hub.rs) require
`B: Backend<FloatElem = f32> + SupportsDType<f32>`. Real checkpoints are bf16
or fp8. `serialize.rs:144` returns `"Unsupported dtype in safetensors"` for
anything outside F32/F64/F16/BF16/U32/I64/U8.

**`load_state_dict` reports no key diff.** There is no missing-key or
unexpected-key accounting anywhere in
[`module.rs`](../../crates/incin-core/src/nn/module.rs) - no `strict=False`
equivalent, and a checkpoint whose names don't match yours fails without
telling you which names those were.

**No container module.** `Parameters`/`StateDict` are implemented for
`Sequential<L1, L2>` ([`module.rs:539`](../../crates/incin-core/src/nn/module.rs),
binary-nested, with `seq!` sugar at `module.rs:1011`), `Option<T>`,
`PhantomData<T>`, and via `impl_dummy_state!(usize, f32)`. There is **no impl
for `[T; N]` or `Vec<T>`**, so `experts: [FeedForward<..>; 8]` cannot be a
module field at all. `Sequential`'s `flat_width` running-index discipline
(`module.rs:555`) is exactly the mechanism an array impl needs - this is an
extension of an existing design, not a new one.

**No attention, no transformer block.** [`nn/`](../../crates/incin-core/src/nn/)
has Linear, Conv1d/2d, Embedding, LayerNorm, RmsNorm, BatchNorm, LSTM, RNN,
Dropout, pooling, Flatten. No MHA, no rotary embeddings, no KV cache.

**No MoE anywhere.** The only occurrence of the concept in the repo is
[`PROPOSALS.md:1063`](../../PROPOSALS.md), a parallelism-strategy table row
("Expert parallel | Distribute experts; all-to-all tokens"), and
[`:1373`](../../PROPOSALS.md) noting expert parallelism as a later opportunity.

**But every routing primitive already exists.** `topk`, `gather`, `scatter`,
`index_select`, `argsort`, `cumsum`, `masked_fill`, `split`, `chunk`, `concat`,
`stack`, `narrow` are all in `crates/incin-core/src/tensor/`. A dense (masked)
MoE forward pass needs **no new kernels**.

**The macro already writes a shape sidecar.**
[`safetensors.rs:159`](../../crates/incin-macros/src/safetensors.rs) caches the
parsed tensor shapes as JSON at `<file>.safetensors.incin_meta`, keyed on
mtime, disabled by `INCIN_DISABLE_META_CACHE=1`. Workstream B is largely the
promotion of this private cache into a first-class, committable artifact.

---

## The one design decision that is not mechanical

**A `fetch_model!` macro that downloads at expansion time is ruled out**, and by
this repo's own rules rather than by taste. [`PROPOSALS.md:1427`](../../PROPOSALS.md)
requires public macros to "avoid filesystem/network access except existing
explicit import macros", and [`:1601`](../../PROPOSALS.md) states that secrets,
rendezvous tokens, and network addresses "are never embedded by a proc macro" - 
which is precisely what an authenticated Hub download would need. `D-031`
already forbids the weaker act of reading the caller's `Cargo.toml` during
expansion.

Independently of policy it is the wrong shape: it makes `cargo check`
network-dependent, breaks offline and sandboxed builds, re-runs on cache misses
the user cannot predict, and puts a multi-hundred-gigabyte transfer inside
rustc where it cannot be interrupted, resumed, or debugged.

**The split that gets the UX anyway:** the macro needs the model's *shapes*, not
its *weights*. Shapes are ~100 KB of JSON; weights are runtime's problem.

```bash
cargo incin fetch moonshotai/Kimi-K2-Instruct --manifest-only
```

touches the network **once, in a CLI**, and writes a versioned, committable
`models/kimi-k2.incin.json`: repo id, pinned revision, per-shard digests, and
every tensor's name/shape/dtype. Then:

```rust
model!("models/kimi-k2.incin.json", KimiK2);
```

reads only that file. The build is hermetic and offline; the manifest is
reviewable in a PR and diffs meaningfully when an upstream revision moves; the
network and the `INCIN_HUB_TOKEN` handling stay in the one place that already
has them ([`hub.rs:13`](../../crates/incin-data/src/hub.rs)).

---

## Workstream A - unblock the small things first (Effort: Low)

These are days, not weeks, and everything below depends on them.

### Task 09.A1 - `Parameters`/`StateDict`/`NamedLayers` for `[T; N]` and `Vec<T>`
Follow `Sequential`'s `flat_width` contract exactly (`module.rs:539-650`): the
array's width is `N * T::flat_width()`, and element `i` starts at
`base_index + i * T::flat_width()`, so keys come out as PyTorch's flat
`0.weight, 1.weight, …` scheme. `Vec<T>` needs a runtime width and therefore
cannot implement `flat_width()` as a `const`-shaped associated fn if that is
what the trait requires - **check this before promising `Vec`**; if it does not
fit, ship `[T; N]` alone and record why. This unblocks both expert lists and
ordinary layer stacks written as arrays rather than nested pairs.

### Task 09.A2 - missing/unexpected-key diff on `load_state_dict`
Return a report (present-and-loaded, missing-from-checkpoint,
present-but-unused, shape-mismatched) rather than a bare error, and a strict
flag that turns a non-empty diff into a failure. Every import failure today is
either silent or opaque; this is the single highest ratio of user pain relieved
to code written in this document.

### Task 09.A3 - extend `cargo incin inspect` to accept a repo id
[`io/inspect.rs`](../../crates/incin-core/src/io/inspect.rs) already prints a
tensor table for a local `.safetensors`/`.gguf`/`.onnx`. Teach the CLI to
resolve a repo id to its index (metadata only - never the shards) and print the
tensor tree, parameter count, and memory footprint at each candidate dtype
**before** anything is downloaded.

## Workstream B - the manifest and the fetch CLI (Effort: Medium)

### Task 09.B1 - a versioned model manifest
Define the `*.incin.json` schema (a `schema` integer first, per the ledger's own
convention) and make the macro's `.incin_meta` writer emit it. One reader,
shared by the macro and the runtime loader - do not let the macro parse a format
the loader doesn't.

### Task 09.B2 - `cargo incin fetch <repo> [--revision] [--manifest-only]`
The only thing in the project that touches the network for weights. Pins a
revision, records per-shard digests, resolves `model.safetensors.index.json`
into an explicit shard list, and verifies digests on subsequent runs.

### Task 09.B3 - `model!` reads a manifest
Accept a manifest path in addition to a checkpoint path. `load_default_weights`
(`safetensors.rs:236`) resolves the manifest's shard list through the hub cache
instead of one hardcoded path.

## Workstream C - actually loading a large checkpoint (Effort: Medium-High)

### Task 09.C1 - sharded index resolution
Read `model.safetensors.index.json`'s `weight_map`, and present N shards as one
logical state dict.

### Task 09.C2 - mmap + streaming placement
Replace `std::fs::read` (`serialize.rs:129`) with a memory map, and load
**shard by shard, tensor by tensor, placing each one and dropping the host copy
before reading the next**. Without this, C1 buys nothing: you would still need
the whole checkpoint resident. Ties directly into `DST-003`/`DST-004` - a
sharded load onto a device mesh is the same traversal with a placement per
tensor.

### Task 09.C3 - import dtype policy
Drop the `FloatElem = f32` bound (`hub.rs:51`, `hub.rs:68`). Decide and document
the cast rule when checkpoint dtype ≠ backend dtype (bf16 → f32 widen, fp8 →
dequantize, refuse silently-lossy narrowing). `ggml-quants` is already a
dependency and [`io/gguf.rs`](../../crates/incin-core/src/io/gguf.rs) already
has quantized-block machinery to reuse.

## Workstream D - MoE (Effort: High)

Depends on A1. Do not start before it.

### Task 09.D1 - attention and a transformer block
MHA with a KV cache, rotary embeddings, and a block that composes them with the
existing `RmsNorm`. MoE is not useful without this, and this is independently
the most-requested missing layer family.

### Task 09.D2 - `Router<E, TopK>`
Typed expert count and top-k as type parameters, so "top-2 of 8" is a type and
not two runtime integers that can disagree. Emits per-expert assignments and the
auxiliary load-balancing loss. Built from `topk` + `cumsum` + `masked_fill`,
which already exist.

### Task 09.D3 - `MoE<E, Expert>` - dense masked path first
The masked path (run every expert, weight by the routing mask) is correct,
simple, and slow. Land it, test it, *then* add the gather/scatter dispatch path
and assert the two agree numerically. Do not write the fast one first.

### Task 09.D4 - expert-parallel mesh axis
Only after `DST-005`/`DST-006` exist. See the note below.

---

## Time-sensitive note for `DST-002`/`DST-003` (open now)

`MeshSpec<DP, TP = TensorParallel<U1>, PP = Pipeline<U1>>`
([`mesh.rs:64`](../../crates/incin-core/src/dist/mesh.rs)) has three axes;
`PROPOSALS.md:1063` lists expert parallelism as a fourth strategy.

Adding a fourth axis *later* is cheap and additive: a defaulted
`EP = ExpertParallel<U1>` parameter leaves every existing spelling compiling
with an unchanged `World`, at the cost of one more bound in the single
`ValidMesh` impl and refreshed compile-fail baselines.

What is **not** free later is the rank-layout convention. `DST-002`'s
`CollectiveGroups` is about to fix how a rank decomposes into coordinates, and
all-to-all is a genuinely different collective from all-reduce. **Write
`CollectiveGroups` generically over the axis list rather than hardcoding three**
 -  it costs nothing today and is the difference between adding an axis and
re-cutting a convention two rows later.

## Verification

- A: `cargo test -p incin-core --test module_containers` (new) and the
  workstream README §2 loop.
- B: a hermetic test - write a manifest to a `tempfile`, expand `model!`
  against it, assert the generated struct's fields, **with no network access
  in the test**. Plus one manual `cargo incin fetch` against a small real repo,
  with the command and output recorded in the PR.
- C: load a genuinely sharded checkpoint (a small multi-shard repo, not a
  synthetic one) and record peak RSS - the number is the point of C2.
- D: numeric agreement between the dense-masked and dispatch paths, to a stated
  tolerance; router load-balance loss against a hand-computed example.

## Risks / DO-NOT

- **DO-NOT put network access in a proc macro**, in any form, however
  convenient. See the design decision above; it is a policy violation, not a
  preference.
- **DO-NOT commit a manifest containing a token, a signed URL, or a
  user-specific cache path.** The manifest is a public, committable artifact.
- **DO-NOT ship C1 without C2.** A shard index without streaming placement
  reads every shard into RAM and still cannot load the models it exists for.
- **DO-NOT write the gather/scatter MoE dispatch path first.** The dense masked
  path is the reference the fast path is checked against; without it there is
  nothing to be right relative to.
- **DO-NOT let this workstream widen the `incin-core` dependency graph.**
  Hub/network code belongs in `incin-data` (D-015).
- **DO-NOT mark C "done" against a synthetic multi-file checkpoint.** Sharded
  loading is done when a real published sharded model loads - the deployment
  analogue of the "compiles clean ≠ verified" rule.

## Demo scripts

- *"`cargo incin fetch deepseek-ai/DeepSeek-V3`. Now the model's 3800 tensors
  are a Rust type, and a wrong shape is a compile error."*
- *"This manifest is 90 KB and it's in git. My build doesn't touch the network,
  and if upstream re-uploads a shard, my PR diff says so."*
- *"Top-2 of 8 experts. The 2 and the 8 are type parameters - routing to an
  expert that doesn't exist doesn't compile."*
