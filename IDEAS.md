# Kindle — Ideas: Easing the PyTorch → Kindle Switch

Running list of concrete UX ideas aimed at one goal: make it as painless as
possible for a working PyTorch developer to pick up Kindle. Each entry is
grounded in an actual gap or actual existing-but-hidden strength found while
reading the code — not a generic wishlist. None of these are started; they
need scoping (like `IMPLEMENTATION_PLAN.md` §8/§9) before any get built.

## Gaps — friction a PyTorch developer would hit immediately

- **No `print(model)` equivalent.** PyTorch developers reflexively run
  `print(model)` to sanity-check an architecture. Kindle already has the
  underlying data for this — `NamedLayers`/`LayerNode`
  (`kindle-core/src/nn/module.rs`) walk the exact same tree `#[module]`
  builds — but nothing turns it into a printable tree. A `Display` impl (or
  a `model.summary()` helper) built on the existing `layer_structure()` call
  would be a small, self-contained win that directly answers a reflex
  PyTorch users have on day one.
- **Compile errors from shape mismatches are real Rust generic-trait-bound
  errors**, which for `typenum`-heavy code can be walls of `Prod<UInt<...>>`
  noise — intimidating for someone whose only error-message experience is
  PyTorch's runtime `RuntimeError: size mismatch`. Rust has a purpose-built,
  stable tool for this: `#[diagnostic::on_unimplemented(message = "...",
  label = "...")]` on the relevant shape-equality traits, which lets the
  compiler print a custom, plain-English message instead of the raw
  trait-resolution failure. Worth auditing which traits are the ones
  actually failing in practice (grep existing `.stderr` compile-fail test
  snapshots for the ugliest ones) and prioritizing those.
- **`Sequential`'s state_dict keys don't look like PyTorch's.** PyTorch's
  `nn.Sequential` state dict is flat: `0.weight`, `1.weight`, `2.weight`. A
  `seq!`-built `Sequential<A, Sequential<B, C>>`'s state dict
  (`kindle-core/src/nn/module.rs`'s `Parameters`/`StateDict` impls) is
  nested by construction: `0.weight`, `1.0.weight`, `1.1.weight`. A PyTorch
  user inspecting a saved `.safetensors` file's key names would find this
  surprising, and it complicates loading weights exported from a real
  PyTorch `nn.Sequential`. This is a real behavior change if fixed (see
  `IMPLEMENTATION_PLAN.md` §8.4-adjacent note — not written up as a full
  proposal yet), not just a docs fix.
- **No `from_pretrained("org/model")` one-liner** — see the strength below,
  the pieces exist but aren't glued together.
- **No optimizer checkpoint/resume.** `AdamW`/`Adam` (`kindle-core/src/optim/mod.rs`)
  hold real per-parameter state (`m`/`v` momentum/variance `BTreeMap`s, a
  `step` counter) but neither implements `StateDict` — confirmed by grep,
  zero `impl StateDict for AdamW/Adam/SGD` anywhere. PyTorch's
  `torch.save({'model': ..., 'optimizer': optimizer.state_dict()})` +
  resume-training workflow has no equivalent here today: interrupting and
  resuming a real training run silently restarts Adam's moment estimates
  from zero instead of continuing them, which measurably changes early-resume
  training dynamics. This is a real training-workflow gap, not just an API
  nicety, and probably the single most PyTorch-switcher-relevant one on this
  list once someone runs anything longer than a toy example.
- **No mixed-precision / autocast context.** Confirmed by grep: no
  `autocast`-equivalent anywhere. The dtype-policy machinery
  (`dtype_policy.rs`, referenced throughout `ROADMAP.md`'s CUDA kernel work)
  already resolves storage/compute/accumulator dtypes per op, which is most
  of the hard part — but there's no user-facing "run this block in f16/bf16
  with f32 accumulation" scope the way `torch.cuda.amp.autocast()` provides.
  Not scoped further here (real design question: a context-manager-like
  RAII guard vs. a generic parameter on the training loop), just flagging
  that it's genuinely absent, not just undocumented.

## Existing strengths that are under-surfaced (market these, don't just build new things)

- **Compile-time `no_grad`, not a runtime context manager.** `Tensor<S, B, K,
  G>`'s fourth type parameter (`Grad`/`NoGrad`, `kindle-core/src/tensor/grad.rs`)
  makes gradient-tracking a *type*, not a runtime flag — `.detach()`
  (`tensor/base.rs:391`) converts `Grad` to `NoGrad` at the type level. This
  is strictly stronger than PyTorch's `with torch.no_grad():` context
  manager: forgetting to detach an inference-only tensor, or accidentally
  running training code inside a `no_grad` block, becomes a *compile error*
  here instead of a silent runtime footgun (both are common real PyTorch
  bugs — the second one is exactly how people accidentally train with a
  frozen backbone that's still eating gradient memory). Currently invisible
  in the docs — same "hidden strength" pattern as `zero_grad` above, and
  arguably an even stronger selling point since it's a category of bug
  Python's type system structurally cannot catch at all.
- **`DataLoader` already implements `Iterator`.** Confirmed:
  `kindle-data/src/loader.rs:68`'s `fn next(&mut self)`. `for batch in
  loader { ... }` already works exactly like PyTorch's `for batch in
  dataloader:` — no gap here, just worth confirming in the same audit pass
  that found the `print(model)` gap, since it's the kind of thing that's
  easy to assume is missing without checking.
- **`SGD::new`/`AdamW::new` already take `model.parameters()` directly** —
  `Parameters::parameters()`'s return type (`BTreeMap<String, B::RawVar>`)
  is exactly what the optimizer constructors want, so
  `SGD::new(model.parameters(), 0.01)` already reads almost identically to
  PyTorch's `optim.SGD(model.parameters(), lr=0.01)`. Not a gap — confirmed
  while checking whether optimizer construction needed its own UX pass; it
  doesn't, this part already matches PyTorch's ergonomics closely.

- **No `zero_grad()` needed, ever.** Confirmed by grep: `zero_grad` appears
  nowhere in this codebase. Every backend's `backward()` builds a fresh
  gradient map from scratch each call (e.g. `cuda/tape.rs::backward`,
  `cpu/tape.rs`) instead of PyTorch's accumulate-by-default model. This is a
  genuine ergonomic and correctness win over PyTorch — forgetting
  `zero_grad()` is one of the most common real-world PyTorch bugs (silently
  accumulating gradients across steps) — but it's currently invisible;
  nothing in the README/docs calls it out as a deliberate design advantage.
  Should be a headline bullet in the README's "Key Features" list, not just
  an implicit consequence of the tape design.
- **A HuggingFace-Hub-style downloader already exists and is undocumented
  at the top level.** `kindle-data::hub` (`HubApi`, `HubRepo::get`,
  `download(repo_id, filename)`) is already re-exported as `kindle::hub`
  (`kindle/src/lib.rs`'s `pub mod hub`). Combined with the just-added
  `load_safetensors`/`save_safetensors` (`64e41b6`), this is 90% of the way
  to a PyTorch/HF-familiar `Model::from_pretrained("org/model")` experience
  — it just isn't documented or wired into a single convenience call
  anywhere. Cheap, high-leverage: write the doc example showing
  `hub::download(...)` + `load_safetensors(...)` chained together, and
  consider a thin `from_pretrained` helper on `StateDict` that does both
  steps in one call.
- **Compile-time shape safety itself is the actual pitch**, and the crate's
  own doc already leads with it (`kindle/src/lib.rs`'s "Key Features") — the
  gap isn't messaging, it's that the *first* time this helps a PyTorch
  user is also the *first* time they see a scary generic error (see gap
  above). Fixing the error messages is what would let this strength
  actually land instead of bouncing people at the first friction point.

## Open questions needing a decision (not just UX — real design calls)

- **`getrandom`/WASM entropy source** (found 2026-07-22, during Phase 0
  NEON/WASM SIMD verification). `kindle-backends` cannot currently compile
  for `wasm32-unknown-unknown` at all — `rand`/`rand_core` transitively pull
  `getrandom 0.2`, which refuses to build for bare `wasm32-unknown-unknown`
  without an explicit entropy-source opt-in. Blocks `CreationOps::rand`/
  `randn` (and therefore any WASM build) entirely — independent of the SIMD
  kernel work itself. Three ways to resolve, each a real product decision:
  1. `getrandom`'s `js` feature as a `wasm32`-only target dependency —
     standard, but commits to assuming a JS host (browser/Node) at runtime.
  2. `getrandom`'s `custom` feature — caller registers an RNG hook, more
     portable, pushes a setup step onto every WASM consumer.
  3. Leave `rand`/`randn` erroring on WASM for now, scope initial WASM
     support to inference-only — simplest, but a real capability cut.
  This matters for the PyTorch-switcher story too: if WASM/browser inference
  is ever part of the pitch (PyTorch's browser story is famously weak —
  ONNX Runtime Web / TF.js fill that gap today), this is the blocker.

## Parked / not yet scoped

- (nothing else yet — add here as things come up)
