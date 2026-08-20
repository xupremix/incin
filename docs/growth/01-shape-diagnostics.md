# 01 - Compile-Time Shape Diagnostics

> **Depends on:** task `00` (the `incin-diagnostics` crate).
> **Effort:** Medium. **Priority:** #1 - build this first. It fixes the exact
> friction point where a curious PyTorch dev bounces *today*, and it is the
> single most shareable demo.

## Goal

When a user writes a shape-invalid operation, the compiler should say something
like:

```
error: Cannot reshape a tensor of 6 elements into a shape of 8 elements
  --> src/model.rs:11:15
   |
11 |     let y = x.reshape::<s![2, 4]>(((), ()));
   |               ^^^^^^^ this reshape changes the element count (6 → 8)
   = note: reshape must preserve the total number of elements
```

instead of today's wall:

```
error[E0271]: type mismatch resolving `<(UInt<UInt<UTerm, B1>, B0>, ...) as ElementCount>::Count == UInt<UInt<UInt<UTerm, B1>, B1>, B0>`
   = note: expected struct `UInt<UInt<UInt<UTerm, B1>, B1>, B0>`
              found struct `UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0>`
```

## Grounding (what exists today)

- `#[diagnostic::on_unimplemented(...)]` is **already applied** to most shape
  traits: `ReshapeShape` (`shapes/reshape.rs:53`), `BroadcastShape`
  (`shapes/broadcast.rs:27`), `ConcatShape` (`shapes/concat.rs:5`), `StackShape`
  (`shapes/stack.rs:4`), `Pool2dShape`/`SpatialConv1d`/`SpatialConv2d`/
  `AdaptiveAvgPool2dShape` (`shapes/spatial.rs`), `ReshapeTarget`/`SliceTarget`
  (`shapes/idx.rs`), `Transpose`/`ReduceDim`/`ReduceKeepDim`/`Flatten`
  (`shapes/shape_ops.rs`).
- The `cargo incin` translator already post-processes rendered diagnostics
  (`translate_typenum_text`, now in `incin-diagnostics` after task 00).
- A representative failing case + its exact stderr is captured at
  `crates/incin-core/tests/compile_fail/reshape_static_mismatch.{rs,stderr}`.

## The technical crux (READ THIS - it is why the messages are still ugly)

`#[diagnostic::on_unimplemented]` only fires for **unsatisfied trait bounds
(`E0277`)**. It does **nothing** for **associated-type projection mismatches
(`E0271`)**.

Look at `shapes/reshape.rs:59-67`:

```rust
#[diagnostic::on_unimplemented(message = "Cannot reshape from `{Self}` to `{Target}`", ...)]
pub trait ReshapeShape<Target: Shape>: Shape {}

impl<S1, S2> ReshapeShape<S2> for S1
where
    S1: Shape + ElementCount,
    S2: Shape + ElementCount<Count = <S1 as ElementCount>::Count>, // ← associated-type EQUALITY
{}
```

The call site (`tensor/ops/manipulation.rs:162-165`) bounds `S:
ReshapeShape<S2>`. When element counts differ, rustc *does* fail - but it fails
while trying to satisfy the blanket impl's `ElementCount<Count = …>`
**equality**, and reports that as an `E0271` projection mismatch
(`<… as ElementCount>::Count == UInt<…>`). Because it is `E0271`, the nice
`on_unimplemented` message on `ReshapeShape` is **never shown**. That is the
entire bug. The attribute is present but structurally bypassed.

### The fix: turn the equality into a bound

Introduce a reflexive marker trait that is implemented **only for equal types**,
and route the count comparison through it. Then a count mismatch becomes an
*unsatisfied trait bound* (`E0277`), which `on_unimplemented` **does** decorate.

```rust
// in incin-core/src/shapes/reshape.rs

/// Witness that two type-level element counts are identical. Implemented
/// reflexively - only `SameCount<N>` for the *same* `N` exists - so requiring
/// `A: SameCount<B>` is a compile-time assertion that `A == B`, but one that
/// fails as an unsatisfied trait bound (E0277) rather than an associated-type
/// projection mismatch (E0271). E0277 is what `#[diagnostic::on_unimplemented]`
/// can decorate; E0271 is not. This indirection is the whole reason the
/// reshape error is now readable.
#[diagnostic::on_unimplemented(
    message = "Cannot reshape: source has {Self} elements but the target shape has a different number",
    label = "element count changes here",
    note = "reshape must preserve the total number of elements"
)]
pub trait SameCount<Rhs> {}
impl<N> SameCount<N> for N {}

impl<S1, S2> ReshapeShape<S2> for S1
where
    S1: Shape + ElementCount,
    S2: Shape + ElementCount,
    <S1 as ElementCount>::Count: SameCount<<S2 as ElementCount>::Count>, // ← now a BOUND, not equality
{}
```

**Caveat that must be verified, not assumed:** even after this change, the
`{Self}`/`{Rhs}` interpolated into the message render as the *raw typenum type
name* (`UInt<UInt<…>>`), because `on_unimplemented` interpolates `Display` of
the type path. So the compiler message becomes *anchored at the call site and
labeled clearly*, but the numbers inside may still be `UInt<…>`. **That is
expected and fine** - the second layer (the translator, and the IDE extension in
doc `02`) converts those to decimals. The two layers are complementary:

1. **This doc** makes rustc emit a clean, call-site-anchored `E0277` with a
   plain-English `message`/`label` instead of a cryptic, location-poor `E0271`.
2. **The translator / doc `02`** rewrites any residual `UInt<…>` in the rendered
   text to decimals.

Do not try to make `on_unimplemented` alone print decimals - it cannot; typenum
values are types, and the attribute has no arithmetic. Chasing that is a rabbit
hole.

> **2026-07-23 status update - Tasks 01.1–01.2 done, verified empirically:**
> `SameCount` landed exactly as designed above, with one refinement: the
> message interpolates **both** `{Self}` and `{Rhs}` (`Rhs` is `SameCount`'s own
> generic parameter name, so it's a valid substitution) - `"Cannot reshape:
> source has {Self} elements but the target shape has {Rhs} elements"` - which
> reads better than the single-sided draft above once both counts are shown.
> Regenerating `reshape_static_mismatch.stderr` and `reshape_rank_mismatch.stderr`
> confirms the top-level error code actually flipped from `E0271` to `E0277`,
> and piping the regenerated message through `cargo incin translate` produces
> exactly the target end state: `Cannot reshape: source has 6 elements but the
> target shape has 8 elements`. The `= note: required for ... to implement
> ReshapeShape<...>` / `required by a bound in reshape` trailer is *retained*
> underneath the new message (not replaced), so the technical trace is still
> there for anyone who wants it - this is strictly additive.
>
> **Task 01.1 audit result:** grepped every `compile_fail/*.stderr` for a
> top-level `error[E0271]` - exactly two files matched
> (`reshape_static_mismatch.stderr`, `reshape_rank_mismatch.stderr`), and both
> go through `ReshapeShape`/`ElementCount`. **No other shape trait needs this
> treatment right now** - `BroadcastShape`, `ConcatShape`, `StackShape`,
> `MatMulShape` (`incin-core/src/tensor/matmul.rs:21-25` - note this trait
> lives outside `shapes/`, not in the list this doc originally enumerated) and
> the `spatial.rs` conv/pool traits all already fail as `E0277` because their
> blanket impls are ordinary trait-impl resolution, not associated-type
> equality. Re-run the same grep after adding any *new* shape trait with a
> `Count = ...`-style blanket impl - that's the specific pattern that produces
> this bug.
>
> **Task 01.5 partial:** added `compile_fail/matmul_static_mismatch.{rs,stderr}`
> - matmul had no static-mismatch snapshot at all despite `MatMulShape` already
> carrying a good `on_unimplemented` message. Confirmed it renders as `E0277`
> with a readable message out of the box. Add/broadcast, conv2d, and linear
> still have snapshots per the original audit; a dedicated add/broadcast
> snapshot beyond `forward_broadcast_mismatch.stderr` was not re-verified this
> pass. Task 01.3 (apply pattern elsewhere) needs no work - 01.1 found no other
> offenders. Task 01.4 (copy-edit every message) was spot-checked, not done
> exhaustively: `broadcast.rs`, `concat.rs`, `stack.rs`, `shape_ops.rs`,
> `spatial.rs`, `idx.rs`, and `matmul.rs` were all read and are already
> plain-English with no trait jargon - none needed a wording change this pass.
>
> **2026-07-23 follow-up - Task 01.4 finished exhaustively, 3 more real gaps
> found and fixed (not just wording).** The previous pass read every
> `on_unimplemented` message that *existed*; it didn't check whether every
> shape trait that *can fail at a call site a beginner reaches* actually
> *had* one. Cross-referencing every `pub trait *Shape*` definition against
> which ones carry `#[diagnostic::on_unimplemented]` (`grep -rn "^pub trait.*Shape"`
> across `src/`, then checked each for the attribute) found three with none
> at all:
> - **`EndsWith<D>`** (`shapes/shape.rs`) - the bound `Linear::forward` (and
>   `RMSNorm`/`LayerNorm`) actually fails on when the input's last dimension
>   is wrong. Confirmed via `forward_linear_static_mismatch.stderr`: before
>   this fix, a **Linear layer shape mismatch - the single most common
>   layer a beginner uses** - rendered as a raw `the trait bound (...):
>   EndsWith<...> is not satisfied` wall with zero framing. This was the
>   biggest real gap in the whole audit.
> - **`HasChannels1D<D>` / `HasChannels2D<D>`** (`shapes/shape.rs`) - the
>   *separate*, additional bound (alongside `SpatialConv1d`/`SpatialConv2d`,
>   which only check rank/spatial arithmetic and are `CIn`-agnostic by
>   design - verified by reading `impl_conv2d_shape!`, `CIn: Dim` is
>   completely free) that actually fails on a `Conv1d`/`Conv2d`/
>   `BatchNorm1d`/`BatchNorm2d` **channel-count** mismatch. Same raw-wall
>   symptom, confirmed via the pre-existing, already-clean
>   `forward_batchnorm_mismatch.rs` fixture (its `.stderr` before this fix
>   showed the exact same "trait bound ... is not satisfied" pattern).
> - **`KernelConv2dShape`** (`tensor/conv2d.rs`) - the raw `Tensor::conv2d`
>   method (as opposed to the `nn::Conv2d` *layer*) had no message at all.
>   Added one; a **new** snapshot (`kernel_conv2d_channel_mismatch.{rs,stderr}`)
>   proves it fires for a channel-count mismatch. It does **not** fire for
>   the pre-existing arithmetic-underflow case in `conv2d_invalid_shape.rs`
>   (kernel bigger than input) - verified that snapshot is byte-identical
>   after this change - because that failure happens earlier, resolving a
>   `Sub<B1>`/`Div` bound in the impl's `where`-clause on a *foreign*
>   typenum trait, before `KernelConv2dShape` itself is ever checked. That
>   is a permanent, structural limitation (you cannot `on_unimplemented` a
>   trait you don't own), not a gap - documented here rather than chased.
>
> All three got a plain-English message (`"Cannot use shape {Self} here:
> its last dimension must be {D}"` / `"...it must have {D} channels"` /
> `"Cannot apply a {K}-shaped kernel to input shape {Self}"`), which
> automatically improved **five** existing/new snapshots at once:
> `forward_linear_static_mismatch`, `forward_linear_partial_mismatch`,
> `forward_conv1d_static_mismatch`, `forward_conv2d_static_mismatch`,
> `forward_batchnorm_mismatch` (the last one needed no `.rs` change - its
> fixture was already correct - just a snapshot regeneration), plus the new
> `kernel_conv2d_channel_mismatch`.
>
> **Also found while regenerating (Task 01.5, "add a snapshot for every op
> a beginner hits" - this is what that audit turns up if you actually try
> to compile the existing ones instead of trusting they still do what their
> name says):** `forward_linear_static_mismatch.rs`,
> `forward_linear_partial_mismatch.rs`, `forward_conv1d_static_mismatch.rs`,
> and `forward_conv2d_static_mismatch.rs` were **all four already broken**
> before today, in a way their checked-in `.stderr` snapshots masked rather
> than caught:
> - All four had a duplicate `use incin_core as incin;` alongside
>   `extern crate incin_core as incin;` (E0254) and a stray, dangling
>   `#[derive(Clone, Default)]` above `fn main` (E0774, "derive may only be
>   applied to structs/enums/unions") - both harmless-looking copy/paste
>   leftovers that `trybuild` had been faithfully snapshotting as if they
>   were the intended test output. Removed both from all four.
> - The two `Conv1d`/`Conv2d` files were worse: they called
>   `Conv1d::<3, 1, 0, 1, s![16, 3, 3], Backend>::new()` - **six** raw
>   generic arguments against a struct that takes three
>   (`S: Conv1dShape, B: Backend, Bias`), and a `.new()` method that has
>   never existed (`.build()` is the only constructor - confirmed against
>   every real usage in `builder_permutations.rs`/`layers.rs`/the working
>   examples). This is E0107, a hard compile error unrelated to shapes - 
>   the test was failing for a reason that had nothing to do with what it
>   claimed to snapshot, and had never actually exercised
>   `HasChannels1D`/`HasChannels2D` at all. Rewrote both using the real
>   `Conv1d::<s![16, 3, 3, 1, 1, 1], B>::build(())` API, with a channel
>   count deliberately wrong (4 vs. the layer's 3) and spatial size large
>   enough to avoid also tripping the unrelated underflow case.
> - `forward_linear_partial_mismatch.rs` had a *third*, independent bug even
>   after the above: `Tensor::<s![dyn, 5], ...>::zeros([2, 5])` - passing
>   the *full* dims as a `[usize; 2]` array to a shape that is only
>   *partially* dynamic, instead of just the one dynamic dim
>   (`.zeros(2)`) - the same `ArgInto`-lifting mistake documented in doc
>   `03`'s findings, caught here because it left an unrelated `ArgInto`
>   error as the snapshot's *first* error, ahead of the real `EndsWith`
>   one it was meant to demonstrate.
>
> None of this was visible from reading the traits or the doc's own prior
> status update - only from actually trying to regenerate every snapshot
> and reading what came out. Full verification after all of the above:
> `cargo test -p incin-core --test compile_tests` (19 files, all green),
> then the complete workspace loop (fmt / clippy `-D warnings` / full
> `cargo test --workspace --all-targets` / examples build / WGPU lib tests
> / CUDA compile-check) - all clean. `docs/growth/README.md`'s "no other
> offenders" claim from the previous pass turned out to be correct for
> *E0271-vs-E0277* specifically (the narrow thing it checked); it just
> wasn't the same question as "does every shape trait have a message at
> all," which is what this pass actually closes out.

## Architecture / data flow

```
user code: x.reshape::<s![2,4]>(...)
        │  (S = [2,3] with 6 elems, S2 = [2,4] with 8 elems)
        ▼
rustc trait solving:
   needs  S: ReshapeShape<S2>
   → blanket impl requires  Count<S>: SameCount<Count<S2>>
   → U6: SameCount<U8>  is NOT implemented   ── E0277 ──▶ on_unimplemented fires
        │
        ▼
rendered diagnostic (may still contain UInt<…> in {Self}/{Rhs})
        │
        ├─▶ plain `cargo build`  : readable message, numbers as UInt<…>
        ├─▶ `cargo incin build`  : incin-diagnostics rewrites UInt<…> → 6 / 8
        └─▶ IDE (doc 02)          : LSP rewrites the same, inline in the editor
```

## Task list

### Task 01.1 - audit which traits fail as E0271 vs E0277
Grep the compile-fail snapshots for the actual error codes:
```bash
grep -l "E0271" crates/incin-core/tests/compile_fail/*.stderr
grep -l "E0277" crates/incin-core/tests/compile_fail/*.stderr
```
Every trait whose failure shows up as **E0271** (associated-type equality) needs
the `SameCount`-style restructuring. Every trait already failing as **E0277**
just needs its `message`/`label`/`note` wording reviewed. Write the list of
affected traits into this doc as a checklist before editing - this is the map
that stops you getting lost across ~10 traits.

Known E0271 offender: `ReshapeShape` (via `ElementCount`). Likely others using
associated-type equality: check `BroadcastShape`, `ConcatShape`, `StackShape`,
the `spatial.rs` conv/pool traits (they compute output dims via associated
types).

### Task 01.2 - implement `SameCount` and reroute `ReshapeShape`
As in the crux section. Add a `compile_fail` test is unnecessary (one exists);
instead **regenerate** the existing snapshot and eyeball it:
```bash
TRYBUILD=overwrite cargo test -p incin-core --test compile_tests
git diff crates/incin-core/tests/compile_fail/reshape_static_mismatch.stderr
```
**Acceptance:** the new `.stderr` shows the `SameCount` `message` line and is
anchored at the `reshape` call site, and does **not** contain a top-level
`E0271`. Commit the regenerated snapshot.

### Task 01.3 - apply the same pattern to every E0271 offender from 01.1
One trait per commit (`fix(shapes): readable broadcast mismatch diagnostic`,
etc.). For each: introduce/reuse a reflexive witness trait, reroute the blanket
impl's equality into a bound, regenerate the affected `compile_fail` snapshot,
confirm no residual top-level `E0271`.

### Task 01.4 - copy-edit every `on_unimplemented` message for a PyTorch reader
Messages must name the operation and the mismatch in words a PyTorch user
recognizes ("element count", "inner dimension", "channel count"), never trait
jargon. Keep `{Self}`/`{Target}` interpolations (the translator handles them).
Example targets:
- matmul: "Cannot matmul: inner dimensions differ (`{Self}` vs `{Rhs}`)".
- concat: "Cannot concatenate along this axis: the other dimensions differ".
- conv2d: "Conv2d input channels must equal the kernel's input channels".

### Task 01.5 - add a `compile_fail` snapshot for each user-facing op that lacks one
Ensure every op a beginner hits early (matmul, add/broadcast, reshape, concat,
stack, conv2d, linear) has a `compile_fail` test so its message is **snapshot-
locked** and cannot silently regress. Model them on the existing files in
`crates/incin-core/tests/compile_fail/`.

## Verification
Standard loop (§2 of README) **plus**:
```bash
cargo test -p incin-core --test compile_tests    # trybuild snapshots
```
After any snapshot change, manually read the regenerated `.stderr` and confirm a
non-Rust-expert could understand it.

## Risks / DO-NOT
- **DO-NOT** delete the `ElementCount` associated types - they are still needed
  to *compute* counts; you are only changing how the *comparison* is bounded.
- **DO-NOT** assume `on_unimplemented` will print decimals. Verify each message
  by reading the real regenerated stderr; if it still shows `UInt<…>` inside the
  interpolation, that is expected - the translator/LSP handles it.
- **DO-NOT** widen any blanket impl so much that a *valid* reshape stops
  compiling. The reflexive `impl<N> SameCount<N> for N {}` must still let equal
  counts through - the existing passing tests
  (`shapes::reshape::tests::reshape_*`) are the guard; they must stay green.

## Demo script (the video)
Split screen. Left: PyTorch, a 6-layer model, one wrong `nn.Linear`, `python
train.py` → 40 s of epoch bar → `RuntimeError: mat1 and mat2 shapes cannot be
multiplied`. Right: the same bug in Incin, red squiggle in VS Code before
running, hover shows "Cannot matmul: inner dimensions differ (784 vs 128)".
Caption: *"Same bug. One of them told me before I hit run."*
