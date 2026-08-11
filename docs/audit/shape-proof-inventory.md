# Shape, dtype, backend, and device proof inventory

**Task:** `SHP-001` · **Status:** complete · **Snapshot:** 2026-07-27
**Regenerate:** `tools/audit-shapes.sh --update`
**Verify:** `tools/audit-shapes.sh --check`

This document is the baseline for the shape workstream (`SHP-002`…`SHP-008`). It
classifies every shape, dtype, backend, and device rule by the stage at which it
is actually proven today, and inventories the sites where a rule that the design
in [PROPOSALS.md](../../PROPOSALS.md) §1.2 wants proven statically or once at
lowering is instead re-checked — or not checked at all — at runtime.

It is deliberately a *measurement*, not a fix. No behavior changes with this
task. Every defect below is assigned to the ledger task that closes it, and the
counts in the generated block are the numbers those tasks move.

## 1. Proof stages

`PROPOSALS.md` §1.2.2 divides validation into type, lowering, resource-binding,
and native-launch layers. This audit uses those four, plus a fifth bucket for
rules that currently reach none of them.

| Stage | Name | Meaning |
|---|---|---|
| **T** | Type | Proven by trait resolution. An illegal program does not compile. |
| **L** | Lowering | Proven once, when an operation is resolved into a descriptor. |
| **B** | Binding | Proven when a resource is constructed, imported, or bound to a plan. |
| **N** | Native | Not re-proven; the executor trusts a sealed descriptor. |
| **U** | Unproven | Re-checked ad hoc at runtime, or rejected only when the backend fails. |

**T is not a hardware claim.** `CudaN<U0>` proves that the CUDA backend at
ordinal 0 was *selected*, not that CUDA device 0 exists on the machine that runs
the binary. Every physical-resource fact is inherently stage **B** or later; no
amount of trait work moves it earlier.

## 2. Rule inventory

Rules are grouped by the family they constrain. "Today" is the earliest stage at
which the rule is actually enforced in this tree; "Target" is the stage
`PROPOSALS.md` assigns it. Rows where the two differ are the workstream.

### 2.1 Shape rules

| Rule | Today | Target | Gap | Owner |
|---|---|---|---|---|
| Rank compatibility, static dims | T | T | none | — |
| Static dimension equality | T | T | none | — |
| Named/runtime dimension equality | U | L | Re-derived per call site; no descriptor to seal. | `EXE-001` |
| Broadcast output arithmetic | **L** | L | closed by `SHP-004`; see §3.2. | — |
| Reshape element-count equality | T | T | Proven only to rank 4; see §4. | `SHP-006` |
| Flatten output arithmetic | **L** | L | closed by `SHP-004`; same chain as broadcast. | — |
| Conv/pool spatial geometry | **L** | L | closed by `SHP-005`; see §3.3. | — |
| Overflow in `numel`/byte length | **L** | L | closed by `SHP-003` for `ShapeBuf`; the storages migrate under `EXE-004`. | `EXE-004` |

### 2.2 Dtype, backend, and device rules

| Rule | Today | Target | Gap | Owner |
|---|---|---|---|---|
| Backend family selected | T | T | none | — |
| Logical device ordinal selected | T | T | Static only for `ConstDevice`; `Cuda`/`Wgpu` are **B**. | — |
| Static dtype/backend legality | **U** | T | `SupportsDType` proves nothing; see §3.1. | `EXE-005` |
| Operation interface available | U | T | No `Execute<O>` bound exists yet. | `EXE-006` |
| Inputs share required device | U | L | Checked per op when the ordinal is stored. | `EXE-001` |
| Physical device/adapter exists | B | B | none — cannot be earlier. | — |
| Actual device supports dtype/op | B | B | none — cannot be earlier. | — |
| Storage matches its metadata | B | B | none | — |
| Byte bounds and offset | B | B | none | — |
| Pointer alignment | B | B | none | — |
| Driver/library compatibility | B | B | none | — |

## 3. Findings

### 3.1 `SupportsDType` proves nothing (stage U, claimed T)

[`crates/incin-core/src/tensor/backend.rs:61`](../../crates/incin-core/src/tensor/backend.rs#L61)

```rust
pub trait SupportsDType<K: DType> {
    fn resolve_dtype(field: &K::Field, _device: &DeviceId) -> Result<DTypeId> {
        Ok(K::to_incin(field))
    }
}
```

The sole method has a blanket default body that forwards to `K::to_incin` and
returns `Ok` unconditionally. No implementor has to reject anything, and none
does. Every unsupported dtype is therefore discovered when the backend fails,
not when the program is compiled. This is the single largest static-selector gap
in the tree: the row "static dtype/backend legality" in the `PROPOSALS.md` proof
table describes a target contract, not current behavior.

Closed by `EXE-005`/`EXE-006`, which make dtype/operation rejection a
trait-resolution failure. `SHP-001` only records it.

### 3.2 The `from_dyn().unwrap()` chain — closed by `SHP-004`

**Status: fixed.** The account below is the finding as `SHP-001` recorded it.
`SHP-004` made `BroadcastShape::output_shape` and `MatMulShape::output_shape`
fallible, added `field_from_dims` as the checked replacement for the round trip,
and converted `checked_broadcast_dim` from an `assert!` to a `Result` per
decision `D-013`. All 39 live sites are gone; the `shapes` surface went from 13
`unwrap`s to 1 and from one `panic!`-class site to none.

Three latent wrong answers surfaced during the conversion, each worse than the
panic it sat next to because it propagated silently:

* broadcast resolved a compatible pair with `lhs.max(rhs)`, so a size-1 axis
  against a size-**0** one produced 1. An axis with no elements cannot gain one
  by being broadcast against; the rule is "take the side that is not 1".
* `Dyn` matmul returned `vec![]` — the *scalar* shape — as a fallthrough for
  every rank combination it did not recognize, including `[m, k] × [k]`, whose
  answer is `[m]`.
* no `MatMulShape` impl checked the contracted dimension at all. `output_shape`
  returned `(lhs.0, rhs.1)` without ever reading `K`, so a disagreement yielded
  a confidently wrong output shape. It is now a `DimensionMismatch` on axis
  `'k'` — the same `D-013` argument as broadcast, since a `dim!` name can carry
  different runtime sizes on each operand.

The `unreachable!` noted below is also gone: a missing axis on the shorter
operand is now an implicit 1, which is what NumPy right-alignment means, so the
fourth match arm no longer exists rather than being asserted away.

#### The finding as recorded


Every `BroadcastShape::output_shape` impl computes dimensions into a `Vec`,
then reconstructs the typed field from that `Vec` and unwraps:

```rust
<Self::Output as Shape>::from_dyn(&broadcast_dims::<Self, (…)>(lhs, rhs)).unwrap()
```

The type system has already fixed `Self::Output`, so the rank is known; the code
nonetheless erases it to a `Vec`, re-parses, and panics if the round-trip fails.
The unwrap is a proof obligation that no type states and no test covers. The
generated block counts these sites; `SHP-004` drives the count to zero by
computing the typed field directly.

Related: [`broadcast.rs:60`](../../crates/incin-core/src/shapes/broadcast.rs#L60)
carries `unreachable!("out_rank is the max of both operands' ranks")`. The
invariant is genuine, but it is asserted in a comment rather than established by
construction.

### 3.3 Conv and pool geometry zeroed runtime dimensions — closed by `SHP-005`

**Status: fixed.** The account below is the finding as `SHP-001` recorded it.
`SHP-005` replaced both defects with the named checked sequence
`spatial_out_size`, which rejects a zero stride, kernel, or dilation by name,
reports a kernel that does not fit as `EmptyOutput` instead of underflowing,
and names each overflowing term individually. The reproduction is now
[`tests/spatial_geometry.rs::pool2d_computes_runtime_spatial_dims`](../../crates/incin-core/tests/spatial_geometry.rs).
The generated block records both chains at 0.

`SHP-005` also closed two silent rank failures found while fixing this: the
`Dyn` conv and pool rules tested `len() == 4` (or `== 3`) and returned the
*input* shape unchanged for every other rank, so a rank-3 `(C, H, W)` dynamic
tensor was never pooled at all and an unsupported rank was never reported.


[`crates/incin-core/src/shapes/spatial.rs:154`](../../crates/incin-core/src/shapes/spatial.rs#L154)
and [`:221`](../../crates/incin-core/src/shapes/spatial.rs#L221):

```rust
($(input.$idx.clone(),)* COut::from_size(out_channels).unwrap(), Default::default())
```

Two distinct defects share this line:

1. `COut::from_size(out_channels).unwrap()` panics rather than reporting a
   channel count that does not fit the target dim type.
2. The spatial extents use `Default::default()`. For a `typenum` dim that is the
   correct static value, but for a `usize` or `symbolic_dim!` extent
   `Default::default()` is **0** — the output shape silently claims a zero-sized
   spatial dimension instead of the computed one.

**Confirmed by execution**, not just by reading. A pooling call on shape
`(U1, U1, usize, usize)` with a runtime 8×8 input, kernel 2, stride 2, padding
0, dilation 1 returns spatial dims `(0, 0)` where `(4, 4)` is correct:

```
pool2d 8x8 k2 s2 p0 d1 -> (0, 0)
assertion `left == right` failed: spatial dims were zeroed
  left: (0, 0)
 right: (4, 4)
```

This is a wrong answer, not a panic: the bad shape propagates. It is the
highest-severity item in this audit. The in-tree comment shows the channel half
was already found and fixed once; the spatial half is still open.

`SHP-005` replaces both with a named, checked sequence and rejects stride 0, and
should carry this case as a regression test. Note that the traits involved
(`Pool2dShape`, `SpatialConv2d`) are `pub` inside a `pub(crate) mod shapes`, so
the reproduction must live in-crate — an integration test under
`crates/incin-core/tests/` cannot reach them.

### 3.4 Operator panics are structural, not accidental

The eight `panic!` sites under `crates/incin-core/src/tensor/ops/` are all inside
`std::ops` impls (`Add`, `Mul`, and scalar variants), whose signatures return
`Self::Output` and cannot return `Result`. They already name the fallible method
to call instead:

> `operands were not broadcast-compatible at runtime; call `.add()` directly
> instead of the operator to handle this as a `Result``

These are counted separately from the `.unwrap()` chains because they are a
deliberate, documented API boundary rather than an unstated obligation. They
stay panicking; what changes is that under the compiled path the lowering layer
proves compatibility first, so the branch becomes dead rather than merely
unlikely. No ledger task removes them.

## 4. Rank coverage

`Shape` is implemented for tuples up to rank 8. Rules implemented below that
ceiling define ranks at which a tensor type is expressible but its operations
cannot resolve — the frontend accepts the shape and then has no proof to offer.

The RFC names `ElementCount` (rank 4) versus `Shape` (rank 8) as the motivating
case. `SHP-006` replaces the eighteen hand-written invocation ladders with one
generator: `MAX_RANK` lives in
[`incin-macros/src/rank.rs`](../../crates/incin-macros/src/rank.rs) and
`rank_sweep!` expands each rule's ladder from it, so raising the ceiling is a
one-line change and no rule can silently drift below it. `incin-core`
re-exports the value as `shapes::MAX_RANK`; a proc-macro crate cannot export a
`const`, and a second copy would reintroduce the drift.

**"N short" is not always a defect.** A rule whose `Output` gains an axis must
stop one rank below the ceiling, because that `Output` is bounded by `Shape`
and no tuple above `MAX_RANK` implements `Shape`. `AppendDim` at 7 and
`StackShape` at 7 are therefore *correct*, not gaps — the earlier reading of
this table as "14 rules short" counted them as defects. The real target is:

| Rule kind | Correct ceiling | Rules |
|---|---|---|
| Rank-preserving | `MAX_RANK` (8) | `Shape`, `ConstShape`, `DynShape`, `PartialDynShape`, `EndsWith`, `ReplaceLastDim`, `ElementCount`, `BroadcastShape`, `ReshapeTarget`, `SliceTarget`, `ConcatShape`, `HasChannels1D`, `HasChannels2D`, `Pool2dShape`, `AdaptiveAvgPool2dShape`, `SpatialConv1d`, `SpatialConv2d` |
| Rank-increasing | `MAX_RANK - 1` (7) | `AppendDim`, `StackShape` |

### 4.1 Migration status — complete

All 19 rules now sit at their correct ceiling: 17 rank-preserving rules at
`MAX_RANK`, and the two that *add* an axis at `MAX_RANK - 1`. Nothing sits
above it.

| Rule | Was | Now | How |
|---|---:|---:|---|
| `ReplaceLastDim` | 12 | **8** | four arms above rank 8 deleted — no tuple that wide implements `Shape`, so they could never be selected |
| `EndsWith` | 6 | **8** | six enumerated arms replaced by one variadic arm |
| `HasChannels1D` | 3 | **8** | was a *single* arm at rank 3, so `(C, L)` — named as valid by the trait's own docs — did not implement it |
| `HasChannels2D` | 4 | **8** | same, and `(C, H, W)` likewise did not |
| `ReshapeTarget`, `SliceTarget` | 4 | **8** | already variadic; simply never invoked above rank 4 |
| `ElementCount` | 4 | **8** | the RFC's motivating case |
| `SpatialConv1d`, `SpatialConv2d` | 7 | **8** | ladder generated with the batch-count offset declared |
| `Pool2dShape`, `AdaptiveAvgPool2dShape` | 4 | **8** | single hand-written rank-4 impls replaced by batch-variadic macros |
| `BroadcastShape` | 4 | **8** | eight macro families, two of them two-dimensional rank-pair sweeps |
| `ConcatShape` | 6 | **8** | 21 hand-written per-axis impls replaced by one macro over a rank × axis sweep |
| `StackShape` | 6 | **7** | 27 hand-written impls likewise; rank-increasing, so 7 is its ceiling |
| `AppendDim` | 7 | 7 | already correct; now declares `max = 7` explicitly |

Two of these were holes at the *bottom* rather than missing ranks at the top:
`HasChannels1D` and `HasChannels2D` each held for exactly one rank, so the
unbatched forms their own documentation names — `(C, L)` and `(C, H, W)` — did
not implement them.

`ElementCount` is the one family that stayed hand-written, and the reason is
specific rather than expedient: `rank_sweep!` varies a *parameter list*, but
each `ElementCount` rank needs a differently-nested `Prod` fold in both the
associated type and every intermediate `where` bound. Emitting that needs a
fold over type expressions. Four explicit impls are cheaper to read than the
macro that would generate them.

### 4.2 A diagnostic cost worth recording

Raising a marker trait from one impl to eight changes what `rustc` prints. With
a single impl it could say *"but trait `HasChannels2D<U40>` is implemented for
it — expected `UTerm`, found `UInt<UTerm, B1>`"*, pinpointing the channel count
that disagreed. With eight candidates it falls back to listing them. The
`#[diagnostic::on_unimplemented]` note still carries the guidance, and the
compile failure itself is unchanged — but the error is less specific than it
was, and that is a real cost of the rank coverage, not a neutral change.

Eleven `compile_fail` expectations were regenerated as a result. Every case
still fails to compile; the deltas are new ranks appearing in impl lists, one
parameter rename (`D0_` → `Rhs` in the concat macro), and one *improvement*
where a prelude re-export shortened `incin::shapes::reshape::SameCount` to
`SameCount`.

## 5. Generated inventory

Everything below is produced by `tools/audit-shapes.sh` and must not be edited by
hand. `--check` fails if it drifts from the tree, so these counts stay honest as
the workstream lands. Counts exclude `#[cfg(test)]` modules: this audit measures
obligations the library imposes on callers, not ones its tests impose on
themselves.

<!-- BEGIN GENERATED: audit-shapes -->

### Panic-class sites by rule surface

| Rule surface | `unwrap` | `expect` | `panic!`-class | `assert!` |
|---|---:|---:|---:|---:|
| `shapes` | 0 | 1 | 2 | 1 |
| `tensor` | 0 | 4 | 0 | 9 |
| `backend` | 0 | 0 | 1 | 0 |

### Named chains with a required terminal count of zero

| Chain | Owner | Sites |
|---|---|---:|
| `from_dyn().unwrap()` | SHP-004 | 0 |
| `from_size().unwrap()` | SHP-005 | 0 |
| `Default::default()` | SHP-005 | 0 |

### Rank ceiling by shape rule

| Rule | Max rank | vs `Shape` |
|---|---:|---|
| `AppendDim` | 0 | aligned |
| `BroadcastShape` | 0 | aligned |
| `DynShape` | 0 | aligned |
| `ElementCount` | 0 | aligned |
| `PartialDynShape` | 0 | aligned |
| `Shape` | 0 | aligned |

<!-- END GENERATED: audit-shapes -->

## 6. Exit criteria

`SHP-001` is complete when the inventory exists and is mechanically verifiable —
both true as of this snapshot. The counts above are the baseline that later tasks
reduce:

| Metric | Baseline | Now | Target | Owner |
|---|---:|---:|---:|---|
| `from_dyn().unwrap()` sites | 28 → **39** | **0** | 0 | `SHP-004` ✅ |
| `from_size().unwrap()` sites | 2 | **0** | 0 | `SHP-005` ✅ |
| `Default::default()` in spatial output | 4 | **0** | 0 | `SHP-005` ✅ |
| Rank-preserving rules short of `MAX_RANK` | 14 | **0** | 0 | `SHP-006` ✅ |
| Rules over `MAX_RANK` | 1 | **0** | 0 | `SHP-006` ✅ |
| `SupportsDType` rejections possible | no | no | yes | `EXE-005` |

The `unwrap` count for the `shapes` surface fell from 13 to **1** over this
period: two `from_size().unwrap()` sites under `SHP-005`, then the ten
`from_dyn().unwrap()` sites under `SHP-004`. The single survivor is
[`idx.rs:143`](../../crates/incin-core/src/shapes/idx.rs#L143), a
`size.unwrap()` guarded by an `is_none()` test on the line above — safe, but
expressible as a `match` — and it sits in a reshape path that also returns
`vec![]` as a sentinel for "more than one inferred dimension". Both belong to
`SHP-007`'s sweep of the remaining mixed and dynamic gaps; neither is part of a
named chain.

**The `from_dyn` baseline was wrong.** `SHP-001` counted these with
`from_dyn\([^)]*\)\s*\.unwrap\(\)`, which stops at the first inner `)` and so
never matched the most common form in this tree —
`from_dyn(&broadcast_dims::<Self, (A, B)>(lhs, rhs)).unwrap()`. Fourteen sites
were invisible. The chain is now counted by balancing parentheses, and the true
figure is 39 live sites across 11 files (42 including `#[cfg(test)]` modules,
which this audit excludes by design). The target is unchanged at 0; the
distance to it was understated by a third.

Re-run `tools/audit-shapes.sh --check` in CI once `GOV-005` lands the budget
gate; until then it is a local and review-time check.
