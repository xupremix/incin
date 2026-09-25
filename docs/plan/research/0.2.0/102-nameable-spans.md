# #102 follow-up: can the dynamic span be NAMED in the type system?

Research lane S2, branch `feat/custom-autograd-dtype`. Issue text:
https://github.com/xupremix/incin/issues/102 (builds on
https://github.com/xupremix/incin/issues/100).
Sketches only — no implementation. The maintainer decides.

**Question.** Top-k MoE routing gives expert `e` a `[n_e, d_model]` span where
`n_e` is data-dependent and no `S` names it. Option C sidesteps naming by
making `n_e` a row range instead of an extent. This memo asks whether the span
itself can be named — and concludes it cannot, usefully, on stable Rust — and
formalizes exactly what the offset-array formulation proves, keeps, and loses.

**Short answer.** Nobody names data-dependent extents in a static type system.
PyTorch, Megablocks, Tutel, JAX, and XLA all route around it
(offsets / padding / `Dyn` / bounded dynamism). XLA is the one system that
*appears* to name dynamic extents (`tensor<?xf32>`), and even it refuses
data-dependent cases: dynamism must trace to input arguments (shape
polymorphism); data-dependent sizes get bounded dynamism, i.e. capacity
padding. Recommendation: adopt the offset-array-as-shape formulation (option C
formalized) as the target signature, keep the `Dyn` per-expert loop (option B)
as the CPU interim behind the same public type, and stop spending design
budget on naming `n_e`.

## 1. Existential / sketched shapes — dead end as proof

Three shapes this idea takes, all investigated against the in-tree
`Tensor<S, B, K, G, P, L>` (generic over a concrete `S: Shape`) and
`Module` (whose `Output` must be a nameable associated type,
`crates/incin-core/src/nn/module.rs`):

**(a) `Box<dyn Shape>`.** `Shape` requires `Clone + Debug + Eq` (plus
`Send + Sync + 'static`), so `dyn Shape` is not even well-formed as a trait
object without stripping the supertraits, and no `impl Shape for Box<dyn …>`
exists. Even if one were written, every expert's span would share the single
type `Box<dyn Shape>` — the mechanism names nothing per-expert, it only
erases. `Tensor<S, …>` needs a `Sized` concrete `S`; an existential cannot be
a tensor's shape parameter.

**(b) `impl Shape` returns.** Return-position `impl Trait` works for a
function, but the span must cross three boundaries that all require a name:
`Module::Output` (associated-type-position `impl Trait` is still unstable),
state/checkpoint traversal (visitors are generic over concrete `S`), and
`Sequential` composition (`L2: Module<L1::Output>` — an opaque output cannot
satisfy downstream bounds). The caller also cannot store the span, compare two
spans, or render it in diagnostics.

**(c) GAT-associated extent witnesses** (e.g. an associated `type Span: Dim`
minted per routing decision). Types are chosen by the *caller* at
monomorphization; `n_e` is chosen by *data* at runtime. There is no channel
from a runtime value back into type selection on stable Rust, so a witness
type can never be minted per forward pass. Any fixed witness (e.g. one tag
per expert index `e`) names the *slot*, not the *extent* — and two tensors
sharing that tag may still hold different runtime extents.

**What actually breaks.** Not the autograd tape: the tape is value-level
(`execute_shaped` takes runtime `ShapeValue`s; descriptors are validated at
dispatch), so an existential would execute fine. What breaks is the static
frontend: (i) `Module::Output` cannot be named or composed; (ii) type equality
would claim extent equality falsely — two spans with the same hidden type and
different `n_e` compare as "equal" with no way to say otherwise (this is the
already-documented `D-013` hazard, see §5); (iii) `incin-diagnostics`/LSP
cannot render an opaque type into decimal shapes, so the failure mode is an
unreadable error at exactly the operation users debug most.

**Verdict: dead end.** Existentials erase where the issue needs naming, and
the three boundaries that need the name (`Output`, traversal, composition)
all reject opacity.

## 2. Const-generic expressions — dead end on stable; the sound approximation is option A

Threading a runtime `n_e` into `Tensor<s![{ n_e }, D], …>` needs
*dependent* types: a const parameter whose value comes from a runtime scan.
Stable Rust has no such channel, and the unstable ones are not close:

- `generic_const_exprs` is explicitly described by the lang/types teams as
  fundamentally flawed with no stabilization path; the replacement
  (`min_generic_const_args`, associated constants and parameters embedded in
  other expressions) is a nightly prototype requiring the new solver, and
  still only covers *type-level* expressions — never runtime values
  (rust-lang/goals const-generics pages; rust-lang/rust#153393 shows even
  nightly `generic_const_exprs` failing to unify a const expression with its
  value as recently as 2026).
- In-tree confirmation of the same wall: `Shape::EXTENT_BUF` exists precisely
  because "an array whose length is an associated const would need
  `generic_const_exprs`, which is unstable"
  (`crates/incin-core/src/shapes/shape.rs`).

The sound approximations:

1. **Capacity bound + proof-carrying truncation.** Fix the operand at
   `s![CAPACITY, D]`; at the boundary, `into_shape::<s![CAPACITY, D]>()`
   (`crates/incin-core/src/tensor/base/convert.rs`) validates the runtime
   dims and rejects overflow. Overflow handling is then either an error or a
   drop — i.e. this *is* option A (capacity-factor padding), with the
   truncation proof carried by the existing `try_from_dims` check. Nothing is
   gained over A, and A's silent-drop cost is why #102 rejects it as the
   default.
2. **`where`-clause const-bound designs** (`where [(); N]: …`-style tricks)
   constrain relationships *between* caller-chosen consts; they cannot admit
   a runtime value either. Dead end for the same reason.

**Verdict: dead end.** Any sound stable-Rust approximation collapses to
option A plus a runtime check, which is already specified and already
rejected as the default.

## 3. Offset-array-as-shape (option C formalized) — viable, recommended

The `[T*k]` buffer + `[E+1]` offsets formulation is nameable today, and the
parts already exist in-tree:

- `bincount::<N>` returns `Dense<DimCons<ConstDim<N>, Nil>, …>` — a
  *statically* `N`-wide histogram
  (`crates/incin-core/src/tensor/ops/manipulation/indexing.rs`), with the doc
  comment stating the design intent outright: `cumsum` over it "gives the
  offsets that describe where each expert's rows begin in a grouped buffer,
  which is how a router avoids making the per-expert token count a tensor
  extent".
- `Routing::expert_offsets` builds the `[E+1]` exclusive offsets with no host
  interop (`bincount` → inclusive `cumsum` → prepend zero by `concat`)
  (`crates/incin-core/src/nn/moe.rs`).
- `grouped_matmul(lhs [T, K], rhs [E, K, N], offsets [E+1]) → [T, N]` takes
  the ordinary `execute_shaped` path and records a gradient over `lhs`/`rhs`
  only — offsets are an integer tile with no cotangent, "the same exclusion
  `scatter_add` applies to its index operand"
  (`crates/incin-core/src/tensor/ops/manipulation/routing.rs`).
- `nonzero` is the contrast case: its output extent is unknowable before the
  scan, so it runs unshaped `dispatch::execute` and reads the shape back —
  the one place `Dyn` is *forced*, and routing never needs it, because the
  grouped buffer's `[T*k, D]` geometry is inferable from the operands alone.

Sketch of the execution (static shapes; `T`, `K`, `D`, `DFF` caller-named
static dims, `E`/`TOPK` consts — full signature in §7):

```rust
// proposed — sketch only
let routing = router.forward(x)?;                       // probs/weights [T, E]→[T, K], indices [T, K]
let perm = routing.indices.flatten().argsort()?;        // [T*K] static (MulDim product)
let buf = x.repeat_rows(TOPK).gather(&perm)?;           // [T*K, D] static
let off = routing.expert_offsets()?;                    // [E+1] static extent, NoGrad
let y = buf.grouped_matmul(&stacked_experts, &off)?;    // [T*K, DFF]→…→[T*K, D] static
let out = scatter_add_identity(&y, &perm.inverse())?;   // [T, D] static
```

Is anything genuinely lost versus naming `n_e`? Four candidates, each
checked:

| Candidate loss | Verdict |
|---|---|
| Kernel fusion granularity (one launch per expert vs one grouped launch) | Not lost. The grouped GEMM *is* the fused form: one launch, per-group tiles. This is the Megablocks trajectory (block-sparse → grouped GEMM on Hopper, `mlp_impl='grouped'`); per-expert launches (option B) are the *unfused* baseline, not the goal. |
| Bounds-check elision | Partially, honestly. The outer buffer `[T*k, D]` is fully static, so `STATIC_EXTENTS`-driven elision is retained for everything addressing the buffer. What is not statically provable is that `offsets[e]..offsets[e+1]` lies inside it — those `E+1` boundary checks stay runtime. Same position as XLA's `dynamic-slice` (runtime offsets into a static buffer). Cost is a missed elision at group boundaries, never a miscompile: the descriptor boundary still validates. |
| Error messages | Shifted, not lost. An out-of-range expert id is a runtime `ShapeError` at the `bincount`/`offsets` construction site rather than a trait-bound error. Mitigation is structural: offsets are *constructed* by `bincount`/`cumsum`/`concat`, never hand-written, so the malformed case is unrepresentable through the public path; only a corrupt index tensor (refused by `bincount`'s range check) can produce it. |
| Per-expert static diagnostics | Lost and immaterial. No per-expert `[n_e, D]` type exists to render in LSP/diagnostics — but there is also no per-expert decision a user can act on statically; the actionable static facts (buffer geometry, offsets extent `E+1`) are all present. |

**Verdict: viable — the winner.** It proves exactly what routing guarantees
(total `T*k`, `E+1` boundaries) and declines to prove what routing cannot
(the split). The per-expert loop (option B) remains the honest CPU interim
*behind the same public type* (see §7).

## 4. What SOTA does — everyone routes around naming

| System | Mechanism | Nameable in a static type system? | Cost |
|---|---|---|---|
| PyTorch nested / jagged (NJT) | Packed `values [sum(x), D]` + `offsets [B+1]`; ragged dim shown symbolically as `j1` in the shape | **No.** `j1` is identity-by-`offsets`-tensor: two NJTs with equal-but-distinct offsets get *different* symbols (`(2, j1)` vs `(2, j2)`) and binary ops refuse them. A name that is really a pointer comparison — the closest anyone comes to naming, and it still routes through offsets. (pytorch docs `torch.nested`; `nested_tensor.py`; issue #150252) | Op coverage limited to single ragged dim; data-dependent ops (`chunk` on batch) break under `torch.compile` |
| Megablocks dMoE | Sort expert ids → `histogram` → bins/offsets; `padded_gather`, block-sparse SDD/DSD or grouped GEMM, `padded_scatter` | **No.** Imbalance lives in `tokens_per_expert`/`bins` tensors, never in a type. Even the sparse path pads each expert's rows up to the 128-block multiple. (Gale et al., MLSys'23; `dmoe.py`) | Block-size padding residue; custom kernels per transpose combination |
| Tutel | Static `capacity_factor` (pad/drop); `0` = dropless minimum-that-fits; negative = capped dropless; adaptive parallelism/pipelining switching | **No.** Capacity is a runtime int (switchable per iteration); dropless mode is dynamic shapes all the way down. (Wu et al., MLSys'23; `tutel/examples/helloworld_switch.py`) | Static mode drops tokens or wastes compute; dropless mode recompiles/pads per step |
| JAX (`jit`) | Shapes must be static at trace time; `bincount` needs a static `length` (over-long inputs *dropped*); ragged data = `(data, offsets, sizes)` triple; data-dependent ops untraceable | **No** — the strictest datapoint: JAX refuses to trace what it cannot name statically, and its ragged story is again offsets. (`jax.numpy.bincount` docs; `lax.ragged_all_to_all`) | Recompilation per shape; padding discipline pushed to the user |
| XLA / StableHLO | `tensor<?xf32>` dynamic dims; bounded `f32[<=5,5]`; `dynamic-slice`/`dynamic-update-slice` take runtime offsets; `set-dimension-size` carries the dynamic size as a *value operand* | **Only input-traced dynamism.** Shape polymorphism requires all dynamism to trace to input arguments; *data-dependent* sizes (canonical example: `nonzero`) get bounded dynamism = padded-to-upper-bound execution, with `PadToStatic` + runtime size metadata. Dynamic *offsets* are values, never types. (StableHLO dynamism docs; `dynamic_dimension_inference.cc`) | Padded execution (kernels sized to the bound); bounded support experimental, GPU-fragile |

**Extract.** The industry converged twice: (i) the dynamic partition is
carried as an *offsets value*, never as a *type*; (ii) when a static name is
unavoidable, the price is padding to a bound (capacity factor / bounded
dynamism / block-size round-up). XLA's `?` dims look like a counterexample
until you read the constraint: they name *caller-provided* dynamism, exactly
the class stable Rust const-generics can already express. Data-dependent
extents are unnamed everywhere.

## 5. Rust type-system limits — the marker sketch and why it proves nothing

Three sub-questions from the directive:

- **Specialization.** Irrelevant to naming. Specialization selects *impls* by
  type; it does not move runtime values into types. (And `min_specialization`
  remains incomplete.) At most it could optimize `Dyn`-vs-static dispatch
  later — an execution concern, not a typing one.
- **`generic_const_exprs` / typenum arithmetic.** Per §2, `generic_const_exprs`
  has no stabilization path. Typenum arithmetic on `Dim`s works and is
  already in-tree (`MulDim`, `AddDim`, `BroadcastExtent`, `broadcast_static`
  in `crates/incin-core/src/shapes/dim.rs`; `MatMulShape` computing `Output`
  from two inputs in `crates/incin-core/src/tensor/matmul.rs`) — but it is
  strictly static→static. `s![T, K]` flattening to a `MulDim`-based `[T*K]`
  extent for the grouped buffer is legitimate and needed (§7); deriving any
  of it from `n_e` is not.
- **`DimCons` carrying a `Dynamic`-but-equal marker.** Sketch, since the
  directive asks for one if viable:

```rust
// proposed — sketch only: a span marker with NO static extent,
// so it can never be mistaken for a proof
pub struct Span<Tag>(core::marker::PhantomData<Tag>);

impl<Tag> Dim for Span<Tag> {
    type KeepDim = typenum::U1;
    const STATIC: StaticExtent = StaticExtent::RuntimeUnknown; // honest: unknown
    type Arg = usize;                                          // value rides in ShapeBuf
    // resolve_arg accepts any usize — same as the existing `usize` Dim
}
// "same tag ⇒ same extent" would need DimCompatible to compare values it
// does not own; the blanket `impl<L: Dim, R: Dim> DimCompatible<R> for L`
// already declines exactly this (dim.rs), leaving enforcement to ShapeBuf
// comparison at the boundary (the D-013 checked_broadcast_dim precedent).
```

**Verdict: viable as documentation, dead end as proof.** `NamedDim<Tag, usize>`
already *is* this design, and the codebase already records why a shared tag
proves nothing: "a `symbolic_dim!` name, unlike `typenum`, can legitimately
hold a *different* runtime value on each operand even when both share the
exact same type" (`broadcast.rs`, `checked_broadcast_dim` docs). A fresh
`SpanTag` per expert *slot* is a fine diagnostic aid (it says *which*
partition a dim belongs to), but the compiler cannot mint a fresh tag per
forward pass and cannot distinguish two different spans sharing a tag. All
enforcement stays where it already is: runtime `ShapeBuf` checks at the
descriptor boundary. Adopt the marker only if diagnostics wants it; do not
present it as naming `n_e`.

## 6. Auxiliary loss and gradient contract (defaults, per maintainer direction)

- **Aux loss lives in `Module::Output` as a tuple** (maintainer decision —
  the only placement that cannot be silently forgotten, and it forces the
  signature question now since it is breaking later). Scalar loss tensor in
  the rank-0 `Nil` shape (the shape `BroadcastShape<Nil> for Nil` already
  treats as scalar).
- **`E` const, `k` (`TOPK`) const.** `E` determines parameter count and hence
  state layout: experts are visited as `experts.0 … experts.E-1` with stable
  `StatePath`s, so a checkpoint with a different `E` is refused rather than
  partially loaded. `TOPK` const keeps the `[T, K]` routing intermediates
  static (and matches the existing const-generic `one_hot::<E>` /
  `bincount::<E>` / `Router<E, TOPK, …>` in `nn/moe.rs`).
- **Gate-weight-only gradients (default).** Discrete selection is
  non-differentiable; no straight-through (it would fabricate a gradient
  through `argmax`). The tape already implements this partition and the
  signature must preserve it by type: indices/permutation/offsets/counts are
  `NoGrad` (`nonzero`, `one_hot`, `bincount` record nothing; `grouped_matmul`
  excludes offsets from the cotangent, same as `scatter_add`'s index
  operand), while gate `weights`/`probs` carry the joined grad so both the
  output path and the aux-loss path reach the router gate. Determinism note:
  the combine accumulates `TOPK` contributions per token in construction
  order — the `scatter_add` determinism guarantee from #103 is part of this
  contract, not adjacent to it.

## 7. RECOMMENDATION — signature sketch for `MoE::forward` under the winner

Winner: **§3 (offset-array-as-shape, option C formalized)**; option B stays
as the CPU interim behind the identical public type; option A remains an
explicit bounded-compute configuration, never the default.

```rust
// proposed — sketch only. S: caller-named static input shape whose trailing
// axis is D_MODEL (EndsWith), e.g. s![T, D]; JG = JoinedGrad<G, Train::TensorGrad>.
impl<const E: usize, const TOPK: usize, Expert, B, K, Train, G, S, L>
    Module<Tensor<S, B, K, G, Local, L>> for MoE<E, TOPK, Expert, B, K, Train>
where
    S: Shape + DynShape + EndsWith<DModel>,   // input [..., D_MODEL], fully static
    Expert: Module<Tensor<BufOut, B, K, JG, Local>, Output = Dense<BufOut, B, K, JG, Local>>,
    // BufOut = the [T*K, D]-family static buffer shape (MulDim product; cf. MatMulShape precedent)
    B: MoEBackend<K> + Execute<op::GroupedMatMul> + Execute<op::ScatterAdd> + Execute<op::Gather>,
    // ... plus #100's BroadcastCompatible bounds for the gate-weight application (see §8)
{
    // (combined output [..., D_MODEL], load-balancing aux loss scalar).
    // Aux loss tuple placement is per maintainer decision; E/TOPK const and
    // gate-weight-only gradients are the defaults (§6).
    type Output = (Dense<S, B, K, JG, Local>, Tensor<Nil, B, K, JG, Local>);
    type Error = Error;

    fn forward(&self, x: Tensor<S, B, K, G, Local, L>) -> Result<Self::Output, Error>;
}
```

Conformance notes (all follow-on work, none of it naming `n_e`): the interim
B implementation returns the same `(output, aux)` type with the aux computed
from the dense masked path (one-hot weights + counts — same values the
offsets path uses, so the interim/target switch is execution-only, as #102
requires); experts stay `[Expert; E]` so state traversal is unchanged; the
aux scalar's gradient reaches the gate through `probs` (differentiable) while
`counts`/`offsets` stay `NoGrad` by type.

## 8. What #100 must deliver for the winner to work

The winner needs three shape-pair mechanisms; the first two are squarely
#100's sequencing ("routing produces shape pairs the reflexive `ShapeEq`
cannot express"):

1. **`BroadcastCompatible<Other>` with associated `Output`** (or equivalent),
   with impls covering the gate-weight application pairs: `[T,K,E]×[T,K,1]`
   (one-hot mask × unsqueezed weights) and `[T,D]×[T,1]` (expert output ×
   narrowed weight column). Trailing-unit-axis broadcast with a *computed*
   static output type — the `MatMulShape` precedent applied to elementwise
   ops. Without this, the static-shape forward cannot multiply gate weights
   into expert outputs.
2. **A typed gather/scatter/index bound naming the output extent from the
   index shape** — the `[T,K]`-indices-into-`[T,D]`-activations pair
   (permutation gather into `[T*K, D]`, inverse-permutation scatter back to
   `[T, D]`). This is *not* broadcast and is not covered by (1); if #100's
   mechanism does not extend to index pairs, that bound is a named follow-on
   that must land with the MoE implementation, not after it.
3. **Out of #100's scope but required:** a static-shape `grouped_matmul`
   overload (today `grouped_matmul` returns `Dense<Dyn, …>`; its `[T, N]`
   output is inferable from the operands, so a `MatMulShape`-style static
   rule applies — implementation work, no design question) and the CUDA
   grouped GEMM itself (#85 → #103).

If #100 lands (1) scoped to `where_cond`/`masked_fill` only (its recommended
narrow scope), the MoE work must widen the same trait to the arithmetic pairs
above — additive, but it must be scheduled, not assumed.

## 9. Genuinely open (for the maintainer)

- Tuple `Output` composition with #101: `Sequential` requires
  `L2: Module<L1::Output>`; a `TransformerDecoderLayer` after an MoE layer
  either threads `(tensor, aux)` through every downstream `Module` impl or
  the block boundary splits the aux off. Both touch the `Module` surface #101
  shares — needs a ruling, not research.
- Whether the `Span<Tag>` marker (§5) is worth its weight as a diagnostic aid
  once nothing rides on it.
- Compiled-execution capture (`incin::experimental::compiled`) of the
  computed buffer/offsets shapes — same check #100 already owes its own
  computed outputs.
- Expert-parallel sharding (#99) over the signature: placement over mesh axes
  should be unaffected (experts remain `[Expert; E]`), but unexamined here.
