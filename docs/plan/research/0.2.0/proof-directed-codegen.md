proof-directed codegen: the evidence is delivered and nobody opens it

Finding: `ShapeEvidence` crosses the backend boundary on every validated
descriptor, carries exactly the facts a kernel specializer wants, and has zero
readers in `incin-backends`.

The chain is complete and already built. `Shape::PROOF` derives a `ProofLevel`
from the shape *type*. `exec::catalog::validated` calls `ShapeEvidence::of::<S>()`
where the typed frontend still has `S` in hand, capturing `proof`, `static_rank`
and `static_numel`. `Validated::shape_evidence()` is a `pub const fn`. Every
backend `Execute<O>` receives `&Validated<O>` inside its `ExecutionRequest`. So a
backend can ask what the frontend proved, at no runtime cost, today.

`grep -rn ShapeEvidence crates/` returns three hits outside its defining file:
one re-export and two construction sites. `grep -rn ProofLevel
crates/incin-backends/` returns nothing at all.

This is not an oversight in the design, which states the intent plainly in
`exec/proof.rs`: "A shape proved entirely at compile time and one checked at
runtime a microsecond ago are both valid, but they justify different amounts of
work: the first can specialize a kernel on constants, the second cannot.
`ProofLevel` records which happened, and travels with the descriptor so a backend
can act on it." No backend acts on it. The road is built and nothing drives on
it.

Why this is the interesting direction rather than importing more compiler
technique. The `codegen` adoption that just landed applied one textbook idea --
SSA linearization with CSE by structural memoisation -- and that was the right
first move, but it is what every compiler does. The lowering literature's hard
problem is *recovering* static structure: TVM, Triton and XLA all specialise on
runtime shapes and recompile per shape bucket, because by the time they see a
tensor its geometry is a runtime fact. Incin's frontend knows the answer from the
type and is already shipping it across the boundary. That asymmetry is the thing
worth exploiting, and it is unavailable to a Python-fronted framework by
construction.

What the current fields buy, concretely, in the pointwise path.

`static_numel` makes the element count a compile-time constant. Two consequences.
The packed templates in `kernel::packed` emit a ragged-tail `else` branch for
`numel % width != 0`; when the numel is a static multiple of the vector width
that branch is provably dead and can be omitted, which removes a divergent branch
and shrinks the kernel. And `numel` stops being a kernel parameter and becomes a
literal, which lets the bounds check fold away entirely in the dense case.

`static_rank` makes the strided template's trip count known. That template
currently runs `for (int i = ndim - 1; i >= 0; i--)` with a runtime `%` and `/`
per axis per element -- integer division is among the most expensive operations
on the device -- and `launch_unary_body` uploads two `int` buffers per launch to
feed it. A known rank fully unrolls the loop.

What the current fields do *not* buy, and the natural extension. `ShapeEvidence`
carries rank and numel but not per-axis extents, so the strides themselves cannot
be folded to literals yet. That is the larger win: with literal extents nvcc
turns each `%` and `/` by a constant into a multiply-shift, and both `clone_htod`
uploads and both pointer parameters disappear. Adding static extents to
`ShapeEvidence` is the obvious follow-up, and it should be weighed against kernel
cache pressure, since specialising on exact extents rather than on
`ShapeBucket` multiplies the number of distinct compiled kernels.

Note that `KernelKey` already has `rank_class`, `shape_bucket` and `alignment`
fields, so the cache is prepared to distinguish specialisations. They are
currently populated from runtime metadata; a proof-directed path would populate
them from evidence and record which, so a statically specialised kernel cannot be
served to a dynamically shaped call.

The plumbing gap. `Execute<O>` has the `Validated`, but the pointwise launchers
in `cuda::ops::elementwise` receive only `CudaStorage`, whose `TensorMeta` is
entirely runtime. So evidence has to be threaded from the executor down to the
launcher and into `KernelKey`. That is the actual work, and it is the same shape
as the `ScalarFragment` change that just landed: add a parameter carrying the
richer fact, keep the existing entry points as wrappers that pass the "nothing
proven" value, and specialise only where the proof is present.

One caveat worth stating before anyone builds on this. `exec/proof.rs` notes that
`dispatch::execute` "is generic over the operation and the backend but not over
the operand shapes, so it has no `S` to read `Shape::PROOF` from and passes
`Dynamic` for everything". So the dynamic dispatch path will never carry a useful
proof, and any specialisation must be a pure optimisation over a correct dynamic
fallback rather than a path anything depends on. That is the right shape for it
anyway, but it means the win is only available to callers who came through the
typed frontend -- which should be measured before the extension is scoped, since
it bounds how much of a real workload can benefit.

Recommendation: adopt this the way codegen was adopted -- one operation family,
end to end, on a machine that can run it. CUDA pointwise again, using
`static_numel` to elide the packed tail branch, because that is the smallest
change with a result you can see in the emitted source and measure on the device.
If it lands, extending `ShapeEvidence` with static extents becomes the argument
for the rest.

**Update: that slice has landed.** `KernelSpecialization` is the backend-side
projection of the evidence; `Execute<op::*>` for the canonical unary family
builds one from `request.operation.shape_evidence()` and threads it to the
renderers, which drop the packed ragged tail when the count is a proven multiple
of the vector width. Absent or non-`Static` evidence specializes nothing.

Two things the implementation clarified that this note had guessed at.

The provenance seal is tighter than expected, and usefully so. `ShapeEvidence::of`
is `pub(crate)` to `incin-core`, so a backend cannot construct evidence at all --
it can only receive what a `Validated` carries. That means the test that a real
static shape type yields a usable count has to live in `incin-core`, and the
backend can only test the absent case. That is the seal working: there is no way
to write a backend test that fakes a proof, which is exactly the property
`exec::proof` was built for.

And the grid-overhang guard is not eliminable. `if (base >= numel) return;` looks
like it should fall to the same proof, but the launch rounds up to whole blocks
regardless of how well-known the element count is, so the overhang is a property
of the launch geometry rather than of the shape. Only the *ragged tail within a
packet* is what the divisibility proof settles. Worth stating because the two
look alike in the source.

Next along this line, in order of value: extend `ShapeEvidence` with static
extents so strides fold to literals and the two `clone_htod` uploads disappear;
apply the same projection to the binary family and to reductions; and measure
what fraction of a real workload arrives through the typed frontend at all,
since `dispatch::execute` reports `Dynamic` and bounds the reachable win.

Depends on: the `ScalarFragment` seam in `codegen-adoption-landed.md`, which is
where a specialised body would be emitted.
