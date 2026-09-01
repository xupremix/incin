CUDA: the gap is verification, not kernels (recount, 2026-09-01)

Finding: the #105 headline "CUDA is missing 88 operations against CPU's 158" is
stale by a wide margin. Recounted from the generated `docs/capabilities.md` at
this commit: CPU advertises 170 distinct operations, CUDA 165, WGPU 71, Metal 41.
CUDA's entire remaining gap is five rows: `log_softmax`, `logsumexp_dim`,
`logsumexp_keepdim`, `scatter_add`, and the coarse `normalization` family row.
Four of those five are the operations added to CPU in this same working tree, so
CUDA was level with CPU immediately before this branch. WGPU is missing 99 and
Metal 129, so those two issues (#91, #92) keep their scale; the CUDA breadth
issues do not.

Advertisement is backed by structure here, not by assertion.
`crates/incin-backends/src/cuda/executor.rs:2341` defines
`assert_every_advertised_cuda_row_executes!`, and the capability declaration
generates a compile-time `Execute<O>` obligation, so an advertised CUDA row
cannot be a missing impl. The pointwise, comparison, logical, reduction, scalar,
index-reduction, binary-math and loss families are all generated through macros
(`impl_cuda_canonical`, `impl_cuda_binary_math`, `cuda_loss_executors`, and
seven more), which is why an operation like `atan2` or `erf` has no file of its
own and is still real.

What none of that establishes is that any of it runs. `.github/workflows/ci.yml`
has one CUDA job, `cuda-compile`, and it is `cargo check` only. Nine CUDA test
files exist in `crates/incin-backends/tests/` (`cuda_executor.rs`,
`cuda_reduce_ops.rs`, `cuda_optimizer.rs`, `cuda_embedding.rs`,
`cuda_shape_dtypes.rs`, and the NCCL and collective suites) and none of them has
ever been executed by CI. `docs/capabilities.md` says in its own header that a
row is "a canonical capability decision, not a claim about a machine". For CUDA
that sentence is still entirely true.

Recommendation: on first contact with a CUDA machine, run the nine existing
suites before writing anything new. They are the cheapest possible source of
real findings, because they encode expectations nobody has ever checked. Only
after they are green is it worth porting the conformance oracle
(`crates/incin-backends/src/conformance/`) to CUDA, which is the other half of
#83 and the thing that turns 165 advertised rows into 165 verified ones. The
oracle is backend-parameterised at the `advertised_tuples(DeviceKind)` level
already; what is CPU-specific is the fixture operand builder and the
`run_cpu_self_check` entry point.

Risk: the five real gaps are cheap and should be closed on the same machine,
since `scatter_add` in particular cannot be honestly declared on CUDA. Its CPU
row claims a fixed summation order, and an atomics-based CUDA kernel cannot
honour that. Declaring it on CUDA therefore requires either a deterministic
kernel or a decision to split the determinism claim per backend, which the
catalog has no column for today.

Reframes: #86, #87, #88, #106 (breadth, largely already closed).
Unblocks: #82, #83, #84, #90, #4, #85.
