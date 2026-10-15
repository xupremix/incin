# Compiled graphs and pointwise fusion

The [stable deep dive into lowering](../#/deep_lowering) introduces the
legality-checked pointwise fuser (CMP-005): groups of pointwise operations
fuse only when the compiler can prove the fusion preserves semantics. This
chapter is about what that buys, what runs today, and what is still
missing.

## Fuse `relu(norm(x))`, keep the gradients

The fusion pass (`crates/incin-core/src/compiled/fusion.rs`,
`FusionPass::find_candidates` plus `apply`) recognizes chains like
normalization followed by activation and lowers them to one kernel
descriptor instead of two launches. The guarantee that matters is the
gradient one: fusion is a forward-only launch optimization, and the
backward rule still differentiates the original chain. A fusion that
changed a gradient would be a miscompilation, so the legality check runs
before the launch-count reduction, not after.

What executes today is the CPU reference evaluator under
`incin::experimental::compiled` — a preview-only surface with no stable
facade contract, no optimized backend, no deployment target, and no
portable artifact ABI. Its serialized plan snapshots are local preview
data. The NVRTC device-dispatch tail is still `#[ignore]`d: compiled and
proven on the host, not yet launched on a device.

## The sequence from here

1. CPU reference evaluator (landed): semantics and gradient parity.
2. NVRTC dispatch tail (in flight, #112): the same fused descriptors,
   launched on CUDA.
3. Versioned plan artifacts (CMP-006, `compiled/artifact.rs`): snapshots
   you can store and compare, still not a deployment format.
4. A deployment story: not started, not promised on any timeline.

Each step is only as trustworthy as the previous one's tests, which is why
the fusion work ships host-side proofs before device launches — the same
evidence ordering as the [WGPU campaign](./next_wgpu_gaps.md) and the
[CUDA push](./next_cuda.md).
