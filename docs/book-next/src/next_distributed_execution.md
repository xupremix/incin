# Distributed execution: from plans to ranks

[Distributed planning](../#/distributed) ends where execution begins: a
`TensorParallelPlan` says which rank owns which shard, but something still
has to run the shards and move the bytes. This chapter covers that
something — the reference transport, the two partition legs, and the
hardware gate beyond which real multi-rank training lives.

## Two partition axes, proven on CPU

A `TensorParallelPlan` for two ranks splits a linear layer two ways.
Column-parallel: rank `r` owns output-column shard `r`, and the shards
gather into the replicated output. Row-parallel: the contraction axis
splits as `X = [X0 | X1]` and `W = [W0; W1]`, each rank computes its
partial sum, and one `AllReduce(Sum)` replicates the total. Both legs run
on real CPU kernels in `crates/incin/tests/tp2_reference.rs`: each rank's
shard matmul executes on the CPU backend, the partial outputs cross the
deterministic in-process reference transport through the plan's own
`AllGather`/`AllReduce` descriptor, and the rebuilt output must match the
single-device matmul within tolerance.

```rust,ignore
// Column leg: shard matmuls, AllGather through the plan's descriptor,
// kernel concat rebuilds the full output within tolerance.
column_parallel_partitioned_matmul_matches_single_device()?;

// Row leg: partial sums, one AllReduce(Sum), the replicated sum matches.
row_parallel_partitioned_matmul_matches_single_device()?;
```

```sh
cargo test -p incin --features cpu,distributed-reference --test tp2_reference
```

Head-parallel attention crosses the same transport at the host-arithmetic
level; wiring it through backend kernels is future work and is not claimed
here. The two-host NCCL leg stays hardware-gated.

## The lowering that already exists

FSDP/ZeRO-1 and ZeRO-2 lower onto the trainer's synchronizer seam,
proven on CPU against scripted peers (`dist/fsdp.rs`, `dist/sync.rs`).
ZeRO-3 parameter sharding, tensor-parallel execution at scale, and
pipeline-parallel schedules are planning surfaces only. Optimizer state
stays full-size per rank, and global gradient clipping is unsupported while
gradients are masked — both documented limits, not oversights.

## The hardware gate

The NCCL transport is hardware-proven outside CI: single-process two-GPU
loopback plus the two-process two-rank harness, including a heterogeneous
pair. In CI, multi-rank execution needs `HARDWARE_CUDA_RUNNER` (#82), which
is unset — the same gate as the [CUDA chapter](./next_cuda.md). Until a
runner exists, "distributed works" means CPU-proven lowering plus
hardware-proven transport, composed but not yet combined in automation.
`Trainer::fit` does not shard its input data either: pair it with
`incin-data`'s `DistributedSampler` yourself.
