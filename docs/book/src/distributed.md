# Distributed planning

The `distributed` feature provides typed meshes, placements, collective
descriptors, and validation for data parallel, tensor parallel, pipeline, FSDP,
and ZeRO plans. Most of that surface is planning: it validates and explains a
plan, it does not run one.

The exception is FSDP/ZeRO data parallel (#99). The preview trainer executes
ZeRO-1 (all-reduce, then mask gradients to the rank's owned slice) and ZeRO-2
(reduce-scatter gradients) plus a parameter all-gather after every optimizer
step, through `Trainer::with_fsdp_synchronizer`. That lowering is a host-side
`f64` protocol proven on the CPU: scripted two-rank arithmetic, a byte
measurement showing reduce-scatter retains `1/N` of the gradient bytes an
all-reduce would, and a two-rank trajectory equal to the single-device
full-batch reference. ZeRO-3 is refused at plan build
(`TrainError::UnsupportedShardingStage`): parameter-sharded execution does not
exist, and an unimplemented stage fails closed rather than approximating a
different one. Tensor- and pipeline-parallel execution, optimizer-state
sharding memory (`1/N` state), and checkpoint recomputation remain planning
only.

`distributed-reference` provides a deterministic in-process transport for
conformance and local plan development. `distributed-nccl` provides the
two-host CUDA transport. The preview trainer refuses a multi-device plan when
collectives are unavailable instead of silently executing a local approximation.

The current limitation is deliberate: there is no promise that a model can be
trained across hosts. The FSDP lowering above runs only against a
caller-supplied synchronizer - no NCCL-wired implementation ships here - and
the rest of the planning APIs validate typed plans and inspect their
requirements rather than replace a distributed runtime.
