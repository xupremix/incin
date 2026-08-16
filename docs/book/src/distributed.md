# Distributed planning

The `distributed` feature provides typed meshes, placements, collective
descriptors, and validation for data parallel, tensor parallel, pipeline, FSDP,
and ZeRO plans. It is a planning surface, not an end-to-end distributed
training runtime.

`distributed-reference` provides a deterministic in-process transport for
conformance and local plan development. `distributed-nccl` provides the
two-host CUDA transport. The preview trainer refuses a multi-device plan when
collectives are unavailable instead of silently executing a local approximation.

The current limitation is deliberate: there is no promise that a model can be
trained across hosts. Use these APIs to validate a typed plan and inspect its
requirements, not as a replacement for a distributed runtime.
