#82 Real GPU execution CI — S (code) + hardware lead time
Finding (defect): with HARDWARE_CUDA_RUNNER unset, dist2-network/multinode stay true and inherit ["ubuntu-latest"] fallback -> NCCL jobs run on GPU-less runner. cuda/wgpu-native skip with warnings. metal + wgpu-software fine.
Recommendation: self-hosted NVIDIA runner (ephemeral, one-shot VMs, no long-lived secrets, schedule/dispatch-only triggers — public-repo runner security); fix resolver to set dist2_network/multinode=false when CUDA runner unset; distinguishable skip conclusions.
Unblocks: #83 and every accelerator close-out; #97/#99 dist jobs.
