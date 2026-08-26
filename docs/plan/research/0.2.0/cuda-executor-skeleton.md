CUDA new-operation checklist (cross-cutting) — M skeleton, S per op after
Today per op: (1) declare in cuda_descriptor_operations! group or legacy row; (2) .cu kernel include_str! or rendered template; (3) launch_* with dtype/contiguity checks + NVRTC compile cache; (4) Execute<op::X> impl w/ arity+downcast+invalid+kernel_error, TapeEntry if training; (5) assert_every_advertised_cuda_row_executes build proof; (6) regenerate capabilities.md.
Public macro must parameterize: backend/storage/BACKEND_NAME, arity+downcast, attribute extraction, dtype gate, tape policy, error mapping, coverage assertion.
Risks: triple bookkeeping drift (only grouped rows compile-checked); hardcoded float* ABI; training=false rows silently dropping grads.
