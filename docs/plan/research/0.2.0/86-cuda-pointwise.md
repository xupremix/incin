#86 CUDA pointwise gap (39 ops) — M
Finding: 23 unary math + 5 activations + 11 scalar/binary missing; launch_unary_op/launch_binary_op accept arbitrary CUDA-C expressions (NVRTC, layout-aware, autotuned); CUDA elementwise declaration group has only 10 ops vs CPU ~40.
Recommendation: cuda_pointwise! declaration macro (name + forward expr + grad expr) emitting executor arm + tape-pushing backward together (cannot advertise without gradient); add to elementwise group; scalar forms stay own descriptors; match CPU dtype sets; claim contiguous until strided tested. If #8 lands first, adopt it.
Risk: NVRTC bf16/f16 math overloads; GELU convention drift vs CPU; #83 must show zero advertised-but-unexecutable.
