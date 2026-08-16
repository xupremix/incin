# Quantization

Quantization is a backend capability, not a general tensor conversion API.
The current public contract exposes the `Q8_0` storage format and the backend
operations `quantize`, `dequantize`, and `quantized_matmul`.

The CPU backend implements these paths. Quantized operations are marked as
inference-only because they do not record an autograd tape entry. There is no
claim here for quantization-aware training, GPTQ, AWQ, or a general quantized
model importer. A future tensor-level API can be added once its dtype,
serialization, and gradient contracts are settled.

The authoritative dtype definitions are in `incin_core::tensor::dtype`. Backend
authors should advertise only the quantized operations their executor really
implements, using the same capability and `Execute` checks as other
operations.
