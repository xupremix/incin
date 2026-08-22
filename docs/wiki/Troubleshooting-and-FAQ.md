# Troubleshooting & FAQ

Frequently asked questions and troubleshooting steps for Incin.

---

## 1. Using `cargo incin`

If rustc emits complex `typenum` trait errors, run `cargo incin check` instead:

```bash
cargo incin check
```

To see detailed rule explanations alongside errors:

```bash
cargo incin check --explain
```

---

## 2. Common Compilation Errors

### `cannot contract dimension UInt<...> with UInt<...>`
* **Cause**: Matrix multiplication inner dimensions do not match (e.g. attempting to multiply `[4, 8]` with `[3, 8]`).
* **Fix**: Ensure `A: [M, K]` is multiplied with `B: [K, N]`. For `[3, 8]`, transpose `B` first (`b.transpose(0, 1)` to get `[8, 3]`).

### `the trait MatMulShape is not satisfied`
* **Cause**: The input tensor ranks are incompatible with 2D or batched matrix multiplication.

### `cannot borrow immutable tensor as mutable`
* **Cause**: In Incin, neural network parameters track autograd gradients without requiring `&mut` references to the parameters themselves (`Tensor` uses interior mutation for gradients during backward pass).

---

## 3. Hardware & GPU Diagnostics

Run `cargo incin doctor` to verify detected accelerators and compute drivers:

```bash
cargo incin doctor
```

Example output:
```text
=== Incin Hardware & Capability Doctor ===
  • Host Architecture : x86_64-unknown-linux-gnu
  • SIMD Features     : AVX2, FMA, SSE4.2 (Detected)
  • CUDA Driver / GPU : NVIDIA RTX 4090 (Compute 8.9) — Available
  • WGPU Adapters     : Vulkan (NVIDIA RTX 4090)
  • Autotune Cache    : ~/.cache/incin/tuning.db (14 entries)
```
