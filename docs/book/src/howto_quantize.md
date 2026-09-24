# Quantize tensors and weights

`quantize` and `dequantize` are two `Tensor` methods; everything else about a
`Q8_0` tensor is decided per operation, at compile time when the dtype is
static and at run time when it is not (issue #93). This chapter is the doing
half of [Quantization](./quantization.md): the recipes, and the refusal each
mistake earns.

## 1. Quantize a tensor and read it back

```rust,no_run
use incin::prelude::*;

# fn main() -> Result<()> {
// 64 elements: exactly two 32-element Q8_0 blocks.
let x = Cpu.randn(shape![64])?;
let q = x.quantize(-1)?;
println!("quantized: {:?} {:?}", q.dims(), q.dtype());

let back = q.dequantize::<f32>()?; // lossy: each block's f16 scale rounds
println!("decoded: {:?}", back.to_vec1::<f32>()?);
# Ok(())
# }
```

`quantize(axis)` takes a float tensor and returns the same shape with dtype
`Q8_0`; `dequantize::<Kout>()` takes a `Q8_0` tensor back to any float dtype.
The compression is real — 32 `f32` values become one 34-byte block — and so
is the loss: the per-block scale is `f16`.

## 2. Pre-flight the block rule before you call

The block axis is always the **last** axis, and its extent must be a whole
multiple of 32. Static shapes that violate it fail to compile; `Dyn` shapes
tell you at run time, and the message names the axis, the extent and the
required multiple:

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
// Extent is not a multiple of 32:
let bad_extent = Tensor::<Dyn, B>::zeros(vec![2, 48])?;
if let Err(err) = bad_extent.quantize(-1) {
    // Generic Message: quantize: axis 1 has extent 48, which is not a
    // multiple of the Q8_0 block size 32; the block axis is the last axis,
    // so its extent must be a whole multiple of 32
    println!("{err}");
}

// Blocks run along the last axis only:
let ok_extent = Tensor::<Dyn, B>::zeros(vec![2, 64])?;
if let Err(err) = ok_extent.quantize(0) {
    // Generic Message: quantize: axis 0 resolves to axis 0, but Q8_0 blocks
    // run along the last axis only; pass -1 (the last axis, extent rules
    // below)
    println!("{err}");
}

// Axis out of range:
if let Err(err) = ok_extent.quantize(5) {
    // Generic Message: quantize: axis 5 is out of bounds for a rank-2 tensor
    println!("{err}");
}

// Rank 0 has no axis at all:
let scalar = Tensor::<s![], B>::zeros(())?;
if let Err(err) = scalar.quantize(-1) {
    // Generic Message: quantize: a rank-0 tensor has no axis to block over;
    // the operand must have rank >= 1
    println!("{err}");
}
# Ok(())
# }
```

The same rule is proved at compile time for static shapes (see
[Debug shape and dtype errors](./howto_debug_shapes_errors.md) for the
`E0080` snippet) — a violation never reaches a kernel.

## 3. Quantize a real parameter

A module's weight is a gradient-marked tensor, and `q8_0` has no gradient
tracking, so `quantize` refuses it until you detach it:

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
let model = Linear::<s![64, 32], B>::build(())?;
let weight = model.weight.as_tensor()?; // [out_features, in_features]
println!("weight: {:?}", weight.dims());

// Without .detach():
//   Dtype ... "q8_0" ... is unsupported by backend 'Cpu' for
//   'gradient tracking'
let blocks = weight.detach().quantize(-1)?;
let floats = blocks.dequantize::<f32>()?;
println!("decoded: {:?}", floats.dims());
# Ok(())
# }
```

`Linear<s![64, 32]>` has weight `[32, 64]`, so the block axis is
`in_features`. It must be a multiple of 32 — a static `s![10, 32]` layer
fails at compile time rather than producing a truncated block. There is no
quantized parameter initializer in this release (issue #93): build in `f32`
and quantize, or load a checkpoint.

## 4. Run a quantized matmul through the dispatcher

`Tensor::matmul` admits floating-point operands only (recipe 5), so a
`q8_0 @ q8_0` product goes through the canonical dispatch surface:

```rust,no_run
use incin::backend_authoring::{
    ExecutionContext, HostReadback, execute,
    operations::{NoAttributes, op},
};
use incin::prelude::*;
use incin_core::exec::TensorHandle;

type B = DefaultBackend;

# fn main() -> Result<()> {
let lhs = Cpu.zeros(shape![2, 32])?.quantize(-1)?;
let rhs = Cpu.zeros(shape![32, 32])?.quantize(-1)?;

let ctx = ExecutionContext::from_scope(B::default());
let h1 = TensorHandle::from_storage::<B, Q8_0, incin_core::dist::placement::Local>(lhs.inner());
let h2 = TensorHandle::from_storage::<B, Q8_0, incin_core::dist::placement::Local>(rhs.inner());
let out = execute::<op::QuantizedMatMul, B>(&ctx, NoAttributes, &[h1, h2])?;
println!("{:?}", B::float_to_vec1::<f32>(&out)?);

// The same call under a training policy is refused, on CPU too:
let train_ctx = ExecutionContext::from_scope(B::default()).with_training(true);
let h1 = TensorHandle::from_storage::<B, Q8_0, incin_core::dist::placement::Local>(lhs.inner());
let h2 = TensorHandle::from_storage::<B, Q8_0, incin_core::dist::placement::Local>(rhs.inner());
if let Err(err) = execute::<op::QuantizedMatMul, B>(&train_ctx, NoAttributes, &[h1, h2]) {
    // backend 'Cpu' refused the request: training is unsupported for
    // quantized_matmul
    println!("{err}");
}
# Ok(())
# }
```

`quantize` and `dequantize` themselves *are* covered for training on CPU —
they record the straight-through estimator — while `quantized_matmul` is
forward-only. The capability row behind each is what `with_training(true)`
consults.

## 5. Read the refusals you will hit

Initializing a `q8_0` parameter directly — no initializer admits it:

```rust,no_run
use incin::nn::{Linear, True};
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
type QLinear = Linear<s![32, 4], B, True, Q8_0>;
if let Err(err) = QLinear::build(()) {
    // backend 'Cpu' refused the request: dtype q8_0 is unsupported for rand
    println!("{err}");
}
# Ok(())
# }
```

A product of two quantized tensors through the front door — `matmul` admits
floating-point operands only:

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
let a = Cpu.randn(shape![2, 32])?.quantize(-1)?;
let b = Cpu.randn(shape![32, 32])?.quantize(-1)?;
if let Err(err) = a.matmul(&b) {
    // matmul: invalid dtype: operation requires floating-point input
    // metadata
    //   attribute: dtype
    //   rule: operation requires floating-point input metadata
    //   fix: give every operand the same dtype, converting one with
    //   `.to_dtype(..)` if they genuinely differ
    println!("{err}");
}
# Ok(())
# }
```

An elementwise float op on a `q8_0` tensor does not compile — the dtype has
no per-element `f32` to read:

```rust,compile_fail
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
let q = Cpu.zeros(shape![64])?.quantize(-1)?;
let r = q.abs()?; // E0277: the trait bound `incin::Q8_0: FloatCapable` is not satisfied
# Ok(())
# }
```

## Common mistakes

- **Quantizing along an axis other than the last.** Blocks index the flat
  row-major buffer; any other axis is refused with *"Q8_0 blocks run along
  the last axis only"*.
- **Ignoring the multiple-of-32 rule for `Dyn` extents.** The message names
  the extent and the fix; for static shapes the compiler stops you first.
- **Calling `quantize` on a gradient-marked parameter.** `q8_0` advertises no
  gradient tracking: `.detach()` first, then quantize.
- **Building `Linear<..., K = Q8_0>`.** No initializer admits `q8_0`; build
  in `f32` and quantize the weights (recipe 3), or load a checkpoint.
- **Expecting `Tensor::matmul` to accept blocks.** Use the dispatch path
  (recipe 4); the front door admits floats only.
- **Saving a `q8_0` module as safetensors.** The safetensors writer refuses
  block dtypes — use `Format::Postcard` (see
  [Save and load](./howto_save_load.md)).

Next: [Debug shape and dtype errors](./howto_debug_shapes_errors.md) for the
`E0080` version of the block rule.
