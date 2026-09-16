# Tensors

`Tensor<S, B, K, G, P, L>` has six type parameters, but you'll write the first
two almost always and let the rest default:

| Parameter | Meaning | Default |
|---|---|---|
| `S` | Shape (see [Shapes](./shapes.md)) | required |
| `B` | Backend (which device this runs on) | required (`DefaultBackend` if `cpu` is on) |
| `K` | Element dtype | `f32` |
| `G` | Gradient tracking (`Grad` / `NoGrad`) | `NoGrad` |
| `P` | Placement (distributed only) | `Local` |
| `L` | Layout (see [Layout](./layout.md)) | `Dyn` |

Each default is the option that claims the least. `Dyn` means nothing has
been established about where the elements live, which is why adding `L` changed
nothing about existing code: a tensor that has proven nothing behaves exactly as
it always did, and checks it could have skipped happen at runtime instead.

## Creating tensors

### 1. Target-first creation (Recommended)

Concrete application code selects a target device directly:

```rust,no_run
use incin::prelude::*;

// 1. Static compile-time target (Cpu, or Cuda::new(0))
let x = Cpu.randn(shape![2, 3])?; // ~ N(0, 1) standard normal

// 2. Rebinding to a specific static dtype (f16, bf16, f64, i64, etc.):
let half_target = Cpu.dtype::<f16>()?;
let h = half_target.zeros(shape![2, 3])?;

// 3. Dynamic runtime-chosen device (detect_device() or DeviceId):
let device = incin_backends::detect_device().unwrap_or_else(DeviceId::cpu);
let target: Target<Native, Dyn> = Target::new((), device, ());
let dynamic_zeros = target.zeros([2, 3])?;

// 4. Dynamic runtime-chosen dtype (.dtype_dynamic):
let desc = DTypeId::F64.descriptor();
let f64_target = target.dtype_dynamic(desc)?;
let dynamic_f64 = f64_target.ones([2, 3])?;
# Ok::<(), incin::Error>(())
```

See [Target-first construction](./target_api.md) for the rest of the
creation methods (`rand`, `zeros`, `ones`, `full`, `arange`, `linspace`,
`tensor`) and their runtime-shape forms.

### 2. Type-level and macro construction

Generic code fixing a backend type or using literal macros:

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let zeros = Tensor::<s![2, 3], B>::zeros(())?;
let ones = Tensor::<s![2, 3], B>::ones(())?;
let filled = Tensor::<s![2, 3], B>::full(7.0, ())?;
let ranged = Tensor::<s![4], B>::arange(1.0, 2.0, ())?;   // start, step, args
let spaced = Tensor::<s![3], B>::linspace(0.0, 1.0, ())?; // start, end, args
let uniform = Tensor::<s![2, 3], B>::rand(())?;
let normal = Tensor::<s![2, 3], B>::randn(())?;

// From literal data macro - shape and dtype inferred from the literal itself.
let literal = tensor![[1.0, 2.0], [3.0, 4.0]]?;         // [2, 2], f32
let integers = tensor![1, 2, 3]?;                        // [3], i64 (matches torch.tensor's default)
let explicit = tensor![1.0, 2.0; dtype: f64]?;

// From a dynamic shape.
let dynamic = Tensor::<Dyn, B>::zeros(vec![2, 3])?;
# Ok::<(), incin::Error>(())
```

`()` as the constructor argument for a fully static shape is intentional:
a static `Shape::Arg` is a tuple of units, and the empty tuple is the only
value of that type. Once any axis is runtime-determined (`Dyn`, or a `Bound`
shape via the [target API](./target_api.md)), the argument carries the actual
sizes.

## dtype

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

// K defaults to f32. Say it explicitly for anything else:
let doubles = Tensor::<s![2, 2], B, f64>::ones(())?;
let ints = Tensor::<s![2, 2], B, i64>::zeros(())?;

// A runtime-chosen dtype uses Dyn as K, carrying the tag at runtime instead
// of in the type:
let runtime_dtype = Tensor::<Dyn, B, Dyn>::ones((vec![2, 2], DTypeId::F64.descriptor()))?;
assert_eq!(runtime_dtype.dtype(), DTypeId::F64.descriptor());

// Or use target-first dynamic rebinding:
let target: Target<Native, Dyn> = Target::new((), DeviceId::cpu(), ());
let dynamic_target = target.dtype_dynamic(DTypeId::F64.descriptor())?;
let dynamic_tensor = dynamic_target.zeros([2, 2])?;
# Ok::<(), incin::Error>(())
```

### Declaring a dtype is not computing in one

Every dtype above *allocates*. Far fewer of them *execute*, and the two are
separate questions with separate answers. On CPU today `matmul` is `f32` only,
and elementwise arithmetic is float only, so `i64` addition and `f16` matmul
are both refused:

```text
backend 'Cpu' refused the request: dtype f16 is unsupported for matmul
```

That refusal is generated from the same capability tables as
[`docs/capabilities.md`](https://github.com/xupremix/incin/blob/master/docs/capabilities.md),
so the table, the error message, and the kernel cannot disagree. You can ask
the registry directly instead of trying and catching, which is what
`cargo incin doctor` does:

```rust,no_run
use incin::prelude::*;
use incin_core::exec::{Capabilities, CapabilityQuery, LayoutClass, MathMode};
use incin_core::shapes::OperationKind;
use incin_core::tensor::device::DeviceKind;
use incin_core::tensor::dtype::DTypeId;

let query = CapabilityQuery {
    operation: incin_core::exec::OperationIdentity::Builtin(OperationKind::MatMul),
    dtype: DTypeId::F16.descriptor(),
    layout: LayoutClass::Contiguous,
    rank: 2,
    training: false,
    math_mode: MathMode::Precise,
};
let level = incin_backends::capability::registry(DeviceKind::Cpu).support(&query);
```

`cargo run -p incin --example dtypes --features cpu` runs the whole axis
end to end: allocation in all eight built-in dtypes, `to_dtype` conversions,
the registry query above, and the refusal.

## Arithmetic

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let a = Tensor::<s![2, 2], B>::ones(())?;
let b = Tensor::<s![2, 2], B>::ones(())?;

let sum = &a + &b;
let diff = &a - &b;
let prod = &a * &b;
let quot = &a / &b;
let eq = a.eq(&b)?;       // elementwise comparison
let both = a.eq(&b)?.logical_and(&a.eq(&b)?)?;
# Ok::<(), incin::Error>(())
```

The checked methods `try_add`, `try_sub`, `try_mul`, and `try_div` broadcast
compatible shapes; use `add_exact`, `sub_exact`, `mul_exact`, and `div_exact`
when the operands must match (`ShapeEq`). `+`, `-`, `*`, and `/` are also
overloaded in every owned and referenced combination. They broadcast
between compatible shapes and return a tensor directly. Operator syntax is the
convenience boundary: a dynamic shape or backend failure panics with a short,
fixed operator-only message. Use the named methods whenever the failure must
remain recoverable.

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let a = Tensor::<s![2, 3], B>::ones(())?;
let b = Tensor::<s![3], B>::full(2.0, ())?; // shorter shape, broadcasts against `a`

let sum = a.clone() + b.clone();            // operator: broadcasts, panics on failure
let sum2 = a.try_add(&b)?;                  // checked broadcast operation
assert_eq!(sum.dims().as_ref(), &[2, 3]);
# Ok::<(), incin::Error>(())
```

## Reductions and shape ops

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let x = Tensor::<s![2, 3], B>::ones(())?;

let by_row = x.sum_keepdim(axis!(1))?;
let by_last_row = x.sum_keepdim(axis!(-1))?;
let idx = x.argmax(axis!(1))?;         // index dtype defaults to u32

// `reshape` changes the geometry and keeps the target shape in the type.
let reshaped = x.reshape(shape![3, 2])?;

// sum_all/mean_all consume the tensor (they're the last op in a reduction
// chain more often than not), so clone first if you still need the original.
let total = x.clone().sum_all()?;
let mean = x.mean_all()?;
# Ok::<(), incin::Error>(())
```

`reshape` is not the same as `to_shape`, and mixing them up is easy:
`reshape` produces a *different* geometry with the same element count;
`to_shape` re-asserts a shape **type** over the *same* dims and fails if they
disagree. Use `to_shape` to recover a static type from a `Dyn` tensor, and
`reshape` to actually change the layout.

## Sorting, counting, and repeating

These CPU examples group rows by an integer assignment without making each
bin's population a tensor extent.

### Stable sorting and reusable indices

`sort(dim, descending)` returns `(values, indices)`, preserving the input
shape in both outputs. Values retain the input dtype; indices are `u32`
positions within the selected axis. Equal keys keep their input order in
both ascending and descending sorts. `argsort` returns just the indices.
Both methods currently take an unsigned `usize` axis, not `axis!(...)` or a
negative axis. Their outputs are dense row-major and `NoGrad`, even when the
input tracks gradients.

```rust
use incin::prelude::*;

let assignment = Cpu.tensor([2_i64, 0, 2, 1, 2, 0])?;
let rows = Cpu.tensor([
    [0.0_f32, 0.5], [1.0, 1.5], [2.0, 2.5],
    [3.0, 3.5], [4.0, 4.5], [5.0, 5.5],
])?;
let (experts, order) = assignment.sort(0, false)?;
let grouped = rows.index_select(axis!(0), &order)?;
assert_eq!(experts.to_vec1::<i64>()?, vec![0, 0, 1, 2, 2, 2]);
assert_eq!(order.to_vec1::<u32>()?, vec![1, 5, 3, 0, 2, 4]);
assert_eq!(grouped.to_vec1::<f32>()?,
    vec![1.0, 1.5, 5.0, 5.5, 3.0, 3.5, 0.0, 0.5, 2.0, 2.5, 4.0, 4.5]);
# Ok::<(), incin::Error>(())
```

`index_select` accepts the proven layout of `order` directly; no layout
erasure or host readback is needed to apply the permutation. Likewise,
`gather` accepts the indices returned by `topk`:

```rust
use incin::prelude::*;

let logits = Cpu.tensor([[0.1_f32, 0.9, 0.5], [0.7, 0.2, 0.8]])?;
let (values, indices) = logits.topk(2, axis!(1), true)?;
let gathered = logits.gather(axis!(1), &indices)?;
assert_eq!(gathered.to_vec1::<f32>()?, values.to_vec1::<f32>()?);
# Ok::<(), incin::Error>(())
```

### Fixed-width histograms

`bincount::<N>()` counts every integer index in the input, regardless of its
shape, into a rank-one, `N`-wide `i64`, `NoGrad` tensor. `N` is an exact,
positive bin count, not a minimum length: negative indices and indices at
least `N` return errors rather than extending the result or being ignored.
Unused bins contain zero.

```rust
use incin::prelude::*;

let assignment = Cpu.tensor([[2_i64, 0], [2, 1], [2, 0]])?;
let counts = assignment.bincount::<3>()?;
let ends = counts.cumsum(axis!(0))?;
assert_eq!(counts.to_vec1::<i64>()?, vec![2, 1, 3]);
assert_eq!(ends.to_vec1::<i64>()?, vec![2, 3, 6]);
# Ok::<(), incin::Error>(())
```

The inclusive cumulative counts are exclusive **end offsets** for the sorted
buffer: the bins occupy `0..2`, `2..3`, and `3..6`. Start offsets are zero
followed by the preceding end offsets, not the cumulative counts themselves.

### Repeating each element instead of tiling

`repeat_interleave(repeats, dim)` uses one positive `usize` repeat count for
every element along a signed `isize` axis. Negative axes count from the end;
zero repeats and out-of-range axes return errors. Unlike `repeat`, which
tiles an entire axis, it places copies of each element next to each other:

```rust
use incin::prelude::*;

let x = Cpu.tensor([1.0_f32, 2.0, 3.0])?;
assert_eq!(x.repeat_interleave(2, -1)?.to_vec1::<f32>()?,
    vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
assert_eq!(x.repeat(&[2])?.to_vec1::<f32>()?,
    vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
# Ok::<(), incin::Error>(())
```

The selected extent is multiplied by `repeats`; other extents are unchanged.
The result has shape type `Dyn` and retains the input dtype and gradient
mode. For gradient-tracking floating-point inputs, backward sums the
contributions from each element's copies. Per-element repeat-count tensors
are not an argument to this method.

## Reading values back to the host

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let x = Tensor::<s![3], B>::ones(())?;
let values: Vec<f32> = x.to_vec1::<f32>()?;
assert_eq!(values, vec![1.0, 1.0, 1.0]);
# Ok::<(), incin::Error>(())
```

Reading a value back is a synchronization point on a device backend, cheap
on CPU, worth batching on an accelerator.
