# Tensors

`Tensor<S, B, K, G, P>` has five type parameters, but you'll write the first
two almost always and let the rest default:

| Parameter | Meaning | Default |
|---|---|---|
| `S` | Shape (see [Shapes](./shapes.md)) | required |
| `B` | Backend (which device this runs on) | required (`DefaultBackend` if `cpu` is on) |
| `K` | Element dtype | `f32` |
| `G` | Gradient tracking (`Grad` / `NoGrad`) | `NoGrad` |
| `P` | Placement (distributed only) | `Local` |

## Creating tensors

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

// From literal data - shape and dtype inferred from the literal itself.
let literal = tensor![[1.0, 2.0], [3.0, 4.0]]?;         // [2, 2], f32
let integers = tensor![1, 2, 3]?;                        // [3], i64 (matches torch.tensor's default)
let explicit = tensor![1.0, 2.0; dtype: f64]?;

// From a dynamic shape.
let dynamic = Tensor::<Dyn, B>::zeros(vec![2, 3])?;
# Ok::<(), incin::Error>(())
```

`()` as the constructor argument for a fully static shape is not decoration  -
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
# Ok::<(), incin::Error>(())
```

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

The checked methods `try_add`, `try_sub`, `try_mul`, and `try_div` require the
two operands' shape types to match exactly (`ShapeEq`). `+`, `-`, `*`, and `/`
are also overloaded in every owned and referenced combination. They broadcast
between compatible shapes and return a tensor directly. Operator failures
panic with operation context; use the checked methods when the failure must
remain a `Result`.

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let a = Tensor::<s![2, 3], B>::ones(())?;
let b = Tensor::<s![3], B>::full(2.0, ())?; // shorter shape, broadcasts against `a`

let sum = a.clone() + b.clone();            // operator: broadcasts
let sum2 = a.try_add(&b)?;                  // checked broadcast operation
assert_eq!(sum.dims().as_ref(), &[2, 3]);
# Ok::<(), incin::Error>(())
```

## Reductions and shape ops

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let x = Tensor::<s![2, 3], B>::ones(())?;

let by_row = x.sum_keepdim::<1>()?;
let by_last_row = x.sum_keepdim_axis(axis!(-1))?;
let idx = x.argmax::<1>()?;         // index dtype defaults to u32

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

## Reading values back to the host

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let x = Tensor::<s![3], B>::ones(())?;
let values: Vec<f32> = x.to_vec1::<f32>()?;
assert_eq!(values, vec![1.0, 1.0, 1.0]);
# Ok::<(), incin::Error>(())
```

Reading a value back is a synchronization point on a device backend  -  cheap
on CPU, worth batching on an accelerator.
