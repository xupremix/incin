# Target-first construction

Incin's application-facing allocation API starts with a target. A target such
as `Cpu` selects the backend and device, while `shape!` supplies static proof
or runtime dimensions. This is part of the normal API and needs no feature
flag.

```rust,no_run
use incin::prelude::*;

let x = Cpu.zeros(shape![2, 3])?;
let batch = 4;
let y = Cpu.zeros(shape![batch, 3])?;
let z = Cpu.zeros([batch, 3])?;
# Ok::<(), incin::Error>(())
```

Static literals and explicit const paths preserve compile-time dimensions:

```rust,no_run
use incin::prelude::*;

const FEATURES: usize = 128;
let weights = Cpu.zeros(shape![const FEATURES, const FEATURES])?;
# Ok::<(), incin::Error>(())
```

The typed constructor remains available when code intentionally fixes the
backend type:

```rust,no_run
use incin::prelude::*;

type Backend = DefaultBackend;
let tensor = Tensor::<s![2, 2], Backend>::ones(())?;
# Ok::<(), incin::Error>(())
```

Layer construction follows the same target-first boundary through the
canonical builder extension. The older direct `Linear::new` and
`new_on_target` spellings are not part of the public prelude.

```rust,no_run
use incin::prelude::*;

let layer = incin_core::nn::linear::linear(shape![4, 3]).init(&Cpu)?;
assert_eq!(layer.weight.shape_dims(), vec![3, 4]);
# Ok::<(), incin::Error>(())
```

Arithmetic operators return tensors directly and include operation context in
their panic message when a runtime broadcast or backend check fails. Use the
`try_add`, `try_sub`, `try_mul`, `try_div`, and `try_neg` methods when the
failure must remain a recoverable `Result`.

## Compile-time axis selectors

Reductions can select a numeric axis with a const generic or with a selector
value. Structural cursor methods remain an advanced escape hatch when the
output shape can be proven statically. Ordinary selector methods return the
runtime shape that is available after resolving the axis.

```rust,no_run
use incin::prelude::*;

let x = Cpu.ones(shape![4, 8, 16])?;
let summed = x.sum::<1>()?;
let kept = x.sum_keepdim_axis(axis!(-2))?;
let first = x.sum_axis(axis!(0))?;
let indices = x.argmax::<2>()?;
let minima = x.argmin::<0>()?;
let means = x.mean_axis(axis!(-1))?;
let maxima = x.max_keepdim_axis(axis!(0))?;
let minima_by_value = x.min_axis(axis!(1))?;
let argmax = x.argmax_axis(axis!(-1))?;
let flattened = x.flatten_range(1, -1)?;
# Ok::<(), incin::Error>(())
```

Numeric axes accept arbitrary signed `isize` values, including negative axes,
without a finite lookup table. Use `axis!(...)` with `sum_axis` or
`sum_keepdim_axis` when the selector is chosen as a value. Named axes remain
available through `sum_named` and `sum_keepdim_named` when the axis identity
matters more than its numeric position. Runtime selectors validate their
normalized position against the tensor rank. `mean_axis`, `max_axis`,
`min_axis`, and their keep-dimension forms follow the same selector rules.
`flatten_range` and `concat_axis` use the same signed runtime axis convention.
For generic known-rank runtime shapes, `Ranked<R>` also provides
`sum_runtime_ranked` and `sum_keepdim_runtime_ranked`; these retain the rank
arithmetic in the type while leaving extents runtime-valued.

Tensor slicing and indexing use `idx![...]`. Axis selection is a separate
concept, so `axis!(-1)` never changes the `idx![-1]` indexing rules. There is
no separate `i!` macro because the existing `idx!` syntax already owns tensor
indexing, slicing, and reshape inference in one type-level grammar.

Named dimensions and const dimensions use different syntax. A named axis is
written as `s![Batch, Features]`; a const path must be marked explicitly:

```rust,no_run
use incin::prelude::*;

const BATCH: usize = 32;
const FEATURES: usize = 128;
dim!(Batch, Features);
let x = Cpu.zeros(shape![Batch = const BATCH, Features = const FEATURES])?;
type X = s![Batch = const BATCH, Features = const FEATURES];
let _: Tensor<X, _, f32, NoGrad> = x;
# Ok::<(), incin::Error>(())
```

The explicit `const` marker preserves the existing bare `s![Batch, ...]`
named-dimension grammar and makes the meaning of a path unambiguous.
