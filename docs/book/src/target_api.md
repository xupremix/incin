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

## Axis selectors

Reductions accept one selector value for static axes, named axes, and runtime
signed axes. Structural cursor methods remain an advanced escape hatch for
authoring low-level shape operations. Static selectors preserve the output
shape at the type level, while named and runtime selectors use the shape facts
available after resolving the axis.

```rust,no_run
use incin::prelude::*;

let x = Cpu.ones(shape![4, 8, 16])?;
let summed = x.sum(axis!(1))?;
let kept = x.sum_keepdim(axis!(-2))?;
let first = x.sum(0isize)?;
let indices = x.argmax(axis!(2))?;
let minima = x.argmin(0isize)?;
let means = x.mean(axis!(-1))?;
let maxima = x.max_keepdim(axis!(0))?;
let minima_by_value = x.min(axis!(1))?;
let argmax = x.argmax(axis!(-1))?;
let flattened = x.flatten_range(1, -1)?;
# Ok::<(), incin::Error>(())
```

Numeric axes accept arbitrary signed `isize` values, including negative axes,
without a finite lookup table. Pass `axis!(...)` to any reduction or pass a
runtime `isize` directly. Named axes use the same methods when the axis
identity matters more than its numeric position:

```rust,no_run
use incin::prelude::*;

dim!(Batch, Channels, Width);
let x = Cpu.ones(shape![Batch, Channels, Width])?;
let channels = x.sum(axis!(Channels))?;
let kept = x.mean_keepdim(axis!(Channels))?;
# Ok::<(), incin::Error>(())
```

Runtime selectors validate their normalized position against the tensor rank.
`mean`, `max`, `min`, and their keep-dimension forms follow the same selector
rules.
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
