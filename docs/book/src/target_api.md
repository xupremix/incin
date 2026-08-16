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

Runtime reshape inference is a value-level shape specification, separate from
indexing. Known extents remain in the output shape proof:

```rust,no_run
use incin::prelude::*;

let x = Tensor::<s![2, 3], DefaultBackend>::ones(())?;
let y = x.reshape_infer(shape![6, infer])?;
// y carries the partial shape s![6, usize].
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
authoring low-level shape operations. Static reductions and flattening preserve
output shape information at the type level. Transpose and runtime selectors
preserve known rank, while named selectors use the shape facts available after
resolving the axis.

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
let transposed = x.transpose(axis!(0), axis!(1))?;
let flattened = x.flatten(axis!(1), axis!(2))?;
let stacked = x.stack(&x, axis!(0))?;
let concatenated = x.concat(&x, axis!(-1))?;
let flattened_runtime = x.flatten(1isize, -1isize)?;
let expanded = x.unsqueeze(axis!(1))?;
let squeezed = expanded.try_squeeze(1isize)?;
# Ok::<(), incin::Error>(())
```

Numeric axes accept arbitrary signed `isize` values, including negative axes,
without a finite lookup table. Pass `axis!(...)` to reductions and other
single-axis operations, or pass a runtime `isize` directly. Named axes use the
same methods when the axis identity matters more than its numeric position:

```rust,no_run
use incin::prelude::*;

dim!(Batch, Channels, Width);
let x = Tensor::<s![Batch = 4, Channels = 8, Width = 16], DefaultBackend>::ones(())?;
let channels = x.sum(axis!(Channels))?;
let kept = x.mean_keepdim(axis!(Channels))?;
# Ok::<(), incin::Error>(())
```

Runtime selectors validate their normalized position against the tensor rank.
`mean`, `max`, `min`, and their keep-dimension forms follow the same selector
rules.
`flatten` uses the same signed runtime axis convention. Use `concat` and
`stack` with a selector for ordinary code. The structural cursor forms remain
available only for low-level shape implementations. `unsqueeze` and
`try_squeeze` use the same selectors: static selectors retain exact structural
shapes, while signed runtime selectors retain known rank and validate the
selected extent for squeezing.

Operations that replace one dimension's runtime extent also retain the
unaffected shape information. For example, a static selector on `narrow`,
`index_select`, `chunk`, `split`, or `topk` produces a partially dynamic shape
such as `s![2, usize]`, while a signed runtime selector produces `Ranked<R>`.
The values and indices returned by `topk` share that same output shape.

Dimension-selecting operations use the same selector vocabulary. `narrow`,
`chunk`, `split`, `gather`, `scatter`, `index_select`, and `topk` accept static,
named, or signed runtime selectors. Runtime selectors are checked against the
rank before dispatch, so invalid negative axes return the normal recoverable
shape error:

```rust,no_run
use incin::prelude::*;

let x = Cpu.ones(shape![4, 8, 16])?;
let indices = Tensor::<s![4, 8, 16], DefaultBackend, u32>::zeros(())?;
let selected = x.index_select(axis!(-1), &Tensor::<s![2], DefaultBackend, u32>::zeros(())?)?;
let parts = x.chunk(2, axis!(1))?;
let pieces = x.split(4, -1isize)?;
let gathered = x.gather(axis!(1), &indices)?;
let _ = x.scatter(axis!(-1), &indices, &x)?;
let _ = x.topk(2, 1isize, true)?;
assert_eq!(selected.dims().as_ref(), &[4, 8, 2]);
assert_eq!(parts.len(), 2);
assert_eq!(pieces.len(), 4);
assert_eq!(gathered.dims().as_ref(), &[4, 8, 16]);
# Ok::<(), incin::Error>(())
```

Axis-preserving operations use the same selectors:

```rust,no_run
use incin::prelude::*;

let x = Cpu.ones(shape![4, 8, 16])?;
let cumulative = x.cumsum(-1)?;
let probabilities = x.softmax(axis!(1))?;
# Ok::<(), incin::Error>(())
```

For generic known-rank runtime shapes, `sum` and `sum_keepdim` retain the rank
arithmetic in the type while leaving extents runtime-valued. The method
selects the rank proof from the input shape, so callers do not need a separate
ranked method name.

Static axis values use separate forward and from-end representations. This
keeps positive and negative proof dispatch independent while their runtime
normalization still follows the same signed-axis rules. The advanced cursor
types are available for low-level shape implementations; application code
should use `axis!(...)` or a signed runtime value.

Named reduction selectors retain known rank. Named dimensions are resolved at
runtime, while the public output uses the strongest shape proof available for
the selected operation. On stable Rust, the blanket recursive lookup cannot
currently produce an exact named post-reduction type without overlapping trait
implementations. The deliberate fallback is `Ranked<R>` or `Ranked<R-1>` for
known-rank inputs, and `Dyn` only when the input rank is dynamic.

Tensor indexing and slicing use `i![...]` with signed indices and ordinary Rust
ranges. Reshape inference is separate from indexing and uses `shape![..., infer]`.
Advanced type-level reshape targets remain available through
`incin::macros::advanced::idx` and `reshape_idx::<idx![... ]>()`. Axis selection is
also separate, so `axis!(-1)` never changes `i![-1]` indexing rules.

```rust,no_run
use incin::prelude::*;

let x = Cpu.ones(shape![4, 8, 16])?;
let last = x.get(i![-1, .., ..])?;
let window = x.get(i![.., 2..6, ..])?;
# Ok::<(), incin::Error>(())
```

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
