# Layers and `#[module]`

Every built-in layer follows the same two-step shape: a `Shape` type
parameter names what's static, and a `build(args)` constructor takes
whatever isn't. `args` uses the same flexible-argument system as tensor
constructors - pass exactly the runtime values the static shape didn't
already pin down, in a tuple if there's more than one, and `()` when there
are none.

## Flattening runtime image axes

Use `Flatten::new(axis!(1), axis!(-1))` when a model receives a runtime-shaped tensor and ordinary
code should describe the axis range directly. The signed end axis `-1` means
the final axis, so this keeps the leading batch axis and flattens the image
axes without exposing structural cursor types. For a statically known range,
call `tensor.flatten(axis!(1), axis!(-1))` to preserve the exact output shape.

```rust,no_run
use incin::prelude::*;

let flatten = Flatten::new(axis!(1), axis!(-1));
let images = Cpu.ones(shape![32, 1, 28, 28])?;
let flat = flatten.forward(images)?;
assert_eq!(flat.dims().as_ref(), &[32, 784]);
# Ok::<(), incin::Error>(())
```

## Initializing layer parameters

Every layer builder takes an `Init` scheme per parameter. The defaults are
deliberate (Kaiming for weights, zeros for biases); override them when you know
better:

```rust,ignore
use incin::prelude::*;

let layer = incin_core::nn::linear::linear(shape![128, 10])
    .weight_init(incin_core::nn::init::kaiming_uniform())   // the default
    .bias_init(incin_core::nn::init::zeros())
    .init(&Cpu)?;
```

`Init` covers `Zeros`, `Ones`, `Rand`, `Randn`, `Constant(f64)`,
`Uniform { bound }`, and both Kaiming variants with a gain parameter. The same
fields exist on conv, normalization, embedding, and RNN builders, so one enum
is the whole initialization story.

## Linear

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let layer = Linear::<s![768, 256], B>::build(())?;
let x = Tensor::<s![32, 768], B>::ones(())?;
let y = layer.forward(x)?;
assert_eq!(y.dims().as_ref(), &[32, 256]);
# Ok::<(), incin::Error>(())
```

## Activations

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let x = Tensor::<s![2, 4], B>::ones(())?;
let a = ReLU.forward(x.clone())?;
let b = GELU.forward(x.clone())?;
let c = Sigmoid.forward(x.clone())?;
let d = Tanh.forward(x)?;
# Ok::<(), incin::Error>(())
```

## Convolution and pooling

`Conv2d`'s shape parameter is six-wide: `(OutChannels, InChannels, Kernel,
Stride, Padding, Dilation)`.

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

type ConvShape = s![4, 1, 3, 1, 0, 1]; // 1 in -> 4 out, 3x3, stride 1, no padding
let conv = Conv2d::<ConvShape, B>::build(())?;
let x = Tensor::<s![1, 1, 8, 8], B>::ones(())?;
let h = conv.forward(x)?;

let pool = MaxPool2d::<typenum::U2, typenum::U2>::new()?; // kernel 2, stride 2
let h = pool.forward(h)?;
assert_eq!(h.dims().as_ref(), &[1, 4, 3, 3]);
# Ok::<(), incin::Error>(())
```

## Normalization

`BatchNorm2d`, `LayerNorm`, and `RMSNorm` all take their channel count as a
one-element shape tuple, `(Channels,)`, and their `build` arguments are
whatever the static shape didn't already fix - an epsilon (and, for batch
norm, a momentum):

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let bn = BatchNorm2d::<s![4], B>::build((1e-5_f32, 0.1_f32))?; // eps, momentum
let x = Tensor::<s![1, 4, 8, 8], B>::ones(())?;
let h = bn.forward(x)?;

let ln = LayerNorm::<s![8], B>::build(1e-5_f32)?; // eps
let x2 = Tensor::<s![2, 8], B>::ones(())?;
let h2 = ln.forward(x2)?;
# Ok::<(), incin::Error>(())
```

## Embedding

`Embedding`'s shape is `(Vocab, EmbedDim)`. Its forward input's element type
matches its own `K` (the layer's float element by default) rather than an
integer dtype - the underlying kernel reads any dtype it can convert to an
exact integer, so an index tensor is written with integer-valued floats:

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let emb = Embedding::<s![16, 4], B>::build(())?; // 16 rows, 4-wide
let idx = Tensor::<s![3], B>::from_slice(&[1.0, 2.0, 3.0], ())?;
let h = emb.forward(idx)?;
assert_eq!(h.dims().as_ref(), &[3, 4]);
# Ok::<(), incin::Error>(())
```

## LSTM

`LSTM`'s shape is `(InFeatures, OutFeatures)`. Unlike the layers above, its
`forward` takes the initial hidden and cell state explicitly - there's no
implicit "start from zero" - and returns both the full output sequence and
the final state:

```rust,ignore
use incin::prelude::*;
type B = DefaultBackend;

let cell = LSTMCell::<s![4, 6], B>::build(())?;
let lstm = LSTM::new(cell);

let x = Tensor::<s![2, 5, 4], B>::ones(())?;   // [batch, seq, in_features]
let h0 = Tensor::<s![2, 6], B>::zeros(())?;
let c0 = Tensor::<s![2, 6], B>::zeros(())?;

// The cell is the executable recurrent primitive; the wrapper preserves this
// contract across a statically-known sequence.
let x_step = x.try_narrow(1isize, 0, 1)?.try_squeeze(1isize)?;
let (h_final, _c_final) = lstm.cell.forward((x_step, (h0, c0)))?;
assert_eq!(h_final.dims().as_ref(), &[2, 6]);
# Ok::<(), incin::Error>(())
```

## Custom modules with `#[module]`

The `#[module]` attribute macro derives composable module traits for structs by
recursively walking its fields. Built-in layers (`Linear`, `Conv2d`, `Dropout`),
nested modules, and `Sequential` blocks are aggregated automatically; non-tensor
fields (or fields marked with `#[module(ignore)]`) are skipped.

### Struct-level arguments

You can selectively disable derived trait implementations:

| Argument | Disables | Purpose |
|:---|:---|:---|
| `no_to_device` | `ToDevice` | Use when struct fields cannot be transferred across device boundaries or when device transfer is handled manually. |
| `no_train_mode` | `TrainMode` | Disables recursive `train()` and `eval()` mode switching. |
| `no_stats` | `ComputeStats` | Disables parameter count and FLOP estimation. |
| `no_parameters` | `VisitParameters` | Disables parameter discovery for optimizers (e.g. for stateless or inference-only modules). |
| `no_state` | `VisitState` / `VisitStateMut` | Disables checkpoint saving and loading visitors. |
| `no_named_layers` | `NamedLayers` | Disables layer hierarchy introspection. |
| `no_shape_info` | `ShapeInfo` | Disables static shape debug formatting. |

### Field-level attributes

* `#[module(ignore)]`: Ignores the field during all module visitor traversals (parameters, state, training modes, device transfer).
* `#[state(name = "custom_name")]`: Overrides the deterministic key used in state snapshots and checkpoints for this field.
* `#[parallel(mesh = "m", stage = 0)]`: Declares distributed pipeline and mesh stage placement for the field.
* `#[shard(axis = "dp")]`: Specifies distributed tensor sharding along a mesh axis.

### Training and evaluation modes (`TrainMode`)

Modules implement `TrainMode` by default, providing:
* `model.train()` (or `model.set_training(true)`): Switches the model and all nested submodules into training mode.
* `model.eval()` (or `model.set_training(false)`): Switches the model and all nested submodules into evaluation/inference mode.

Leaf layers without mode-dependent behavior (`Linear`, `Conv2d`, activations, normalization) implement `TrainMode` as a zero-cost no-op.

Mode-sensitive layers like `Dropout` actively respond to `TrainMode`:
* In **training mode** (`train()`), Dropout randomly zeroes activations with probability $p$ and scales surviving elements by $\frac{1}{1 - p}$.
* In **evaluation mode** (`eval()`), Dropout acts as a zero-overhead identity pass-through.

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

#[module]
pub struct MLP {
    fc1: Linear<s![768, 256], B>,
    drop: Dropout,
    fc2: Linear<s![256, 10], B>,
}

impl MLP {
    pub fn new() -> Result<Self> {
        Ok(Self {
            fc1: Linear::build(())?,
            drop: Dropout::new(0.2),
            fc2: Linear::build(())?,
        })
    }

    pub fn forward(&self, x: Tensor<s![2, 768], B>) -> Result<Tensor<s![2, 10], B, f32, Grad>> {
        let h = self.fc1.forward(x)?;
        let h = ReLU.forward(h)?;
        let h = self.drop.forward(h)?;
        self.fc2.forward(h)
    }
}

# fn main() -> Result<()> {
let mut model = MLP::new()?;
let x = Tensor::<s![2, 768], B>::ones(())?;

// Switch to evaluation mode for validation/inference
model.eval();
let y_eval = model.forward(x.clone())?;
assert_eq!(y_eval.dims().as_ref(), &[2, 10]);

// Switch back to training mode
model.train();
let _optimizer = AdamW::<B>::from_module(&model, 1e-2)?;
# Ok(())
# }
```
