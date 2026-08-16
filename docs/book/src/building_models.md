# Layers and `#[module]`

Every built-in layer follows the same two-step shape: a `Shape` type
parameter names what's static, and a `build(args)` constructor takes
whatever isn't. `args` uses the same flexible-argument system as tensor
constructors  -  pass exactly the runtime values the static shape didn't
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
whatever the static shape didn't already fix  -  an epsilon (and, for batch
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
integer dtype  -  the underlying kernel reads any dtype it can convert to an
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
`forward` takes the initial hidden and cell state explicitly  -  there's no
implicit "start from zero"  -  and returns both the full output sequence and
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
let x_step = x.try_narrow(1, 0, 1)?.try_squeeze(1)?;
let (h_final, _c_final) = lstm.cell.forward((x_step, (h0, c0)))?;
assert_eq!(h_final.dims().as_ref(), &[2, 6]);
# Ok::<(), incin::Error>(())
```

## Custom modules with `#[module]`

`#[module]` derives typed state and parameter visitors for a struct by walking
its fields: built-in layers and nested `Sequential` values are delegated to
recursively; anything else is skipped. State consumers and optimizer adapters
are built on those visitors.

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

#[module(no_shape_info)]
pub struct MLP {
    fc1: Linear<s![768, 256], B>,
    fc2: Linear<s![256, 10], B>,
}

impl MLP {
    pub fn new() -> Result<Self> {
        Ok(Self {
            fc1: Linear::build(())?,
            fc2: Linear::build(())?,
        })
    }

    pub fn forward(&self, x: Tensor<s![2, 768], B>) -> Result<Tensor<s![2, 10], B, f32, Grad>> {
        let h = self.fc1.forward(x)?;
        let h = ReLU.forward(h)?;
        self.fc2.forward(h)
    }
}
# fn main() -> Result<()> {
let model = MLP::new()?;
let x = Tensor::<s![2, 768], B>::ones(())?;
let y = model.forward(x)?;
assert_eq!(y.dims().as_ref(), &[2, 10]);
let _optimizer = AdamW::<B>::from_module(&model, 1e-2)?;
# Ok(())
# }
```
