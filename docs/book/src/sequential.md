# Sequential models

`Sequential<L1, L2>` chains two layers, and only two  -  a three-or-more-layer
model needs right-nesting, which is exactly what `seq!`/`SeqTy!` automate.
They mirror each other: `seq!` builds the **value**, `SeqTy!` names the
**type**, from the same flat layer list, so you never hand-nest either one.

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

type Net = SeqTy!(
    Linear<s![768, 256], B>,
    ReLU,
    Linear<s![256, 10], B>
);

let net: Net = seq!(
    Linear::<s![768, 256], B>::build(())?,
    ReLU,
    Linear::<s![256, 10], B>::build(())?
);

let x = Tensor::<s![2, 768], B>::ones(())?;
let y = net.forward(x)?;
assert_eq!(y.dims().as_ref(), &[2, 10]);
# Ok::<(), incin::Error>(())
```

## Inside a `#[module]` struct

`SeqTy!` is what lets a `Sequential` field's type be written the same way
the value is constructed, instead of hand-nesting `Sequential<A,
Sequential<B, C>>` separately and having the two drift:

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

#[module(no_shape_info)]
pub struct MLP {
    net: SeqTy!(
        Linear<s![768, 256], B>,
        ReLU,
        Linear<s![256, 10], B>
    ),
}

impl MLP {
    pub fn new() -> Result<Self> {
        Ok(Self {
            net: seq!(
                Linear::<s![768, 256], B>::build(())?,
                ReLU,
                Linear::<s![256, 10], B>::build(())?
            ),
        })
    }

    pub fn forward(&self, x: Tensor<s![2, 768], B>) -> Result<Tensor<s![2, 10], B, f32, Grad>> {
        self.net.forward(x)
    }
}
# fn main() -> Result<()> {
let model = MLP::new()?;
let x = Tensor::<s![2, 768], B>::ones(())?;
let y = model.forward(x)?;
assert_eq!(y.dims().as_ref(), &[2, 10]);
# Ok(())
# }
```
