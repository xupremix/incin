// #113's other half: `transpose_view` deliberately returns `Dyn`, because a
// view is not contiguous. `reshape_view` is bounded on `L: Contiguous`, so
// accepting this would open a path where the layout claim and the strides
// disagree. The compile error is the contract.
use incin::advanced::{Here, Next};
use incin::prelude::*;

fn main() -> Result<()> {
    let t = Cpu.tensor([[1.0_f32, 2.0], [3.0, 4.0]])?;
    let viewed = t.transpose_view::<Here, Next<Here>>()?;
    let _ = viewed.reshape_view::<s![4]>()?;
    Ok(())
}
