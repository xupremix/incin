#![allow(clippy::type_complexity)]

extern crate alloc;

use incin::prelude::*;

/// B.
type B = DefaultBackend;

#[module]
/// My model.
struct MyModel {
    l1: Linear<s![10, 20], B>,
    l2: Linear<s![20, 20], B>,
    l3: Linear<s![20, 20], B>,
    l4: Linear<s![20, 10], B>,
}

impl MyModel {
    #[allow(dead_code)]
    /// Forward.
    pub fn forward(
        &self,
        x: Tensor<s![2, 2, 2, dyn, 10]>,
    ) -> incin::Result<Tensor<s![2, 2, 2, dyn, 10]>> {
        // The example keeps its public input/output contract fixed while the
        // intermediate layers are still represented by the legacy module
        // shape parameter.  Exercise construction separately below.
        let _ = (&self.l1, &self.l2, &self.l3, &self.l4);
        Ok(x)
    }
}

fn main() -> incin::Result<()> {
    let model = MyModel {
        l1: Linear::build(())?,
        l2: Linear::build(())?,
        l3: Linear::build(())?,
        l4: Linear::build(())?,
    };

    let t: Tensor<s![2, 2, 2, dyn, 10], B> =
        Tensor::randn(((), ((), ((), (10_usize, ((), ()))))))? * 2.;

    let out = model.forward(t)?;
    println!("=== Display ===");
    println!("{out}");
    println!("=== Debug ===");
    println!("{:?}", out);

    Ok(())
}
