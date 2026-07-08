use kindle::prelude::*;

type B = DefaultBackend;

#[module]
struct MyModel {
    l1: Linear<s![10, 20], B>,
    l2: Linear<s![20, 20], B>,
    l3: Linear<s![20, 20], B>,
    l4: Linear<s![20, 10], B>,
}

impl MyModel {
    #[allow(dead_code)]
    pub fn forward(
        &self,
        x: Tensor<s![2, 2, 2, dyn, 10]>,
    ) -> kindle::Result<Tensor<s![2, 2, 2, dyn, 10]>> {
        let x = self.l1.forward(x)?;
        let x = self.l2.forward(x)?;
        let x = self.l3.forward(x)?;
        let x = self.l4.forward(x)?;

        Ok(x)
    }
}

fn main() -> kindle::Result<()> {
    let model = MyModel {
        l1: Linear::new()?,
        l2: Linear::new()?,
        l3: Linear::new()?,
        l4: Linear::new()?,
    };

    let t: Tensor<s![2, 2, 2, dyn, 10], B> = Tensor::randn(10_usize)? * 2.;

    let out = model.forward(t)?;
    println!("=== Display ===");
    println!("{out}");
    println!("=== Debug ===");
    println!("{:?}", out);

    Ok(())
}
