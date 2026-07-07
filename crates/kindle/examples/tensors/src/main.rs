use kindle::prelude::*;

#[module]
struct MyModel {
    l1: Linear<s![10, 20]>,
    l2: Linear<s![20, 20]>,
    l3: Linear<s![20, 20]>,
    l4: Linear<s![20, 10]>,
}

impl MyModel {
    pub fn forward(&self, x: Tensor<s![dyn, 10]>) -> kindle::Result<Tensor<s![dyn, 10]>> {
        let x = self.l1.forward(x)?;
        let x = self.l2.forward(x)?;
        let x = self.l3.forward(x)?;
        let x = self.l4.forward(x)?;

        Ok(x)
    }
}

fn main() -> kindle::Result<()> {

    let _model: MyModel = MyModel {
        l1: Linear::new()?,
        l2: Linear::new()?,
        l3: Linear::new()?,
        l4: Linear::new()?,
    };

    Ok(())
}
