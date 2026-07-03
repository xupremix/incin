use kindle::prelude::*;

#[kindle::module]
struct ResNetBlock {
    // fields would go here
}

impl ResNetBlock {
    #[kindle::forward]
    fn forward(&self, x: Tensor<s![dyn, 64, 224, 224]>) -> Result<Tensor<s![dyn, 64, 224, 224]>> {
        // Just return the tensor to prove compilation boundary works!
        Ok(x)
    }
}

fn main() {
    let t: Tensor<Dyn> = Tensor::zeros([32, 64, 224, 224]).unwrap();
    let t_static: Tensor<s![dyn, 64, 224, 224]> = t.into_shape().unwrap();
    
    let block = ResNetBlock {};
    let out = block.forward(t_static).unwrap();
    
    println!("ResNet output shape: {:?}", out.dims());
}
