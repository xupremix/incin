use kindle::prelude::*;

fn main() {
    let t: Tensor<Dyn> = Tensor::zeros([2, 3]).unwrap();
    println!("{t:?}");
}
