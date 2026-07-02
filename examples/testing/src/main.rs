use kindle::prelude::*;

fn main() {
    let t: Tensor<Dyn, f32, Cpu, Dyn> = Tensor::new(([2, 3], true));
    println!("{t:#?}");
}
