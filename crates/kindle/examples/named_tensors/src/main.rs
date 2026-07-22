extern crate alloc;
use kindle::prelude::*;

symbolic_dim!(Batch, Seq, Feature);

fn main() {
    let _dev = DeviceId::cpu();

    let t1: Tensor<s![Batch, 10]> = Tensor::zeros((32usize, ())).unwrap();
    let _t2: Tensor<s![Batch, 20]> = Tensor::zeros((32usize, ())).unwrap();
    let t3: Tensor<s![Batch, 10]> = Tensor::zeros((32usize, ())).unwrap();

    // Should compile
    let _t4 = t1.add(&t3).unwrap();

    // Should fail with shape mismatch
    // let _t5 = t1.add(&t2).unwrap(); // This correctly fails to compile!
    println!("Compiled successfully!");
}
