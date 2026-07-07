use kindle::prelude::*;

symbolic_dim!(Batch, Seq, Feature);

fn main() {
    let dev = KindleDevice::cpu();

    let t1: Tensor<s![sym Batch, 10]> = Tensor::zeros((32usize, ())).unwrap();
    let t2: Tensor<s![sym Batch, 20]> = Tensor::zeros((32usize, ())).unwrap();
    let t3: Tensor<s![sym Batch, 10]> = Tensor::zeros((32usize, ())).unwrap();

    // Should compile
    let _t4 = t1.add(&t3).unwrap();

    // Should fail with shape mismatch
    // let _t5 = t1.add(&t2).unwrap(); // This correctly fails to compile!
    println!("Compiled successfully!");
}
