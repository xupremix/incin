use kindle::prelude::*;
use kindle_backends::dummy::DummyBackend;

fn main() {
    let t: Tensor<s![2, 3, 4], DummyBackend<f32, Cpu>, Grad> = Tensor::zeros(()).unwrap();
    println!("Original shape: {:?}", t.dims());

    // reshape to (2, 12)
    let t2 = t.reshape_idx::<idx![2, -1]>().unwrap();
    println!("Reshaped shape: {:?}", t2.dims());
    assert_eq!(t2.dims().as_slice(), &[2, 12]);
    
    // slice to (1, 12) via idx![0..1, ..]
    let t3 = t2.slice_idx::<idx![0..1, ..]>().unwrap();
    println!("Sliced shape: {:?}", t3.dims());
    assert_eq!(t3.dims().as_slice(), &[1, 12]);
}
