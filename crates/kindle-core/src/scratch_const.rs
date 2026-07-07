pub struct Tensor<const N: usize, const SHAPE: [usize; N]>;

fn main() {
    let t: Tensor<2, { [10, 20] }> = Tensor;
}
