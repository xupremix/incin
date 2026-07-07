use kindle::prelude::*;

fn main() -> Result<()> {
    // 1. Batched Matrix multiplied by a Batched Matrix (Static)
    let t1 = Tensor::<s![3, 10, 20]>::zeros(())?;
    let t2 = Tensor::<s![20, 5]>::zeros(())?;

    // The macro generated output shape (3, 10, 5)
    let t3 = t1.matmul(&t2)?;
    println!("t1 (static) shape: {:?}", t1.dims());
    println!("t2 (static) shape: {:?}", t2.dims());
    println!("t3 (static) shape after matmul: {:?}", t3.dims());

    // 2. Dynamic batch dimension broadcasting
    // [Batch, 10, 20] where Batch=32
    let dyn_t1: Tensor<s![dyn, 10, 20]> = Tensor::zeros((32usize, (), ()))?;
    // [20, 5] (static weight matrix)
    let static_weights: Tensor<s![20, 5]> = Tensor::zeros(())?;

    // The macro generated output shape [Batch, 10, 5]
    let dyn_t3 = dyn_t1.matmul(&static_weights)?;

    println!("dyn_t1 shape: {:?}", dyn_t1.dims());
    println!("static_weights shape: {:?}", static_weights.dims());
    println!("dyn_t3 shape after broadcasted matmul: {:?}", dyn_t3.dims());

    // 3. Multi-Head Attention Style (4 dimensions)
    let q: Tensor<s![2, 4, 16, 8]> = Tensor::zeros(())?;
    let k_t: Tensor<s![2, 4, 8, 16]> = Tensor::zeros(())?;

    let attn_scores = q.matmul(&k_t)?;
    println!("q shape: {:?}", q.dims());
    println!("k_t shape: {:?}", k_t.dims());
    println!("attn_scores shape: {:?}", attn_scores.dims());

    Ok(())
}
