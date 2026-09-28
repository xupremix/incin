//! Example: split one linear layer two ways and check both paths agree.
//!
//! A `[2, 8]` input through a `[4, 8]` weight (a Linear without bias) is
//! computed three times: whole, column-split over the output dim (the two
//! shards concatenate back), and row-split over the input dim (the two
//! partial products add back). All three must match.
//!
//! Run it with `cargo run -p incin --example tensor_parallel_split`.

use incin::prelude::*;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() -> Result<()> {
    // Fixed values, no randomness: W[i, j] = (i * 8 + j) * 0.01.
    let w_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
    let x_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.5).collect();
    let w = Tensor::<Dyn, DefaultBackend>::from_slice(&w_data, vec![4, 8])?;
    let x = Tensor::<Dyn, DefaultBackend>::from_slice(&x_data, vec![2, 8])?;

    // 1. Whole layer: y = x @ W^T, a [2, 4] output.
    let full = x.matmul(&w.transpose(0isize, 1isize)?)?;
    assert_eq!(full.dims().as_ref(), &[2, 4]);
    let expected = full.to_vec1::<f32>()?;
    println!("full output shape: {:?}", full.dims());

    // 2. Column split (tensor parallelism over outputs): W becomes two
    // [2, 8] shards, each half of y is computed separately, then joined.
    let w_shards = w.chunk(2, 0isize)?;
    assert_eq!(w_shards.len(), 2);
    let y_a = x.matmul(&w_shards[0].transpose(0isize, 1isize)?)?;
    let y_b = x.matmul(&w_shards[1].transpose(0isize, 1isize)?)?;
    let joined = y_a.concat(&y_b, 1isize)?;
    assert_eq!(joined.dims().as_ref(), &[2, 4]);
    let got = joined.to_vec1::<f32>()?;
    let drift = max_abs_diff(&got, &expected);
    println!("column split max drift vs whole: {drift:e}");
    assert!(drift < 1e-5, "column shards must rebuild the whole output");

    // 3. Row split (tensor parallelism over inputs): x becomes two [2, 4]
    // halves, W becomes two [4, 4] halves, and the partial products add.
    let x_shards = x.chunk(2, 1isize)?;
    let w_halves = w.chunk(2, 1isize)?;
    assert_eq!((x_shards.len(), w_halves.len()), (2, 2));
    let p_a = x_shards[0].matmul(&w_halves[0].transpose(0isize, 1isize)?)?;
    let p_b = x_shards[1].matmul(&w_halves[1].transpose(0isize, 1isize)?)?;
    let summed = p_a.try_add(&p_b)?;
    assert_eq!(summed.dims().as_ref(), &[2, 4]);
    let got = summed.to_vec1::<f32>()?;
    let drift = max_abs_diff(&got, &expected);
    println!("row split max drift vs whole: {drift:e}");
    assert!(drift < 1e-5, "row partials must sum to the whole output");

    println!("PASS: column and row splits both match the whole layer");
    Ok(())
}
