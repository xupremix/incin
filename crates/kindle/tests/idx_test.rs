use kindle::prelude::*;

#[test]
fn test_idx_macro() -> Result<()> {
    // Generate a tensor of shape [2, 3, 4]
    let t: Tensor<Dyn> = Tensor::zeros([2, 3, 4])?;
    
    // Test slice using idx macro: t[:, 1..2, 0]
    let sliced = t.slice(idx![.., 1..2, 0])?;
    
    // The shape should be [2, 1] because the third dimension was dropped by idx![0]
    let dims = sliced.dims();
    assert_eq!(dims, &[2, 1]);
    
    Ok(())
}
