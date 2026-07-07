use kindle::prelude::*;

type CpuBackend = DefaultBackend;

fn to_vec(t: &Tensor<Dyn, CpuBackend>) -> Vec<f32> {
    t.inner().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn test_unary_ops() -> Result<()> {
    let ones: Tensor<s![10, 20], CpuBackend> = Tensor::ones(())?;
    let zeros: Tensor<s![10, 20], CpuBackend> = Tensor::zeros(())?;
    
    let neg_ones = ones.neg()?;
    let abs_t = neg_ones.abs()?;
    assert_eq!(to_vec(&abs_t.into_dyn())[0], 1.0);
    
    let relu_neg = neg_ones.relu()?;
    assert_eq!(to_vec(&relu_neg.into_dyn())[0], 0.0);
    let relu_pos = ones.relu()?;
    assert_eq!(to_vec(&relu_pos.into_dyn())[0], 1.0);
    
    let gelu_z = zeros.gelu()?;
    assert_eq!(to_vec(&gelu_z.into_dyn())[0], 0.0);
    
    let swish_z = zeros.swish()?;
    assert_eq!(to_vec(&swish_z.into_dyn())[0], 0.0);
    
    let sm = zeros.softmax(1)?;
    let val = to_vec(&sm.into_dyn())[0];
    assert!((val - 0.05).abs() < 1e-4);
    
    assert_eq!(to_vec(&neg_ones.into_dyn())[0], -1.0);
    
    let sqrt_t = ones.sqrt()?;
    assert_eq!(to_vec(&sqrt_t.into_dyn())[0], 1.0);
    
    let exp_z = zeros.exp()?;
    assert_eq!(to_vec(&exp_z.into_dyn())[0], 1.0);
    
    let log_o = ones.log()?;
    assert_eq!(to_vec(&log_o.into_dyn())[0], 0.0);
    
    let tanh_z = zeros.tanh()?;
    assert_eq!(to_vec(&tanh_z.into_dyn())[0], 0.0);
    
    let sig_z = zeros.sigmoid()?;
    assert_eq!(to_vec(&sig_z.into_dyn())[0], 0.5);
    
    let mul_s = ones.mul_scalar(5.0)?;
    assert_eq!(to_vec(&mul_s.into_dyn())[0], 5.0);
    
    let add_s = zeros.add_scalar(3.14)?;
    assert!((to_vec(&add_s.into_dyn())[0] - 3.14).abs() < 1e-4);

    Ok(())
}

#[test]
fn test_binary_ops() -> Result<()> {
    let twos = Tensor::<s![5, 5], CpuBackend>::ones(())?.mul_scalar(2.0)?;
    let threes = Tensor::<s![5, 5], CpuBackend>::ones(())?.mul_scalar(3.0)?;
    
    let add_t = twos.add(&threes)?;
    assert_eq!(to_vec(&add_t.into_dyn())[0], 5.0);
    
    let sub_t = threes.sub(&twos)?;
    assert_eq!(to_vec(&sub_t.into_dyn())[0], 1.0);
    
    let mul_t = twos.mul(&threes)?;
    assert_eq!(to_vec(&mul_t.into_dyn())[0], 6.0);
    
    let div_t = threes.div(&twos)?;
    assert_eq!(to_vec(&div_t.into_dyn())[0], 1.5);

    Ok(())
}

#[test]
fn test_broadcast_ops() -> Result<()> {
    let matrix = Tensor::<s![5, 5], CpuBackend>::ones(())?;
    let scalar = Tensor::<s![5, 5], CpuBackend>::ones(())?.mul_scalar(9.0)?; // Use matching shapes for now
    
    let b_add = matrix.broadcast_add(&scalar)?;
    assert_eq!(to_vec(&b_add.into_dyn())[0], 10.0);
    
    // Testing std::ops
    let b_sub = matrix - scalar;
    assert_eq!(to_vec(&b_sub.into_dyn())[0], -8.0);
    
    Ok(())
}

#[test]
fn test_reduction_ops() -> Result<()> {
    let t = Tensor::<s![10, 20], CpuBackend>::ones(())?;
    
    let sum_all = t.clone().sum_all()?;
    assert_eq!(to_vec(&sum_all.into_dyn())[0], 200.0);
    
    let mean_all = t.clone().mean_all()?;
    assert_eq!(to_vec(&mean_all.into_dyn())[0], 1.0);
    
    let sum_dim1 = t.clone().sum_dim::<1>()?;
    let val = to_vec(&sum_dim1.into_dyn());
    assert_eq!(val[0], 20.0);
    assert_eq!(val.len(), 10);
    
    let max_all = t.clone().max_all()?;
    assert_eq!(to_vec(&max_all.into_dyn())[0], 1.0);
    
    Ok(())
}

#[test]
fn test_manipulation_ops() -> Result<()> {
    let t = Tensor::<s![10, 20], CpuBackend>::ones(())?;
    
    // Reshape [10, 20] -> [200]
    let r = t.reshape_idx::<idx![200]>()?;
    assert_eq!(r.rank(), 1);
    
    // Transpose
    let tr = t.transpose::<0, 1>()?;
    let tr_dims: [usize; 2] = tr.dims().into();
    assert_eq!(tr_dims, [20, 10]);
    
    // Flatten
    let f = t.flatten::<0, 1>()?;
    let f_dims: [usize; 1] = f.dims().into();
    assert_eq!(f_dims, [200]);
    
    // Narrow (extract 5 elements from dim 1 dynamically)
    let n = t.clone().try_narrow(1, 0, 5)?;
    let n_dims: Vec<usize> = n.dims().into();
    assert_eq!(n_dims, vec![10, 5]);
    
    Ok(())
}

#[test]
fn test_indexing_slicing() -> Result<()> {
    let t1 = Tensor::<s![10, 20], CpuBackend>::ones(())?;
    let t2 = Tensor::<s![10, 20], CpuBackend>::ones(())?;
    
    // Concat on dim 0 -> [20, 20]
    let c = t1.concat::<s![10, 20], kindle::prelude::typenum::U0>(&t2)?;
    let c_dims: [usize; 2] = c.dims().into();
    assert_eq!(c_dims, [20, 20]);
    
    // Stack on dim 0 -> [2, 10, 20]
    let s = t1.stack::<kindle::prelude::typenum::U0>(&t2)?;
    let s_dims: [usize; 3] = s.dims().into();
    assert_eq!(s_dims, [2, 10, 20]);
    
    // Try Concat Slice
    let c_slice = Tensor::<s![10, 20], CpuBackend>::try_concat_slice(&[&t1, &t2], 0)?;
    let c_slice_dims: Vec<usize> = c_slice.dims().into();
    assert_eq!(c_slice_dims, vec![20, 20]);
    
    Ok(())
}

#[test]
fn test_loss_functions() -> Result<()> {
    let pred = Tensor::<s![10, 20], CpuBackend>::ones(())?;
    let target = Tensor::<s![10, 20], CpuBackend>::zeros(())?;
    
    // MSE Loss
    let mse = pred.mse_loss(&target)?;
    assert_eq!(to_vec(&mse.into_dyn())[0], 1.0);
    
    Ok(())
}
