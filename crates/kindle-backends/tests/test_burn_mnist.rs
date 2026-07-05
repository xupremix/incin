#![cfg(feature = "burn")]

use kindle_core::prelude::*;
use kindle_backends::burn_backend::BurnBackend;
use burn_ndarray::NdArray;
use burn::backend::Autodiff;
use kindle_core::tensor::device::Cpu;

type BackendType = Autodiff<NdArray>;
type BurnNdarrayBackend = BurnBackend<BackendType, f32, Cpu>;

fn test_burn_mnist_mock() -> Result<()> {
    // Model definition: A simple 2-layer MLP for "MNIST"
    // Input: [batch_size, 784]
    // Output: [batch_size, 10]
    let k_device = KindleDevice::cpu();
    
    // We simulate Kindle's Linear layer manually using backend ops.
    // Use ones/zeros since rand/randn are not implemented yet in the dummy
    let w1 = BurnNdarrayBackend::var_ones(&[784, 128], KindleDType::F32, &k_device)?;
    let b1 = BurnNdarrayBackend::var_zeros(&[128], KindleDType::F32, &k_device)?;
    
    let w2 = BurnNdarrayBackend::var_ones(&[128, 10], KindleDType::F32, &k_device)?;
    let b2 = BurnNdarrayBackend::var_zeros(&[10], KindleDType::F32, &k_device)?;
    
    let batch_size = 32;
    let x = BurnNdarrayBackend::ones(&[batch_size, 784], KindleDType::F32, &k_device)?;
    let target = BurnNdarrayBackend::zeros(&[batch_size], KindleDType::F32, &k_device)?; 
    
    let mut params = vec![w1, b1, w2, b2];
    
    // Forward pass
    for step in 0..2 {
        let w1_t = BurnNdarrayBackend::var_as_tensor(&params[0])?;
        let b1_t = BurnNdarrayBackend::var_as_tensor(&params[1])?;
        let w2_t = BurnNdarrayBackend::var_as_tensor(&params[2])?;
        let b2_t = BurnNdarrayBackend::var_as_tensor(&params[3])?;
        
        let h = BurnNdarrayBackend::matmul(&x, &w1_t)?;
        let b1_reshaped = BurnNdarrayBackend::reshape(&b1_t, &[1, 128])?;
        let h = BurnNdarrayBackend::add(&h, &b1_reshaped)?; // Burn supports broadcasting in Add
        let h = BurnNdarrayBackend::relu(&h)?;
        
        let out = BurnNdarrayBackend::matmul(&h, &w2_t)?;
        let b2_reshaped = BurnNdarrayBackend::reshape(&b2_t, &[1, 10])?;
        let out = BurnNdarrayBackend::add(&out, &b2_reshaped)?;
        
        let loss = BurnNdarrayBackend::cross_entropy_loss(&out, &target)?;
        
        let grads = BurnNdarrayBackend::backward(&loss)?;
        BurnNdarrayBackend::step_adamw(&mut params, &grads, 1e-3)?;
        
        println!("Step {} ran successfully, loss computed.", step);
    }
    
    Ok(())
}

#[test]
fn run_test_burn_mnist_mock() {
    assert!(test_burn_mnist_mock().is_ok());
}
