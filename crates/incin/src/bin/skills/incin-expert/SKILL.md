---
name: incin-expert
description: Expert AI assistant skill for building deep learning models, training pipelines, and tensor applications using Incin in Rust. Use when writing, debugging, or optimizing Incin code, defining neural networks, using the Target API, configuring static s![] vs runtime shape! dimensions, or setting up autograd.
---

# Incin Expert Guide for Model Developers

Use this skill when developing models, training loops, neural networks, or high-performance tensor algebra using the `incin` crate in Rust.

---

## 1. Core Idioms & Target-First Construction

Incin builds tensors directly from a target device handle:

```rust
use incin::prelude::*;

fn main() -> Result<(), IncinError> {
    // 1. Static shapes (compile-time verified)
    let a: Tensor<s![4, 8], Cpu> = Cpu.randn(())?;
    let b: Tensor<s![8, 2], Cpu> = Cpu.randn(())?;
    let c = a.matmul(&b)?; // Result is Tensor<s![4, 2], Cpu>

    // 2. Dynamic runtime shapes
    let x: Tensor<Dyn, Cpu> = Cpu.randn(shape![16, 3, 224, 224])?;
    
    // 3. GPU device targets (CUDA / WGPU / Metal)
    #[cfg(feature = "cuda")]
    {
        let cuda = Cuda::default();
        let gpu_tensor: Tensor<s![32, 512], Cuda> = cuda.zeros(())?;
    }

    Ok(())
}
```

---

## 2. Neural Networks & Layers (`incin::nn`)

Build modular models implementing the `Module` trait:

```rust
use incin::prelude::*;
use incin::nn::{Linear, Conv2d, Sequential, Module};

// Define a Convolutional Block
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<s![3, 64, 3, 3], B>,
    linear: Linear<s![64 * 14 * 14, 10], B>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(target: &B) -> Result<Self, IncinError> {
        Ok(Self {
            conv: Conv2d::new(target)?,
            linear: Linear::new(target)?,
        })
    }

    pub fn forward<S>(&self, input: &Tensor<S, B>) -> Result<Tensor<s![16, 10], B>, IncinError>
    where
        S: Shape,
    {
        let x = self.conv.forward(input)?.relu()?;
        let x = x.max_pool2d(s![2, 2])?;
        let x = x.flatten(s![1, 3])?;
        self.linear.forward(&x)
    }
}
```

---

## 3. Training Loop, Autograd & Optimization

```rust
use incin::prelude::*;
use incin::optim::{Adam, Optimizer};

fn train_step<B: Backend>(
    model: &mut MyModel<B>,
    optimizer: &mut Adam<B>,
    batch_data: &Tensor<s![32, 784], B>,
    batch_targets: &Tensor<s![32, 10], B>,
) -> Result<f32, IncinError> {
    // 1. Forward pass with gradient tracking
    let predictions = model.forward(batch_data)?;
    
    // 2. Compute loss
    let loss = predictions.cross_entropy_loss(batch_targets)?;
    let loss_scalar = loss.to_scalar()?;

    // 3. Backward pass to compute gradients
    loss.backward()?;

    // 4. Optimizer step and zero gradients
    optimizer.step(model.parameters_mut())?;
    optimizer.zero_grad(model.parameters_mut());

    Ok(loss_scalar)
}
```

---

## 4. Troubleshooting Shape Mismatches

1. **Typenum Errors**: If rustc shows `UInt<UInt<...>>`, run `cargo incin check` or `cargo incin check --explain` for humanized shape errors.
2. **Inner Dimension Mismatches**: In `a.matmul(&b)`, ensure dimension `K` matches between `[M, K]` and `[K, N]`.
3. **Reshape Elements**: `a.reshape(s![...])` requires the product of source dimensions to equal the product of target dimensions.
