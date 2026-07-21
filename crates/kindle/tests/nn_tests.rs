use kindle::prelude::*;

/// Implementation of `CpuBackend` for the respective backend.
type CpuBackend = kindle_backends::cpu::CpuBackend<f32, Cpu>;

#[test]
/// Test param.
fn test_param() -> Result<()> {
    // Test creating a Param from zeros
    let param = Param::<s![10, 10], CpuBackend>::zeros(())?;

    // Test getting a tensor out
    let t = param.as_tensor()?;
    let t_dims: [usize; 2] = t.dims();
    assert_eq!(t_dims, [10, 10]);

    Ok(())
}

#[test]
/// Test linear.
fn test_linear() -> Result<()> {
    let linear = Linear::<s![10, 5], CpuBackend>::new()?;
    let input = Tensor::<s![2, 10], CpuBackend>::ones(())?;

    let out = linear.forward(input)?;
    let out_dims: [usize; 2] = out.dims();
    assert_eq!(out_dims, [2, 5]);

    Ok(())
}

#[test]
/// Test conv2d.
fn test_conv2d() -> Result<()> {
    // 3 InChannels, 16 OutChannels, 3x3 Kernel, Stride=1, Padding=1, Dilation=1
    // (OutC, InC, K, S, P, D)
    /// Conv shape.
    type ConvShape = (
        kindle::prelude::typenum::U16,
        kindle::prelude::typenum::U3,
        kindle::prelude::typenum::U3,
        kindle::prelude::typenum::U1,
        kindle::prelude::typenum::U1,
        kindle::prelude::typenum::U1,
    );
    let conv = Conv2d::<ConvShape, CpuBackend>::new()?;

    // Input: Batch=2, Channels=3, H=32, W=32
    let input = Tensor::<s![2, 3, 32, 32], CpuBackend>::ones(())?;

    let out = conv.forward(input)?;
    let out_dims: [usize; 4] = out.dims();
    // Out: Batch=2, Channels=16, H=32, W=32 (because kernel=3, padding=1, stride=1)
    assert_eq!(out_dims, [2, 16, 32, 32]);

    Ok(())
}

#[test]
/// Test layer norm.
fn test_layer_norm() -> Result<()> {
    let ln = LayerNorm::<s![20], CpuBackend>::new(1e-5)?;
    let input = Tensor::<s![5, 10, 20], CpuBackend>::ones(())?;

    let out = ln.forward(input)?;
    let out_dims: [usize; 3] = out.dims();
    assert_eq!(out_dims, [5, 10, 20]);

    Ok(())
}

#[test]
/// Test batch norm2d.
fn test_batch_norm2d() -> Result<()> {
    // 16 Channels
    let bn = BatchNorm2d::<s![16], CpuBackend>::new(1e-5, 0.1)?;

    // Input: Batch=2, Channels=16, H=32, W=32
    let input = Tensor::<s![2, 16, 32, 32], CpuBackend>::ones(())?;

    let out = bn.forward(input)?;
    let out_dims: [usize; 4] = out.dims();
    assert_eq!(out_dims, [2, 16, 32, 32]);

    Ok(())
}

#[test]
/// Test sequential.
fn test_sequential() -> Result<()> {
    let seq = seq!(
        Linear::<s![10, 5], CpuBackend>::new()?,
        ReLU,
        Linear::<s![5, 2], CpuBackend>::new()?
    );

    let input = Tensor::<s![4, 10], CpuBackend>::ones(())?;

    let out = seq.forward(input)?;
    let out_dims: [usize; 2] = out.dims();
    assert_eq!(out_dims, [4, 2]);

    Ok(())
}

#[test]
/// Test embedding.
fn test_embedding() -> Result<()> {
    // Vocab=100, EmbedDim=32
    /// Embed shape.
    type EmbedShape = (
        kindle::prelude::typenum::U100,
        kindle::prelude::typenum::U32,
    );
    let weight = Param::<EmbedShape, CpuBackend>::randn(())?;
    let emb = Embedding::<EmbedShape, CpuBackend> { weight };
    // Input: Batch=2, SeqLen=10
    let input = Tensor::<s![2, 10], CpuBackend>::ones(())?;

    let out = emb.forward(input)?;
    let out_dims: [usize; 3] = out.dims();
    assert_eq!(out_dims, [2, 10, 32]);

    Ok(())
}
