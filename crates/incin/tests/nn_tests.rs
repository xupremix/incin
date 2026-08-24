//! Integration coverage for `test_param` on the documented public surface.
#![cfg(feature = "cpu")]

use incin::prelude::*;

/// Implementation of `CpuBackendImpl` for the respective backend.
type CpuBackendImpl = incin_backends::cpu::CpuBackendImpl;

#[test]
/// Test param.
fn test_param() -> Result<()> {
    // Test creating a Param from zeros
    let param = Param::<s![10, 10], CpuBackendImpl>::zeros(())?;

    // Test getting a tensor out
    let t = param.as_tensor()?;
    assert_eq!(t.dims().dims(), &[10, 10]);

    Ok(())
}

#[test]
/// Test linear.
fn test_linear() -> Result<()> {
    let linear = Linear::<s![10, 5], CpuBackendImpl>::build(())?;
    let input = Tensor::<s![2, 10], CpuBackendImpl>::ones(())?;

    let out = linear.forward(input)?;
    assert_eq!(out.dims().dims(), &[2, 5]);

    Ok(())
}

#[test]
/// Test conv2d.
fn test_conv2d() -> Result<()> {
    // 3 InChannels, 16 OutChannels, 3x3 Kernel, Stride=1, Padding=1, Dilation=1
    // (OutC, InC, K, S, P, D)
    /// Conv shape.
    type ConvShape = s![16, 3, 3, 1, 1, 1];
    let conv = Conv2d::<ConvShape, CpuBackendImpl>::build(())?;

    // Input: Batch=2, Channels=3, H=32, W=32
    let input = Tensor::<s![2, 3, 32, 32], CpuBackendImpl>::ones(())?;

    let out = conv.forward(input)?;
    let out_dims = out.dims();
    // Out: Batch=2, Channels=16, H=32, W=32 (because kernel=3, padding=1, stride=1)
    assert_eq!(out_dims.dims(), &[2, 16, 32, 32]);

    Ok(())
}

#[test]
/// Test layer norm.
fn test_layer_norm() -> Result<()> {
    let ln = LayerNorm::<s![20], CpuBackendImpl>::build(1e-5)?;
    let input = Tensor::<s![5, 10, 20], CpuBackendImpl>::ones(())?;

    let out = ln.forward(input)?;
    assert_eq!(out.dims().dims(), &[5, 10, 20]);

    Ok(())
}

#[test]
/// Test batch norm2d.
fn test_batch_norm2d() -> Result<()> {
    // 16 Channels
    let bn = BatchNorm2d::<s![16], CpuBackendImpl>::build((1e-5, 0.1))?;

    // Input: Batch=2, Channels=16, H=32, W=32
    let input = Tensor::<s![2, 16, 32, 32], CpuBackendImpl>::ones(())?;

    let out = bn.forward(input)?;
    assert_eq!(out.dims().dims(), &[2, 16, 32, 32]);

    Ok(())
}

#[test]
/// Test sequential.
fn test_sequential() -> Result<()> {
    let seq = seq!(
        Linear::<s![10, 5], CpuBackendImpl>::build(())?,
        ReLU,
        Linear::<s![5, 2], CpuBackendImpl>::build(())?
    );

    let input = Tensor::<s![4, 10], CpuBackendImpl>::ones(())?;

    let out = seq.forward(input)?;
    assert_eq!(out.dims().dims(), &[4, 2]);

    Ok(())
}

#[test]
/// `seq!(Linear, ReLU, Linear)` builds the right-nested value
/// `Sequential(Linear, Sequential(ReLU, Linear))`. Before flat numbering,
/// this produced keys `0.weight/0.bias, 1.1.weight/1.1.bias` (encoding the
/// tree's nesting depth). PyTorch's `nn.Sequential` numbers by flat
/// position instead - index `1` (`ReLU`, no parameters) is simply absent
/// from the state dict rather than renumbering what follows, so the second
/// `Linear` keeps index `2`, not `1`.
fn test_sequential_state_dict_keys_are_flat_like_pytorch() -> Result<()> {
    let seq = seq!(
        Linear::<s![10, 5], CpuBackendImpl>::build(())?,
        ReLU,
        Linear::<s![5, 2], CpuBackendImpl>::build(())?
    );

    let params = ParameterGroup::<CpuBackendImpl, f32>::from_module(&seq)?;
    let mut keys: Vec<&String> = params.iter().map(|(key, _)| key).collect();
    keys.sort();
    assert_eq!(
        keys,
        vec!["0.bias", "0.weight", "2.bias", "2.weight"],
        "expected flat PyTorch-style numbering, got nested keys instead"
    );

    Ok(())
}

#[test]
/// `SeqTy!` must name the exact type `seq!` builds a value of -
/// this only compiles at all if the two macros' nesting rules stay in sync,
/// so it's a compile-time proof, not just a runtime assertion.
fn test_seq_ty_matches_seq_value_type() -> Result<()> {
    type Net = SeqTy!(
        Linear<s![10, 5], CpuBackendImpl>,
        ReLU,
        Linear<s![5, 2], CpuBackendImpl>
    );

    let net: Net = seq!(
        Linear::<s![10, 5], CpuBackendImpl>::build(())?,
        ReLU,
        Linear::<s![5, 2], CpuBackendImpl>::build(())?
    );

    let input = Tensor::<s![4, 10], CpuBackendImpl>::ones(())?;
    let out = net.forward(input)?;
    assert_eq!(out.dims().dims(), &[4, 2]);

    let params = ParameterGroup::<CpuBackendImpl, f32>::from_module(&net)?;
    assert_eq!(params.len(), 4); // 2 Linear layers x (weight, bias)

    Ok(())
}

#[test]
/// `.eval()` on a `Sequential` propagates through to a nested `Dropout`
/// without the caller reaching into the tree by hand. `Linear` opts into
/// `TrainMode` via its default no-op body (like every other stateless
/// leaf layer), which is what makes `Sequential<Linear<..>, Dropout>: TrainMode`
/// satisfiable at all - see `TrainMode`'s own doc for why this needs
/// explicit bounds rather than relying on an unconstrained generic fallback.
fn test_train_mode_propagates_through_sequential_dropout() -> Result<()> {
    let mut seq = seq!(
        Linear::<s![4, 4], CpuBackendImpl>::build(())?,
        Dropout::new(0.9)
    );

    let input = Tensor::<s![2, 4], CpuBackendImpl>::ones(())?;

    // Fresh Dropout defaults to is_training = true, and Sequential itself
    // has no set_training call yet, so this is still train mode.
    let linear_only = seq.0.forward(input.clone())?.to_vec1::<f32>()?;

    seq.eval();
    assert!(
        !seq.1.is_training,
        "eval() should have flipped is_training to false"
    );
    let out_eval = seq.forward(input.clone())?.to_vec1::<f32>()?;
    // Eval-mode Dropout is an identity function, so this must exactly match
    // Linear's own output with no randomness/scaling applied.
    assert_eq!(out_eval, linear_only);

    seq.train();
    assert!(seq.1.is_training);

    Ok(())
}

#[test]
/// Test embedding.
fn test_embedding() -> Result<()> {
    // Vocab=100, EmbedDim=32
    /// Embed shape.
    type EmbedShape = s![100, 32];
    let weight = Param::<EmbedShape, CpuBackendImpl>::randn(())?;
    let emb = Embedding::<EmbedShape, CpuBackendImpl> { weight };
    // Input: Batch=2, SeqLen=10
    let input = Tensor::<s![2, 10], CpuBackendImpl, i64>::ones(())?;

    let out = emb.forward(input)?;
    assert_eq!(out.dims().dims(), &[2, 10, 32]);

    Ok(())
}

#[test]
/// A loss built without naming its reduction is the `Mean` one, and a
/// non-default reduction stays reachable through `with_reduction`.
fn loss_reduction_defaults_to_mean_and_sum_is_reachable() -> Result<()> {
    let pred = Tensor::<s![2, 2], CpuBackendImpl>::ones(())?;
    let target = Tensor::<s![2, 2], CpuBackendImpl>::zeros(())?;

    // No turbofish here: `new()` resolves to the `Mean` instantiation on its
    // own, which a single generic `new` could not do.
    let mean = MSELoss::new().forward(&pred, &target)?;
    let sum = MSELoss::<incin::nn::Sum>::with_reduction().forward(&pred, &target)?;

    // Every squared error is 1.0, over four elements.
    assert_eq!(mean.to_vec1::<f32>()?[0], 1.0);
    assert_eq!(sum.to_vec1::<f32>()?[0], 4.0);

    Ok(())
}
