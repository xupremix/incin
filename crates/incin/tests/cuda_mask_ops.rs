//! Integration coverage for `a_cuda_comparison_feeds_where_cond` on the documented public surface.
#![cfg(feature = "cuda")]
//! End-to-end check that a `bool` mask CUDA produces is actually reachable
//! from the public `Tensor` API, not just admitted by the capability
//! registry and backed by a real `Execute<>` impl.
//!
//! Capability admission (`incin_backends::capability`'s
//! `mixed_mask_and_data_operations_admit_both_operand_dtypes_on_every_backend`)
//! and the existence of `Execute<op::CmpLt>`/`Execute<op::WhereCond>` for
//! `CudaBackendImpl<D>` are both necessary but neither is sufficient: the
//! `Tensor`-level methods that call them (`.lt`/`.where_cond`, in
//! `incin-core/src/tensor/ops/{binary,manipulation}.rs`) carry their own
//! `where` clauses (`B: Execute<op::X>`, `<B as Execute<op::X>>::Output:
//! Into<B::Storage<K>>`), and nothing before this test checked that chain
//! resolves for the CUDA backend specifically. Hardware-gated like every
//! other CUDA execution test in this workspace: compile-checked everywhere,
//! run with `cargo test --features cuda,std -- --ignored` on real hardware.
//!
//! `Cuda` is a Tier-2 device (runtime ordinal), so tensors are built through
//! the target-first `gpu.tensor(...)`/`TargetExt` surface
//! (`incin_backends::target`) rather than `Tensor::<S, B>::from_slice` - see
//! `crates/incin-backends/tests/target_api_wgpu.rs`'s own doc for why that
//! constructor handles Tier-2 devices worst.

use incin_backends::prelude::*;
use incin_core::prelude::*;

#[test]
#[ignore = "requires CUDA hardware"]
fn a_cuda_comparison_feeds_where_cond() -> Result<()> {
    let gpu = Cuda::new(0);
    let a = gpu.tensor([1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = gpu.tensor([2.0_f32, 2.0, 2.0, 2.0, 2.0, 2.0])?;
    let mask = a.lt(&b)?;

    let on_true = gpu.tensor([10.0_f32; 6])?;
    let on_false = gpu.tensor([-1.0_f32; 6])?;
    let selected = mask.where_cond(&on_true, &on_false)?;

    assert_eq!(
        selected.to_vec1::<f32>()?,
        vec![10.0, 10.0, -1.0, -1.0, -1.0, -1.0]
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware"]
fn a_cuda_comparison_feeds_masked_fill() -> Result<()> {
    let gpu = Cuda::new(0);
    let a = gpu.tensor([1.0_f32, 2.0, 3.0, 4.0])?;
    let mask = a.ge(&gpu.tensor([3.0_f32, 3.0, 3.0, 3.0])?)?;
    let filled = a.masked_fill(&mask, 99.0)?;
    assert_eq!(filled.to_vec1::<f32>()?, vec![1.0, 2.0, 99.0, 99.0]);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_comparisons_feed_logical_connectives() -> Result<()> {
    let gpu = Cuda::new(0);
    let a = gpu.tensor([1.0_f32, 2.0, 3.0, 4.0])?;
    let above_one = a.gt(&gpu.tensor([1.0_f32; 4])?)?;
    let below_four = a.lt(&gpu.tensor([4.0_f32; 4])?)?;

    let both = above_one.logical_and(&below_four)?;
    assert_eq!(both.to_vec1::<bool>()?, vec![false, true, true, false]);

    let either = above_one.logical_or(&below_four)?;
    assert_eq!(either.to_vec1::<bool>()?, vec![true, true, true, true]);

    let neither = above_one.logical_not()?;
    assert_eq!(neither.to_vec1::<bool>()?, vec![true, false, false, false]);
    Ok(())
}

// The lower-rank-mask broadcast `incin_backends::cuda::ops::select`'s
// `launch_broadcast_bool_mask` exists for is exercised directly at the
// storage level by
// `cuda::backend::tests::where_cond_broadcasts_a_lower_rank_mask_before_selecting`
// in `incin-backends`, not here: the static-shape `Tensor::where_cond` above
// requires `S: ShapeEq<S2>` (`tensor/ops/index.rs`), which only holds when
// the mask's shape type and the data's are the *same* type (`impl<S>
// ShapeEq<S> for S` is the only impl) - so a statically lower-rank mask
// cannot even be named at this call site. Reaching that broadcast through
// the public API would need a `Dyn`-shaped mask, where `ShapeEq` is
// trivially satisfied by `Dyn: ShapeEq<Dyn>` regardless of the two tensors'
// actual runtime shapes; out of scope here, since the descriptor/backend
// composition it would exercise is already covered.
