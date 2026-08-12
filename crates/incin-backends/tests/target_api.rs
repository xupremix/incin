//! The `target-api` prototype: device values as allocation targets.
//!
//! These are the tests the architecture review named as the prototype's
//! acceptance criteria. Each one asserts a property the current constructor
//! surface cannot offer, so a passing run is evidence the design works rather
//! than evidence it compiles.

#![cfg(all(feature = "target-api", feature = "cpu"))]

// `s!` expands to `::incin::prelude::…`, which does not resolve inside this
// crate. The repo's own doctests use the same alias.
extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_backends::prelude::*;
use incin_backends::target::{DtypeView, TargetBackend, TargetBackendFor};
use incin_core::prelude::*;

// ============================================================================
// Target behaviour
// ============================================================================

/// A device value is a target with no construction step: `Cpu` is a unit
/// struct, not something that has to be initialized or unwrapped.
#[test]
fn a_device_value_is_the_target() {
    let cpu = Cpu;
    assert_eq!(cpu.device_id().unwrap(), DeviceId::cpu());
}

/// Targets are cheap values, so cloning one and dropping the original cannot
/// affect anything: they own no resources. Backend state lives in process
/// globals, which is exactly why a `Runtime` object would have nothing to own.
#[test]
fn dropping_a_target_does_not_invalidate_its_tensors() {
    let tensor = {
        let cpu = Cpu;
        cpu.zeros(shape![2, 3]).unwrap()
        // `cpu` drops here.
    };
    assert_eq!(tensor.dims(), [2, 3]);
    assert_eq!(tensor.to_vec1::<f32>().unwrap(), vec![0.0; 6]);
}

// ============================================================================
// Typed data — dtype comes from the data, never from the target
// ============================================================================

/// A nested Rust array carries both its shape and its element type, which is
/// the whole argument against needing a literal macro for this.
#[test]
fn a_nested_array_keeps_its_static_shape_and_dtype() {
    let x = Cpu.tensor([[1.0_f32, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(x.dims(), [2, 2]);
    assert_eq!(x.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// Row-major: the outer array is the slowest-varying axis.
#[test]
fn nested_arrays_flatten_row_major() {
    let x = Cpu.tensor([[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    assert_eq!(x.dims(), [2, 3]);
    assert_eq!(
        x.to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

/// The target's float dtype is `f32`, and integer data must survive it
/// unchanged. Silently casting a label vector to float is the exact bug the
/// MNIST example works around today with `label as f32`.
#[test]
fn integer_data_is_not_cast_to_the_targets_float() {
    let labels = Cpu.tensor([0_i64, 1, 2]).unwrap();
    assert_eq!(labels.dims(), [3]);
    assert_eq!(labels.to_vec1::<i64>().unwrap(), vec![0, 1, 2]);
}

/// `f64` data on an `f32` target likewise keeps its own width.
#[test]
fn f64_data_is_not_narrowed_to_the_targets_float() {
    let x = Cpu.tensor([1.5_f64, 2.5]).unwrap();
    assert_eq!(x.to_vec1::<f64>().unwrap(), vec![1.5, 2.5]);
}

// ============================================================================
// Gradient rule — follows the object being created, not a target setting
// ============================================================================

/// Data tensors are `NoGrad`. This is a type-level fact, so the assertion is
/// the signature itself; the runtime check just documents it.
#[test]
fn data_and_generated_tensors_do_not_track_gradients() {
    let data = Cpu.tensor([1.0_f32, 2.0]).unwrap();
    let generated = Cpu.zeros(shape![4]).unwrap();
    assert!(!data.requires_grad());
    assert!(!generated.requires_grad());
}

/// Parameters do, and they come from the same construction path rather than a
/// second allocator.
#[test]
fn parameters_track_gradients() {
    let weight = Cpu.parameter(shape![4, 2], GeneratedFill::Normal).unwrap();
    assert!(weight.requires_grad());
    assert_eq!(weight.dims(), [4, 2]);
}

// ============================================================================
// Shape specification — the argument decides the result type
// ============================================================================

/// Fully static: shape! produces a fully static tensor.
#[test]
fn a_static_spec_produces_a_static_tensor() {
    let x = Cpu.zeros(shape![32, 784]).unwrap();
    assert_eq!(x.dims(), [32, 784]);
    let _typed: Tensor<s![32, 784], _, f32, NoGrad> = x;
}

/// Fully dynamic: an ordinary array of runtime extents yields `Tensor<Dyn, ..>`.
#[test]
fn an_array_spec_produces_a_dynamic_tensor() {
    let batch = 7usize;
    let x = Cpu.zeros(vec![batch, 128]).unwrap();
    assert_eq!(&x.dims()[..], &[7, 128][..]);
    let _typed: Tensor<Dyn, _, f32, NoGrad> = x;
}

/// Partially static: dynamic axis + static axis preserves shape type.
#[test]
fn a_bound_spec_preserves_partial_shape_information() {
    let batch = 5usize;
    let x = Cpu.zeros(shape![batch, 784]).unwrap();
    assert_eq!(x.dims(), [5, 784]);
    let _typed: Tensor<s![usize, 784], _, f32, NoGrad> = x;
}

/// A zero-length runtime dimension is legal, matching `vec![]` and
/// `torch.tensor([])`.
#[test]
fn a_zero_length_runtime_dimension_is_allowed() {
    let x = Cpu.zeros(shape![0usize, 784]).unwrap();
    assert_eq!(x.dims(), [0, 784]);
}

// ============================================================================
// Runtime-sized data
// ============================================================================

#[test]
fn a_runtime_vector_fills_a_dynamic_shape() {
    let values: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let x = Cpu.tensor_from_vec(values, [3, 4]).unwrap();
    assert_eq!(&x.dims()[..], &[3, 4][..]);
    assert_eq!(x.to_vec1::<f32>().unwrap()[11], 11.0);
}

/// The same data with a typed partial shape keeps the static axis.
#[test]
fn a_runtime_vector_can_fill_a_partial_shape() {
    let batch = 2usize;
    let values: Vec<f32> = vec![0.0; batch * 784];
    let x = Cpu.tensor_from_vec(values, shape![batch, 784]).unwrap();
    assert_eq!(x.dims(), [2, 784]);
    let _typed: Tensor<s![usize, 784], _, f32, NoGrad> = x;
}

/// A length that disagrees with the requested shape is caught before any
/// allocation happens.
#[test]
fn a_length_mismatch_is_reported_rather_than_truncated() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0];
    let error = Cpu.tensor_from_vec(values, [2, 4]);
    assert!(error.is_err(), "3 values cannot fill a 2x4 tensor");
}

// ============================================================================
// Dtype rebinding
// ============================================================================

/// `dtype` changes what *generated* tensors are made of, and nothing else.
#[test]
fn with_dtype_rebinds_the_generated_dtype() {
    let fp64 = Cpu.dtype::<f64>().unwrap();
    let x = fp64.zeros(shape![2, 2]).unwrap();
    assert_eq!(x.to_vec1::<f64>().unwrap(), vec![0.0_f64; 4]);
    assert_eq!(fp64.device_id().unwrap(), DeviceId::cpu());
}

/// `dtype` answers for storage, not for every operation. The CPU holds
/// `Q8_0` blocks, so rebinding to it succeeds; it has no fill kernel for them,
/// so `zeros` on that view still refuses.
#[test]
fn with_dtype_admits_a_dtype_the_device_can_store_but_not_fill() {
    let quantized = Cpu
        .dtype::<Q8_0>()
        .expect("CPU storage holds every dtype, including Q8_0");
    assert!(quantized.zeros(shape![2, 2]).is_err());
}

/// The CPU storage row lists every dtype, so every rebinding of it is admitted.
#[test]
fn with_dtype_admits_every_dtype_the_cpu_can_store() {
    assert!(Cpu.dtype::<f16>().is_ok());
    assert!(Cpu.dtype::<bf16>().is_ok());
    assert!(Cpu.dtype::<f64>().is_ok());
    assert!(Cpu.dtype::<u8>().is_ok());
    assert!(Cpu.dtype::<u32>().is_ok());
    assert!(Cpu.dtype::<i64>().is_ok());
}

// ============================================================================
// Generated fills
// ============================================================================

#[test]
fn generated_fills_produce_their_documented_values() {
    let zeros = Cpu.zeros(shape![3]).unwrap();
    let ones = Cpu.ones(shape![3]).unwrap();
    assert_eq!(zeros.to_vec1::<f32>().unwrap(), vec![0.0, 0.0, 0.0]);
    assert_eq!(ones.to_vec1::<f32>().unwrap(), vec![1.0, 1.0, 1.0]);

    let sampled = Cpu.rand(shape![64]).unwrap();
    for value in sampled.to_vec1::<f32>().unwrap() {
        assert!((0.0..1.0).contains(&value), "uniform sample out of range");
    }

    let normal = Cpu.randn(shape![8]).unwrap();
    assert_eq!(normal.dims(), [8]);
}

// ============================================================================
// Target backend family ownership (B8 Sentinel test)
// ============================================================================

/// Test-only sentinel target proving that `DtypeView<T, K>`
/// delegates backend selection to `T` rather than re-inferring it from device.
#[derive(Clone, Copy)]
struct SentinelTarget;

impl TensorTarget for SentinelTarget {
    type Dtype = f32;
    type ParameterDtype = f32;
    type Device = Cpu;
    type Backend = CpuBackendImpl<Cpu>;
    fn device_arg(&self) {}
    fn dtype_field(&self) -> <Self::Dtype as DType>::Field {
        core::marker::PhantomData
    }
    fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field {
        core::marker::PhantomData
    }
    fn precision_policy(&self) -> incin_backends::target::RuntimePrecisionPolicy {
        incin_backends::target::RuntimePrecisionPolicy::fp32()
    }
}

#[test]
fn dtype_view_preserves_target_backend_family() {
    let target = SentinelTarget;
    let _rebound = target.dtype::<i64>().unwrap();

    trait Same<T> {}
    impl<T> Same<T> for T {}
    fn assert_same_backend<B1, B2>()
    where
        B1: Same<B2>,
        B2: Same<B1>,
    {
    }

    assert_same_backend::<TargetBackendFor<DtypeView<SentinelTarget, i64>>, CpuBackendImpl<Cpu>>();
}

// ============================================================================
// Canonical lowering
// ============================================================================

/// Every fill in the creation family routes through canonical lowering.
#[test]
fn the_whole_fill_family_routes_through_canonical_dispatch() {
    let zeros = Cpu.zeros(shape![2, 3]).unwrap();
    assert_eq!(zeros.to_vec1::<f32>().unwrap(), vec![0.0; 6]);

    let ones = Cpu.ones(shape![2, 3]).unwrap();
    assert_eq!(ones.to_vec1::<f32>().unwrap(), vec![1.0; 6]);

    let uniform = Cpu.rand(shape![2, 3]).unwrap();
    let drawn = uniform.to_vec1::<f32>().unwrap();
    assert_eq!(drawn.len(), 6);
    assert!(drawn.iter().all(|value| (0.0..1.0).contains(value)));

    let normal = Cpu.randn(shape![2, 3]).unwrap();
    let drawn = normal.to_vec1::<f32>().unwrap();
    assert_eq!(drawn.len(), 6);
    assert!(drawn.iter().all(|value| value.is_finite()));
}

/// A request the capability registry refuses is refused before allocation.
/// `Q8_0` is a packed block format with no `zeros` kernel.
#[test]
fn canonical_zeros_refuses_an_unsupported_dtype() {
    let quantized = Cpu.dtype::<incin_core::prelude::Q8_0>().unwrap();
    let refused = quantized.zeros(shape![2, 3]);
    assert!(
        refused.is_err(),
        "the capability registry advertises no zeros kernel for Q8_0"
    );
}

// ============================================================================
// Layer initialization
// ============================================================================

#[test]
fn a_linear_layer_is_built_from_a_target() {
    let cpu = Cpu;
    let layer = incin_core::nn::Linear::<Dyn, _>::new(4, 3, &cpu).unwrap();
    assert_eq!(layer.weight.shape_dims(), vec![3, 4]);
    assert!(layer.bias.is_some(), "bias is present by default");
}

#[test]
fn layer_parameters_are_shaped_from_the_feature_counts() {
    let layer = incin_core::nn::Linear::<Dyn, _>::new(4, 3, &Cpu).unwrap();
    assert_eq!(layer.weight.shape_dims(), vec![3, 4]);
    assert_eq!(
        layer.bias.as_ref().map(incin_core::nn::Param::shape_dims),
        Some(vec![3])
    );
}

// ============================================================================
// Dtype selection
// ============================================================================

#[test]
fn a_target_can_be_rebound_to_an_integer_dtype_for_fills() {
    let idx = Cpu.dtype::<i64>().unwrap();
    let t = idx.zeros(shape![4]).unwrap();
    assert_eq!(t.to_vec1::<i64>().unwrap(), vec![0_i64; 4]);

    let ones = idx.ones(shape![3]).unwrap();
    assert_eq!(ones.to_vec1::<i64>().unwrap(), vec![1_i64; 3]);
}

#[allow(deprecated)]
#[test]
fn legacy_with_dtype_alias_works() {
    let idx = Cpu.with_dtype::<i64>().unwrap();
    let t = idx.zeros(shape![4]).unwrap();
    assert_eq!(t.to_vec1::<i64>().unwrap(), vec![0_i64; 4]);
}

#[test]
fn a_rebound_target_still_does_not_touch_data_dtype() {
    let idx = Cpu.dtype::<i64>().unwrap();
    let floats = idx.tensor([1.5_f32, 2.5]).unwrap();
    assert_eq!(floats.to_vec1::<f32>().unwrap(), vec![1.5, 2.5]);
}

#[test]
fn the_value_carrying_constructors_are_reachable_from_a_target() {
    let filled = Cpu.full(shape![2, 2], 7.0).unwrap();
    assert_eq!(filled.to_vec1::<f32>().unwrap(), vec![7.0; 4]);

    let stepped = Cpu.arange(shape![4], 1.0, 2.0).unwrap();
    assert_eq!(stepped.to_vec1::<f32>().unwrap(), vec![1.0, 3.0, 5.0, 7.0]);

    let spaced = Cpu.linspace(shape![3], 0.0, 1.0).unwrap();
    assert_eq!(spaced.to_vec1::<f32>().unwrap(), vec![0.0, 0.5, 1.0]);
}

#[test]
fn the_value_carrying_constructors_accept_every_shape_specification() {
    let batch = 3usize;
    assert_eq!(Cpu.full(shape![2, 3], 1.0).unwrap().dims(), [2, 3]);
    assert_eq!(&Cpu.full([batch, 4], 1.0).unwrap().dims()[..], &[3, 4][..]);
}

// ============================================================================
// Engine & Target<E, D, P> tests (Phase D)
// ============================================================================

#[test]
fn native_on_cpu_and_bare_cpu_resolve_same_backend_family() {
    let native_target = Native::on(Cpu);
    let bare_target = Cpu;

    trait Same<T> {}
    impl<T> Same<T> for T {}
    fn assert_same_backend<B1, B2>()
    where
        B1: Same<B2>,
        B2: Same<B1>,
    {
    }

    assert_same_backend::<TargetBackend<Target<Native, Cpu>>, TargetBackend<Cpu>>();

    let a = native_target.zeros(shape![2, 3]).unwrap();
    let b = bare_target.zeros(shape![2, 3]).unwrap();
    assert_eq!(a.dims(), [2, 3]);
    assert_eq!(b.dims(), [2, 3]);
}

#[cfg(feature = "external-candle")]
#[test]
fn candle_and_native_engines_resolve_different_backend_families() {
    let candle_target = Candle::on(Cpu);
    let native_target = Native::on(Cpu);

    assert_ne!(
        core::any::TypeId::of::<TargetBackend<Target<Candle, Cpu>>>(),
        core::any::TypeId::of::<TargetBackend<Target<Native, Cpu>>>()
    );

    let c = candle_target.zeros(shape![2, 3]).unwrap();
    let n = native_target.zeros(shape![2, 3]).unwrap();
    assert_eq!(c.dims(), [2, 3]);
    assert_eq!(n.dims(), [2, 3]);
}

#[cfg(feature = "external-candle")]
#[test]
fn candle_dtype_view_preserves_candle_backend_family() {
    let candle_target = Candle::on(Cpu);
    let _rebound = candle_target.dtype::<i64>().unwrap();

    trait Same<T> {}
    impl<T> Same<T> for T {}
    fn assert_same_backend<B1, B2>()
    where
        B1: Same<B2>,
        B2: Same<B1>,
    {
    }

    assert_same_backend::<
        TargetBackendFor<DtypeView<Target<Candle, Cpu>, i64>>,
        incin_backends::external::candle::CandleBackend<Cpu>,
    >();
}

#[cfg(feature = "external-candle")]
#[test]
fn candle_data_construction_preserves_candle_backend_family() {
    let candle_target = Candle::on(Cpu);
    let t = candle_target.tensor([1_i64, 2, 3]).unwrap();
    assert_eq!(t.dims(), [3]);
    assert_eq!(t.to_vec1::<i64>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn target_with_precision_mutates_precision_slot() {
    let target = Native::on(Cpu);
    let exact_target = target.with_precision(precision::Exact::<f64>::new());
    let x = exact_target.zeros(shape![2, 2]).unwrap();
    assert_eq!(x.to_vec1::<f64>().unwrap(), vec![0.0_f64; 4]);
}
