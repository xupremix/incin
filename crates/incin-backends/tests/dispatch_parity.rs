//! `EXE-009`: the runtime-selected backend must not refuse what it routes to.
//!
//! `DispatchBackend` is what `IncinBackend<_, Dyn>` resolves to, so it is the
//! backend a user gets without naming one. Its operation traits used to be
//! empty impls, which compiles: every method fell through to a default body
//! returning `UnsupportedBackendOperation`. The result was a backend that
//! advertised normalization, quantization, and cross-entropy through the
//! capability registry and then refused them at run time, with nothing at
//! compile time to catch the gap.
//!
//! These cases pin the routing down against the concrete backend underneath,
//! so a family that stops being routed fails here rather than in a user's
//! training loop.
#![cfg(feature = "cpu")]

extern crate incin_core as incin;

use incin_backends::cpu::{CpuBackendImpl, CpuStorage};
use incin_backends::dispatch::{DispatchBackend, DispatchStorage};
use incin_core::backend_authoring::{
    Backend, Execute, ExecutionRequest, LossOps, ModuleOps, QuantizedOps, ReductionOps,
    StorageBackend,
};
use incin_core::exec::{
    Conv2dArgs, Conv2dRule, ExecutionContext, Pool2dRule, Pool2dSpec, PoolOp, ReduceAtRule,
    ReduceOp, ReductionSpec, ReshapeRule, ReshapeSpec, ShapeRule, TensorHandle, Validated,
};
use incin_core::prelude::*;
use incin_core::shapes::idx::{Here, Next};
use incin_core::shapes::shape::{DimCons, Nil};
use incin_core::typenum::{U0, U1, U2, U3, U4, U6};

type Dispatch = DispatchBackend<Dyn>;
type DirectCpu = CpuBackendImpl<Cpu>;
type R2 = DimCons<U2, DimCons<U3, Nil>>;
type Shape26 = DimCons<U2, DimCons<U6, Nil>>;
type Shape34 = DimCons<U3, DimCons<U4, Nil>>;
type Shape1144 = DimCons<U1, DimCons<U1, DimCons<U4, DimCons<U4, Nil>>>>;

fn dispatch_from(values: &[f32], shape: &[usize]) -> DispatchStorage {
    <Dispatch as Backend>::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .expect("cpu-device creation must route to the cpu backend")
}

fn dispatch_indices(values: &[i64], shape: &[usize]) -> DispatchStorage {
    <Dispatch as Backend>::from_bytes::<i64>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::I64.descriptor(),
        &DeviceId::cpu(),
    )
    .expect("index creation must route to the cpu backend")
}

fn cpu_from(values: &[f32], shape: &[usize]) -> CpuStorage {
    <DirectCpu as Backend>::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .expect("cpu creation must succeed")
}

fn cpu_indices(values: &[i64], shape: &[usize]) -> CpuStorage {
    <DirectCpu as Backend>::from_bytes::<i64>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::I64.descriptor(),
        &DeviceId::cpu(),
    )
    .expect("cpu index creation must succeed")
}

fn dispatch_values(storage: &DispatchStorage) -> Vec<f32> {
    let bytes = <Dispatch as Backend>::to_bytes::<f32>(storage).expect("readback must succeed");
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn cpu_values(storage: &CpuStorage) -> Vec<f32> {
    let bytes = <DirectCpu as Backend>::to_bytes::<f32>(storage).expect("readback must succeed");
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn assert_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len(), "routed output changed length");
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "element {index} diverged: routed {a} vs direct {b}"
        );
    }
}

#[test]
fn layer_norm_routes_instead_of_reporting_unsupported() {
    let input = [0.5_f32, -1.5, 2.0, 0.25, 1.0, -0.75, 0.0, 3.0];
    let weight = [1.25_f32, 0.5, -0.75, 2.0];

    let routed = <Dispatch as ModuleOps<Dispatch>>::layer_norm::<f32>(
        &dispatch_from(&input, &[2, 4]),
        &dispatch_from(&weight, &[4]),
        None,
        1e-5,
    )
    .expect("layer_norm must route to the cpu backend");

    let direct = <DirectCpu as ModuleOps<DirectCpu>>::layer_norm::<f32>(
        &cpu_from(&input, &[2, 4]),
        &cpu_from(&weight, &[4]),
        None,
        1e-5,
    )
    .expect("cpu layer_norm must succeed");

    assert_close(&dispatch_values(&routed), &cpu_values(&direct));
}

#[test]
fn embedding_routes_with_its_second_operand() {
    let table = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = dispatch_indices(&[2, 0], &[2]);

    let routed = <Dispatch as ModuleOps<Dispatch>>::embedding::<f32, i64>(
        &indices,
        &dispatch_from(&table, &[3, 2]),
    )
    .expect("embedding must route");

    assert_eq!(dispatch_values(&routed), vec![5.0, 6.0, 1.0, 2.0]);
}

#[test]
fn cross_entropy_is_the_one_loss_without_a_composed_default_and_still_routes() {
    let logits = [2.0_f32, 1.0, 0.1, 0.5, 2.5, 0.3];
    let targets = dispatch_indices(&[0, 1], &[2]);

    let routed = <Dispatch as LossOps<Dispatch>>::cross_entropy_loss::<f32, i64>(
        &dispatch_from(&logits, &[2, 3]),
        &targets,
        Reduction::Mean,
    )
    .expect("cross_entropy_loss must route");

    let cpu_targets = cpu_indices(&[0, 1], &[2]);
    let direct = <DirectCpu as LossOps<DirectCpu>>::cross_entropy_loss::<f32, i64>(
        &cpu_from(&logits, &[2, 3]),
        &cpu_targets,
        Reduction::Mean,
    )
    .expect("cpu cross_entropy_loss must succeed");

    assert_close(&dispatch_values(&routed), &cpu_values(&direct));
}

#[test]
fn quantization_round_trips_through_the_routed_backend() {
    let values: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25 - 8.0).collect();

    let quantized = <Dispatch as QuantizedOps<Dispatch>>::quantize::<f32, Q8_0>(&dispatch_from(
        &values,
        &[2, 32],
    ))
    .expect("quantize must route");
    let restored = <Dispatch as QuantizedOps<Dispatch>>::dequantize::<Q8_0, f32>(&quantized)
        .expect("dequantize must route");

    let restored = dispatch_values(&restored);
    assert_eq!(restored.len(), values.len());
    for (index, (original, roundtrip)) in values.iter().zip(&restored).enumerate() {
        assert!(
            (original - roundtrip).abs() < 0.1,
            "element {index} lost more than quantization error: {original} vs {roundtrip}"
        );
    }
}

#[test]
fn the_index_returning_reductions_route_too() {
    // `argmax`, `argmin`, `argsort`, and `topk` are generic over a second
    // index dtype, which is why they were missed when the rest of the
    // reduction family was routed by a single-generic macro.
    let values = [3.0_f32, 1.0, 4.0, 1.5];
    let routed = dispatch_from(&values, &[4]);
    let direct = cpu_from(&values, &[4]);

    let max_index = <Dispatch as ReductionOps<Dispatch>>::argmax::<f32, i64>(&routed, None)
        .expect("argmax must route");
    let direct_max = <DirectCpu as ReductionOps<DirectCpu>>::argmax::<f32, i64>(&direct, None)
        .expect("cpu argmax must succeed");
    assert_eq!(
        <Dispatch as Backend>::to_bytes::<i64>(&max_index).expect("readback"),
        <DirectCpu as Backend>::to_bytes::<i64>(&direct_max).expect("readback"),
    );

    <Dispatch as ReductionOps<Dispatch>>::argmin::<f32, i64>(&routed, None)
        .expect("argmin must route");
    <Dispatch as ReductionOps<Dispatch>>::argsort::<f32, i64>(&routed, 0, false)
        .expect("argsort must route");

    let (top_values, _) =
        <Dispatch as ReductionOps<Dispatch>>::topk::<f32, i64>(&routed, 2, 0, true)
            .expect("topk must route");
    assert_eq!(dispatch_values(&top_values), vec![4.0, 3.0]);
}

#[test]
fn no_routed_operation_family_answers_with_the_unsupported_default() {
    // The failure this guards against is a whole family silently reverting to
    // the trait's default body. Any `UnsupportedBackendOperation` naming the
    // dispatch backend means routing was dropped.
    let input = dispatch_from(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let weight = dispatch_from(&[1.0, 1.0], &[2]);

    let results: Vec<(&str, Result<DispatchStorage>)> = vec![
        (
            "layer_norm",
            <Dispatch as ModuleOps<Dispatch>>::layer_norm::<f32>(&input, &weight, None, 1e-5),
        ),
        (
            "adaptive_avg_pool2d",
            <Dispatch as ModuleOps<Dispatch>>::adaptive_avg_pool2d::<f32>(
                &dispatch_from(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
                (1, 1),
            ),
        ),
    ];

    for (name, result) in results {
        if let Err(Error::UnsupportedBackendOperation { op, backend }) = &result {
            panic!("{name} fell through to the unsupported default: op={op} backend={backend}");
        }
        result.unwrap_or_else(|error| panic!("{name} must route: {error}"));
    }
}

#[test]
fn a_validated_reshape_descriptor_routes_to_the_backend_holding_the_operand() {
    // The descriptor path has the same failure mode as the operation families:
    // a dispatch backend that answers for itself instead of routing gives a
    // runtime-selected device weaker validation than a statically-selected one.
    let values: Vec<f32> = (1..=12).map(|value| value as f32).collect();
    let validated: Validated<ReshapeSpec> = <ReshapeRule as ShapeRule<(Shape26, Shape34)>>::lower(
        &(
            <Shape26 as Shape>::try_from_dims(&[2, 6]).unwrap(),
            <Shape34 as Shape>::try_from_dims(&[3, 4]).unwrap(),
        ),
        (),
    )
    .expect("12 elements either way");

    let routed_input = dispatch_from(&values, &[2, 6]);
    let context = ExecutionContext::new(Dispatch::new());
    let inputs = [TensorHandle::from_storage::<Dispatch, f32, Local>(
        &routed_input,
    )];
    let routed = context
        .backend()
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &inputs,
            context: &context,
        })
        .expect("a reshape descriptor must route to the cpu backend");

    let direct_input = cpu_from(&values, &[2, 6]);
    let direct_context = ExecutionContext::new(DirectCpu::new());
    let direct_inputs = [TensorHandle::from_storage::<DirectCpu, f32, Local>(
        &direct_input,
    )];
    let direct = direct_context
        .backend()
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &direct_inputs,
            context: &direct_context,
        })
        .expect("the cpu backend must execute the same descriptor");

    assert_eq!(
        <Dispatch as StorageBackend>::metadata::<f32>(&routed)
            .shape()
            .dims(),
        &[3, 4]
    );
    assert_close(&dispatch_values(&routed), &cpu_values(&direct));
}

#[test]
fn a_validated_conv2d_descriptor_routes_with_all_three_of_its_operands() {
    // Convolution is the first descriptor with an optional operand, so routing
    // has to carry the bias along rather than drop it. A dropped bias would
    // still produce a correctly shaped output, which is exactly why it is
    // compared against the directly-executed result instead of just its shape.
    type Conv3x3 = Conv2dRule<U3, U3, U1, U1, U1>;
    type ConvInput = s![1, 2, 4, 4];

    let validated = <Conv3x3 as ShapeRule<ConvInput>>::lower(
        &<ConvInput as Shape>::try_from_dims(&[1, 2, 4, 4]).unwrap(),
        Conv2dArgs::dense(3),
    )
    .expect("a 3x3 window with padding 1 fits a 4x4 input");

    let input: Vec<f32> = (0..32).map(|value| value as f32 * 0.25 - 4.0).collect();
    let weight: Vec<f32> = (0..54).map(|value| value as f32 * 0.1 - 2.5).collect();
    let bias = [0.5_f32, -0.25, 1.0];

    let context = ExecutionContext::new(Dispatch::new());
    let (routed_input, routed_weight, routed_bias) = (
        dispatch_from(&input, &[1, 2, 4, 4]),
        dispatch_from(&weight, &[3, 2, 3, 3]),
        dispatch_from(&bias, &[3]),
    );
    let inputs = [
        TensorHandle::from_storage::<Dispatch, f32, Local>(&routed_input),
        TensorHandle::from_storage::<Dispatch, f32, Local>(&routed_weight),
        TensorHandle::from_storage::<Dispatch, f32, Local>(&routed_bias),
    ];
    let routed = context
        .backend()
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &inputs,
            context: &context,
        })
        .expect("a conv2d descriptor must route to the cpu backend");

    let direct_context = ExecutionContext::new(DirectCpu::new());
    let (direct_input, direct_weight, direct_bias) = (
        cpu_from(&input, &[1, 2, 4, 4]),
        cpu_from(&weight, &[3, 2, 3, 3]),
        cpu_from(&bias, &[3]),
    );
    let direct_inputs = [
        TensorHandle::from_storage::<DirectCpu, f32, Local>(&direct_input),
        TensorHandle::from_storage::<DirectCpu, f32, Local>(&direct_weight),
        TensorHandle::from_storage::<DirectCpu, f32, Local>(&direct_bias),
    ];
    let direct = direct_context
        .backend()
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &direct_inputs,
            context: &direct_context,
        })
        .expect("the cpu backend must execute the same descriptor");

    assert_eq!(
        <Dispatch as StorageBackend>::metadata::<f32>(&routed)
            .shape()
            .dims(),
        &[1, 3, 4, 4]
    );
    assert_close(&dispatch_values(&routed), &cpu_values(&direct));
}

#[test]
fn a_reduction_routes_the_accumulation_its_descriptor_names() {
    // The geometry is the same for every accumulation, so a dispatch layer that
    // routed on geometry alone would look correct here and be silently wrong.
    // Comparing against the direct backend per operator is what catches that.
    let values: Vec<f32> = vec![1., 2., 3., 4., 5., 6.];

    for op in [
        ReduceOp::Sum,
        ReduceOp::Mean,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::Prod,
    ] {
        let validated: Validated<ReductionSpec> =
            <ReduceAtRule<Next<Here>> as ShapeRule<R2>>::lower(
                &<R2 as Shape>::try_from_dims(&[2, 3]).unwrap(),
                op,
            )
            .expect("axis 1 is in range");

        let routed_input = dispatch_from(&values, &[2, 3]);
        let context = ExecutionContext::new(Dispatch::new());
        let inputs = [TensorHandle::from_storage::<Dispatch, f32, Local>(
            &routed_input,
        )];
        let routed = context
            .backend()
            .execute(ExecutionRequest {
                operation: &validated,
                inputs: &inputs,
                context: &context,
            })
            .unwrap_or_else(|error| panic!("{op} must route to the cpu backend: {error:?}"));

        let direct_input = cpu_from(&values, &[2, 3]);
        let direct_context = ExecutionContext::new(DirectCpu::new());
        let direct_inputs = [TensorHandle::from_storage::<DirectCpu, f32, Local>(
            &direct_input,
        )];
        let direct = direct_context
            .backend()
            .execute(ExecutionRequest {
                operation: &validated,
                inputs: &direct_inputs,
                context: &direct_context,
            })
            .unwrap_or_else(|error| panic!("{op} must execute directly: {error:?}"));

        assert_close(&dispatch_values(&routed), &cpu_values(&direct));
    }
}

#[test]
fn a_pool_routes_the_accumulation_its_descriptor_names() {
    let values: Vec<f32> = (1..=16).map(|value| value as f32).collect();

    for op in [PoolOp::Max, PoolOp::Average] {
        let validated: Validated<Pool2dSpec> =
            <Pool2dRule<U2, U2, U0, U1> as ShapeRule<Shape1144>>::lower(
                &<Shape1144 as Shape>::try_from_dims(&[1, 1, 4, 4]).unwrap(),
                op,
            )
            .expect("a 2x2 window strided by 2 tiles a 4x4 input");

        let routed_input = dispatch_from(&values, &[1, 1, 4, 4]);
        let context = ExecutionContext::new(Dispatch::new());
        let inputs = [TensorHandle::from_storage::<Dispatch, f32, Local>(
            &routed_input,
        )];
        let routed = context
            .backend()
            .execute(ExecutionRequest {
                operation: &validated,
                inputs: &inputs,
                context: &context,
            })
            .unwrap_or_else(|error| panic!("{op} pooling must route: {error:?}"));

        let direct_input = cpu_from(&values, &[1, 1, 4, 4]);
        let direct_context = ExecutionContext::new(DirectCpu::new());
        let direct_inputs = [TensorHandle::from_storage::<DirectCpu, f32, Local>(
            &direct_input,
        )];
        let direct = direct_context
            .backend()
            .execute(ExecutionRequest {
                operation: &validated,
                inputs: &direct_inputs,
                context: &direct_context,
            })
            .unwrap_or_else(|error| panic!("{op} pooling must execute directly: {error:?}"));

        assert_eq!(
            <Dispatch as StorageBackend>::metadata::<f32>(&routed)
                .shape()
                .dims(),
            &[1, 1, 2, 2]
        );
        assert_close(&dispatch_values(&routed), &cpu_values(&direct));
    }
}

/// A device this dispatcher carries no variant for is refused *by name*.
///
/// Metal is the case that exists today: it is a first-class feature, and
/// `DispatchStorage` has no arm for it, so every route lands on the fallback.
/// That fallback used to answer `BackendUnavailable { backend: "Unknown" }`,
/// which told a user on Apple Silicon nothing about which backend was missing
/// and read identically to a genuinely unrecognized device. The refusal itself
/// is correct and is not what this pins; the attribution is.
#[test]
fn an_unrouted_device_is_refused_by_name_not_as_unknown() {
    // `let ... else` rather than `expect_err`: the success type is
    // `DispatchStorage`, which is not `Debug`, so `expect_err` will not compile.
    let Err(error) = <Dispatch as Backend>::from_bytes::<f32>(
        bytemuck::cast_slice(&[1.0_f32, 2.0]),
        &[2],
        DTypeId::F32.descriptor(),
        &DeviceId::metal(0),
    ) else {
        panic!("this dispatcher has no Metal variant to route to");
    };

    assert_eq!(
        error.to_string(),
        "Backend 'Metal' is unavailable in this build"
    );
}
