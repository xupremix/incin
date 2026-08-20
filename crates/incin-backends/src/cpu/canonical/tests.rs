use super::*;
use crate::cpu::gradcheck::gradcheck;
use crate::cpu::storage::{CpuBuffer, CpuStorage};
use incin_core::dist::Local;
use incin_core::exec::GradMode;
use incin_core::exec::catalog::{
    ArangeAttributes, AxisAttributes, AxisVarianceAttributes, BatchNormAttributes, ChunkAttributes,
    Conv1dAttributes, Conv2dAttributes, CreationAttributes, DTypeAttributes, DropoutAttributes,
    EpsilonAttributes, LayerNormAttributes, LinearAttributes, LinspaceAttributes, LossAttributes,
    LossReduction, NoAttributes, NormAttributes, QuantizationAttributes, ShapeAttributes,
    SplitAttributes, VarianceAttributes, op,
};
use incin_core::exec::{ExecutionContext, TensorHandle, dispatch};
use incin_core::tensor::device::{Cpu, DeviceId};
use incin_core::tensor::dtype::DTypeId;

type TestBackend = CpuBackendImpl<Cpu>;

fn storage(values: &[f32], shape: &[usize]) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(values.to_vec()), shape)
        .expect("test storage must be well formed")
}

fn handle(storage: &CpuStorage) -> TensorHandle<'_> {
    TensorHandle::from_storage::<TestBackend, f32, Local>(storage)
}

fn context() -> ExecutionContext<TestBackend> {
    ExecutionContext::new(TestBackend::new())
}

fn inference_context() -> ExecutionContext<TestBackend> {
    ExecutionContext::new(TestBackend::new()).with_grad_mode(GradMode::Disabled)
}

const GRADIENT_STEP: f64 = 1e-2;
const GRADIENT_TOLERANCE: f64 = 1e-3;

#[test]
fn canonical_pointwise_gradients_match_finite_differences() {
    let context = context();
    let lhs = storage(&[0.5, 1.5, -2.0, 3.0], &[4]);
    let rhs = storage(&[2.0, -1.0, 0.5, 1.25], &[4]);

    let error = gradcheck(
        |inputs| {
            let product = dispatch::execute::<op::Mul, _>(
                &context,
                NoAttributes,
                &[handle(&inputs[0]), handle(&inputs[1])],
            )
            .expect("mul executes");
            dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &[handle(&product)])
                .expect("sum_all executes")
        },
        &[lhs, rhs],
        GRADIENT_STEP,
    );
    assert!(
        error < GRADIENT_TOLERANCE,
        "canonical mul gradient error {error} exceeds {GRADIENT_TOLERANCE}"
    );
}

#[test]
fn canonical_view_gradients_match_finite_differences() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let error = gradcheck(
        |inputs| {
            let reshaped = dispatch::execute::<op::ReshapeExact, _>(
                &context,
                ShapeAttributes { shape: vec![3, 2] },
                &[handle(&inputs[0])],
            )
            .expect("reshape executes");
            let scaled = dispatch::execute::<op::Mul, _>(
                &context,
                NoAttributes,
                &[handle(&reshaped), handle(&reshaped)],
            )
            .expect("mul executes");
            dispatch::execute::<op::MeanAll, _>(&context, NoAttributes, &[handle(&scaled)])
                .expect("mean_all executes")
        },
        &[input],
        GRADIENT_STEP,
    );
    assert!(
        error < GRADIENT_TOLERANCE,
        "canonical reshape gradient error {error} exceeds {GRADIENT_TOLERANCE}"
    );
}

#[test]
fn canonical_axis_reduction_gradients_match_finite_differences() {
    let context = context();
    let input = storage(&[0.5, 1.5, -2.0, 3.0, 0.25, -0.75], &[2, 3]);

    let error = gradcheck(
        |inputs| {
            let reduced = dispatch::execute::<op::SumDim, _>(
                &context,
                AxisAttributes { axis: 1 },
                &[handle(&inputs[0])],
            )
            .expect("sum_dim executes");
            let squared = dispatch::execute::<op::Mul, _>(
                &context,
                NoAttributes,
                &[handle(&reduced), handle(&reduced)],
            )
            .expect("mul executes");
            dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &[handle(&squared)])
                .expect("sum_all executes")
        },
        &[input],
        GRADIENT_STEP,
    );
    assert!(
        error < GRADIENT_TOLERANCE,
        "canonical sum_dim gradient error {error} exceeds {GRADIENT_TOLERANCE}"
    );
}

#[test]
fn canonical_matmul_gradients_match_finite_differences() {
    let context = context();
    let lhs = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let rhs = storage(&[0.5, -1.0, 2.0, 0.25, -0.5, 1.5], &[3, 2]);

    let error = gradcheck(
        |inputs| {
            let product = dispatch::execute::<op::MatMulExact, _>(
                &context,
                NoAttributes,
                &[handle(&inputs[0]), handle(&inputs[1])],
            )
            .expect("matmul executes");
            dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &[handle(&product)])
                .expect("sum_all executes")
        },
        &[lhs, rhs],
        GRADIENT_STEP,
    );
    assert!(
        error < GRADIENT_TOLERANCE,
        "canonical matmul gradient error {error} exceeds {GRADIENT_TOLERANCE}"
    );
}

#[test]
fn canonical_and_backend_helper_gradients_are_identical() {
    use crate::cpu::tape;

    let context = context();
    let lhs = storage(&[0.5, 1.5, -2.0, 3.0], &[4]);
    let rhs = storage(&[2.0, -1.0, 0.5, 1.25], &[4]);

    let canonical_scalar = {
        let product =
            dispatch::execute::<op::Mul, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
                .expect("mul executes");
        dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &[handle(&product)])
            .expect("sum_all executes")
    };
    let canonical = tape::backward(&canonical_scalar).expect("backward succeeds");
    let canonical_lhs = canonical
        .get(lhs.id)
        .expect("lhs receives a gradient")
        .clone();
    let canonical_rhs = canonical
        .get(rhs.id)
        .expect("rhs receives a gradient")
        .clone();

    let helper_product =
        crate::cpu::ops::elementwise::mul_storage(&lhs, &rhs).expect("helper mul executes");
    let helper_scalar =
        crate::cpu::ops::reduce::sum_all(&helper_product).expect("helper sum_all executes");
    let helper = tape::backward(&helper_scalar).expect("backward succeeds");
    let helper_lhs = helper.get(lhs.id).expect("lhs receives a gradient");
    let helper_rhs = helper.get(rhs.id).expect("rhs receives a gradient");

    for (index, (canonical, helper)) in [(&canonical_lhs, helper_lhs), (&canonical_rhs, helper_rhs)]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            canonical.shape.to_vec(),
            helper.shape.to_vec(),
            "operand {index} gradient shape diverged"
        );
        for flat in 0..canonical.shape.iter().product::<usize>() {
            let mut multi = vec![0usize; canonical.shape.len()];
            let mut remaining = flat;
            for axis in (0..canonical.shape.len()).rev() {
                multi[axis] = remaining % canonical.shape[axis];
                remaining /= canonical.shape[axis];
            }
            assert_eq!(
                canonical.get(&multi),
                helper.get(&multi),
                "operand {index} gradient diverged at {multi:?}"
            );
        }
    }
}

fn batch_norm_attributes(training: bool, epsilon: f64) -> BatchNormAttributes {
    BatchNormAttributes {
        epsilon,
        momentum: 0.1,
        training,
        has_weight: false,
        has_bias: false,
        has_running_mean: !training,
        has_running_variance: !training,
    }
}

#[test]
fn a_training_batch_norm_normalizes_by_the_batch_statistics() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let normalized = dispatch::execute::<op::BatchNorm, _>(
        &context,
        batch_norm_attributes(true, 1e-5),
        &[handle(&input)],
    )
    .expect("a training batch norm computes batch statistics");

    assert_eq!(normalized.shape.to_vec(), vec![2, 3]);
    for row in 0..2 {
        for column in 0..3 {
            let value = normalized.get(&[row, column]);
            let expected = if row == 0 { -1.0 } else { 1.0 };
            assert!(
                (value - expected).abs() < 1e-3,
                "[{row}, {column}] was {value}, expected {expected}"
            );
        }
    }
}

#[test]
fn training_and_inference_batch_norm_disagree() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let running_mean = storage(&[0.0, 0.0, 0.0], &[3]);
    let running_variance = storage(&[1.0, 1.0, 1.0], &[3]);

    let training = dispatch::execute::<op::BatchNorm, _>(
        &context,
        batch_norm_attributes(true, 1e-5),
        &[handle(&input)],
    )
    .unwrap();
    let inference = dispatch::execute::<op::BatchNorm, _>(
        &context,
        batch_norm_attributes(false, 1e-5),
        &[
            handle(&input),
            handle(&running_mean),
            handle(&running_variance),
        ],
    )
    .unwrap();

    assert!(
        (training.get(&[0, 0]) - inference.get(&[0, 0])).abs() > 1e-3,
        "training {} and inference {} agree",
        training.get(&[0, 0]),
        inference.get(&[0, 0])
    );
}

#[test]
fn an_inference_batch_norm_with_running_statistics_executes() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let running_mean = storage(&[0.0, 0.0, 0.0], &[3]);
    let running_variance = storage(&[1.0, 1.0, 1.0], &[3]);

    let output = dispatch::execute::<op::BatchNorm, _>(
        &context,
        batch_norm_attributes(false, 1e-5),
        &[
            handle(&input),
            handle(&running_mean),
            handle(&running_variance),
        ],
    )
    .expect("an inference batch norm with running statistics executes");
    assert_eq!(output.shape.to_vec(), vec![2, 3]);
}

#[test]
fn an_epsilon_that_flushes_to_zero_in_f32_is_refused() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let running_mean = storage(&[0.0, 0.0, 0.0], &[3]);
    let running_variance = storage(&[1.0, 1.0, 1.0], &[3]);

    let error = dispatch::execute::<op::BatchNorm, _>(
        &context,
        batch_norm_attributes(false, 1e-300),
        &[
            handle(&input),
            handle(&running_mean),
            handle(&running_variance),
        ],
    )
    .expect_err("an epsilon that narrows to zero is not the epsilon that was asked for");
    let message = format!("{error}");
    assert!(
        message.contains("positive finite f32"),
        "the refusal must name the reason, not just fail: {message}"
    );
}

#[test]
fn canonical_layer_norm_executes_and_normalizes_each_row() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let weight = storage(&[1.0, 1.0, 1.0], &[3]);

    let output = dispatch::execute::<op::LayerNorm, _>(
        &context,
        LayerNormAttributes {
            normalized_shape: vec![3],
            epsilon: 1e-5,
            has_bias: false,
        },
        &[handle(&input), handle(&weight)],
    )
    .expect("layer norm executes");
    assert_eq!(output.shape.to_vec(), vec![2, 3]);
    for row in 0..2 {
        let mean: f64 = (0..3).map(|column| output.get(&[row, column])).sum::<f64>() / 3.0;
        assert!(
            mean.abs() < 1e-5,
            "row {row} was not normalized: mean {mean}"
        );
    }
}

#[test]
fn a_convolution_with_a_bias_is_not_refused_by_its_own_rank_bound() {
    let context = context();
    let activation = storage(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let weight = storage(&[1.0], &[1, 1, 1, 1]);
    let bias = storage(&[0.5], &[1]);

    let output = dispatch::execute::<op::Conv2dExact, _>(
        &context,
        Conv2dAttributes {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            has_bias: true,
        },
        &[handle(&activation), handle(&weight), handle(&bias)],
    )
    .expect("a biased conv2d executes");
    assert_eq!(output.shape.to_vec(), vec![1, 1, 2, 2]);
}

#[test]
fn each_canonical_loss_computes_its_own_formula() {
    let context = context();
    let prediction = storage(&[1.0, 3.0], &[2]);
    let target = storage(&[0.0, 0.0], &[2]);

    for (expected, run) in [
        (
            5.0,
            dispatch::execute::<op::MseLoss, _>(
                &context,
                LossAttributes {
                    reduction: LossReduction::Mean,
                },
                &[handle(&prediction), handle(&target)],
            ),
        ),
        (
            2.0,
            dispatch::execute::<op::L1Loss, _>(
                &context,
                LossAttributes {
                    reduction: LossReduction::Mean,
                },
                &[handle(&prediction), handle(&target)],
            ),
        ),
    ] {
        let output = run.expect("the loss executes");
        assert!(
            output.shape.is_empty(),
            "a mean reduction produces a scalar"
        );
        assert!(
            (output.get(&[]) - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            output.get(&[])
        );
    }
}

#[test]
fn a_loss_with_no_reduction_keeps_the_elementwise_shape() {
    let context = context();
    let prediction = storage(&[1.0, 3.0], &[2]);
    let target = storage(&[0.0, 0.0], &[2]);

    let output = dispatch::execute::<op::MseLoss, _>(
        &context,
        LossAttributes {
            reduction: LossReduction::None,
        },
        &[handle(&prediction), handle(&target)],
    )
    .expect("an unreduced loss executes");
    assert_eq!(output.shape.to_vec(), vec![2]);
    assert_eq!(output.get(&[0]), 1.0);
    assert_eq!(output.get(&[1]), 9.0);
}

#[test]
fn an_allocation_is_refused_by_a_row_it_does_not_match() {
    let context = inference_context();

    let error = dispatch::execute::<op::UniformRandom, _>(
        &context,
        CreationAttributes {
            shape: vec![2, 2],
            dtype: DTypeId::I64.descriptor(),
            device: DeviceId::cpu(),
        },
        &[],
    )
    .expect_err("a uniform draw over an integer dtype is not advertised");
    let message = format!("{error}");
    assert!(
        message.to_lowercase().contains("dtype") || message.contains("I64"),
        "the refusal must name the dtype the row does not carry: {message}"
    );

    dispatch::execute::<op::UniformRandom, _>(
        &context,
        CreationAttributes {
            shape: vec![2, 2],
            dtype: DTypeId::F32.descriptor(),
            device: DeviceId::cpu(),
        },
        &[],
    )
    .expect("a uniform draw over f32 is advertised");
}

#[test]
fn the_ranged_fills_read_their_parameters_in_the_right_order() {
    let context = inference_context();
    let shape = vec![4];
    let device = DeviceId::cpu();

    let stepped = dispatch::execute::<op::Arange, _>(
        &context,
        ArangeAttributes {
            shape: shape.clone(),
            dtype: DTypeId::F32.descriptor(),
            device,
            start: 10.0,
            step: 2.0,
        },
        &[],
    )
    .expect("arange executes");
    for (index, expected) in [10.0, 12.0, 14.0, 16.0].into_iter().enumerate() {
        assert_eq!(stepped.get(&[index]), expected);
    }

    let spaced = dispatch::execute::<op::Linspace, _>(
        &context,
        LinspaceAttributes {
            shape,
            dtype: DTypeId::F32.descriptor(),
            device,
            start: 0.0,
            end: 3.0,
        },
        &[],
    )
    .expect("linspace executes");
    for (index, expected) in [0.0, 1.0, 2.0, 3.0].into_iter().enumerate() {
        assert!(
            (spaced.get(&[index]) - expected).abs() < 1e-6,
            "element {index}: expected {expected}, got {}",
            spaced.get(&[index])
        );
    }
}

#[test]
fn an_allocation_given_an_operand_is_refused() {
    let context = inference_context();
    let stray = storage(&[1.0], &[1]);

    let error = dispatch::execute::<op::Zeros, _>(
        &context,
        CreationAttributes {
            shape: vec![2],
            dtype: DTypeId::F32.descriptor(),
            device: DeviceId::cpu(),
        },
        &[handle(&stray)],
    )
    .expect_err("zeros reads nothing, so an operand is a malformed request");
    assert!(format!("{error}").contains("zeros"));
}

#[test]
fn dropout_zeroes_some_elements_and_scales_the_rest_by_the_keep_reciprocal() {
    let context = context();
    let input = storage(&[1.0; 4096], &[4096]);

    let output = dispatch::execute::<op::Dropout, _>(
        &context,
        DropoutAttributes {
            probability: 0.5,
            training: true,
        },
        &[handle(&input)],
    )
    .expect("dropout executes");
    assert_eq!(output.shape.to_vec(), vec![4096]);

    let mut kept = 0;
    for index in 0..4096 {
        let value = output.get(&[index]);
        if value == 0.0 {
            continue;
        }
        kept += 1;
        assert!(
            (value - 2.0).abs() < 1e-6,
            "a surviving element was {value}, not the input scaled by 1 / (1 - p)"
        );
    }
    assert!(
        (2048 - 128..=2048 + 128).contains(&kept),
        "kept {kept} of 4096 elements, which is not a half-probability drop"
    );
}

#[test]
fn dropout_outside_training_returns_the_operand_unchanged() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0, 4.0], &[4]);

    for (probability, training) in [(0.5, false), (0.0, true)] {
        let output = dispatch::execute::<op::Dropout, _>(
            &context,
            DropoutAttributes {
                probability,
                training,
            },
            &[handle(&input)],
        )
        .expect("dropout executes");
        for index in 0..4 {
            assert_eq!(
                output.get(&[index]),
                input.get(&[index]),
                "p={probability} training={training} changed element {index}"
            );
        }
    }
}

#[test]
fn a_linear_layer_transposes_its_weight_and_adds_its_bias() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0], &[1, 3]);
    let weight = storage(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
    let bias = storage(&[10.0, 20.0], &[2]);

    let plain = dispatch::execute::<op::Linear, _>(
        &context,
        LinearAttributes { has_bias: false },
        &[handle(&input), handle(&weight)],
    )
    .expect("linear executes");
    assert_eq!(plain.shape.to_vec(), vec![1, 2]);
    assert_eq!(plain.get(&[0, 0]), 1.0);
    assert_eq!(plain.get(&[0, 1]), 2.0);

    let biased = dispatch::execute::<op::Linear, _>(
        &context,
        LinearAttributes { has_bias: true },
        &[handle(&input), handle(&weight), handle(&bias)],
    )
    .expect("a biased linear executes");
    assert_eq!(biased.get(&[0, 0]), 11.0);
    assert_eq!(biased.get(&[0, 1]), 22.0);
}

#[test]
fn rms_norm_scales_by_the_root_mean_square_without_centring() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0], &[1, 3]);
    let weight = storage(&[1.0, 1.0, 1.0], &[3]);

    let output = dispatch::execute::<op::RmsNorm, _>(
        &context,
        EpsilonAttributes { epsilon: 0.0 },
        &[handle(&input), handle(&weight)],
    )
    .expect("rms_norm executes");
    assert_eq!(output.shape.to_vec(), vec![1, 3]);

    let root_mean_square = (14.0f64 / 3.0).sqrt();
    for (index, original) in [1.0, 2.0, 3.0].into_iter().enumerate() {
        let expected: f64 = original / root_mean_square;
        let actual = output.get(&[0, index]);
        assert!(
            (actual - expected).abs() < 1e-6,
            "element {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn quantization_round_trips_within_the_block_format_error() {
    let context = inference_context();
    let values: Vec<f32> = (0..32).map(|index| index as f32 - 8.0).collect();
    let input = storage(&values, &[32]);

    let blocks = dispatch::execute::<op::Quantize, _>(
        &context,
        QuantizationAttributes {
            dtype: DTypeId::Q8_0.descriptor(),
        },
        &[handle(&input)],
    )
    .expect("quantize executes");
    let restored = dispatch::execute::<op::Dequantize, _>(
        &context,
        QuantizationAttributes {
            dtype: DTypeId::F32.descriptor(),
        },
        &[handle(&blocks)],
    )
    .expect("dequantize executes");
    assert_eq!(blocks.dtype, DTypeId::Q8_0.descriptor());
    assert_eq!(restored.dtype, DTypeId::F32.descriptor());
    assert_eq!(restored.shape.to_vec(), vec![32]);

    let largest = values
        .iter()
        .fold(0.0f32, |seen, &value| seen.max(value.abs()));
    let tolerance = f64::from(largest / 127.0 / 2.0) + 1e-6;
    for (index, &original) in values.iter().enumerate() {
        let difference = (restored.get(&[index]) - f64::from(original)).abs();
        assert!(
            difference <= tolerance,
            "element {index} moved by {difference}, more than the {tolerance} the \
             block scale allows"
        );
    }
}

#[test]
fn a_compression_into_an_unsupported_representation_is_refused() {
    let context = inference_context();
    let input = storage(&[0.0; 32], &[32]);

    let error = dispatch::execute::<op::Quantize, _>(
        &context,
        QuantizationAttributes {
            dtype: DTypeId::F16.descriptor(),
        },
        &[handle(&input)],
    )
    .expect_err("f16 is not a quantized representation this backend produces");
    let message = format!("{error}");
    assert!(
        message.contains("quantize"),
        "the refusal must name the operation: {message}"
    );
}

#[test]
fn the_dot_and_outer_products_contract_and_expand() {
    let context = context();
    let lhs = storage(&[1.0, 2.0], &[2]);
    let rhs = storage(&[3.0, 4.0], &[2]);

    let inner =
        dispatch::execute::<op::Dot, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .expect("dot executes");
    assert!(inner.shape.is_empty(), "a dot product is a scalar");
    assert_eq!(inner.get(&[]), 11.0);

    let grid =
        dispatch::execute::<op::Outer, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .expect("outer executes");
    assert_eq!(grid.shape.to_vec(), vec![2, 2]);
    for (row, left) in [1.0, 2.0].into_iter().enumerate() {
        for (column, right) in [3.0, 4.0].into_iter().enumerate() {
            assert_eq!(grid.get(&[row, column]), left * right);
        }
    }
}

#[test]
fn an_uneven_axis_leaves_a_shorter_final_piece() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);

    let chunks = dispatch::execute::<op::Chunk, _>(
        &context,
        ChunkAttributes { chunks: 2, axis: 0 },
        &[handle(&input)],
    )
    .expect("chunk executes");
    assert_eq!(
        chunks
            .iter()
            .map(|piece| piece.shape.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![3], vec![2]]
    );
    assert_eq!(chunks[1].get(&[0]), 4.0);

    let pieces = dispatch::execute::<op::Split, _>(
        &context,
        SplitAttributes {
            split_size: 2,
            axis: 0,
        },
        &[handle(&input)],
    )
    .expect("split executes");
    assert_eq!(
        pieces
            .iter()
            .map(|piece| piece.shape.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![2], vec![2], vec![1]]
    );
    assert_eq!(pieces[2].get(&[0]), 5.0);
}

#[test]
fn chunking_beyond_the_axis_extent_produces_fewer_pieces_not_empty_ones() {
    let context = context();
    let input = storage(&[1.0, 2.0], &[2]);

    let chunks = dispatch::execute::<op::Chunk, _>(
        &context,
        ChunkAttributes { chunks: 5, axis: 0 },
        &[handle(&input)],
    )
    .expect("chunk executes");
    assert_eq!(chunks.len(), 2, "a two-wide axis has at most two pieces");
    assert!(chunks.iter().all(|piece| piece.shape.to_vec() == vec![1]));
}

#[test]
fn the_variance_estimators_differ_by_their_correction() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0, 4.0], &[4]);

    for (unbiased, expected) in [(false, 1.25), (true, 5.0 / 3.0)] {
        let output = dispatch::execute::<op::VarianceAll, _>(
            &context,
            VarianceAttributes { unbiased },
            &[handle(&input)],
        )
        .expect("var_all executes");
        assert!(
            (output.get(&[]) - expected).abs() < 1e-6,
            "unbiased={unbiased}: expected {expected}, got {}",
            output.get(&[])
        );
    }
}

#[test]
fn the_axis_variance_forms_reduce_the_axis_they_name() {
    let context = context();
    let input = storage(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    let attributes = AxisVarianceAttributes {
        axis: 1,
        unbiased: false,
    };
    let expected = 2.0 / 3.0;

    let reduced =
        dispatch::execute::<op::VarianceDim, _>(&context, attributes.clone(), &[handle(&input)])
            .expect("var_dim executes");
    assert_eq!(reduced.shape.to_vec(), vec![2]);
    assert!((reduced.get(&[0]) - expected).abs() < 1e-6);

    let kept = dispatch::execute::<op::VarianceKeepDim, _>(
        &context,
        attributes.clone(),
        &[handle(&input)],
    )
    .expect("var_keepdim executes");
    assert_eq!(kept.shape.to_vec(), vec![2, 1]);
    assert!((kept.get(&[0, 0]) - expected).abs() < 1e-6);

    let deviation = dispatch::execute::<op::StdDim, _>(&context, attributes, &[handle(&input)])
        .expect("std_dim executes");
    assert_eq!(deviation.shape.to_vec(), vec![2]);
    assert!((deviation.get(&[0]) - expected.sqrt()).abs() < 1e-6);
}

#[test]
fn the_norm_orders_agree_where_the_fast_paths_meet_the_general_one() {
    let context = context();
    let input = storage(&[3.0, -4.0], &[2]);

    let l1 = dispatch::execute::<op::Norm, _>(
        &context,
        NormAttributes { order: 1.0 },
        &[handle(&input)],
    )
    .expect("the l1 norm executes");
    assert!((l1.get(&[]) - 7.0).abs() < 1e-6, "got {}", l1.get(&[]));

    let l2 = dispatch::execute::<op::Norm, _>(
        &context,
        NormAttributes { order: 2.0 },
        &[handle(&input)],
    )
    .expect("the l2 norm executes");
    assert!((l2.get(&[]) - 5.0).abs() < 1e-6, "got {}", l2.get(&[]));

    let near = dispatch::execute::<op::Norm, _>(
        &context,
        NormAttributes { order: 2.001 },
        &[handle(&input)],
    )
    .expect("a general-order norm executes");
    assert!(
        (near.get(&[]) - 5.0).abs() < 1e-2,
        "the general path diverged from the fast one: {}",
        near.get(&[])
    );
}

#[test]
fn a_conversion_to_a_quantized_dtype_is_refused_by_name() {
    let context = context();
    let input = storage(&[1.0, 2.0], &[2]);

    let error = dispatch::execute::<op::ToDType, _>(
        &context,
        DTypeAttributes {
            dtype: DTypeId::Q8_0.descriptor(),
        },
        &[handle(&input)],
    )
    .expect_err("the CPU conversion kernel has no quantized target");
    let message = format!("{error}");
    assert!(
        message.contains("to_dtype"),
        "the refusal must name the operation: {message}"
    );
}

#[test]
fn canonical_conv1d_executes_at_the_ranks_its_row_advertises() {
    let context = context();
    let activation = storage(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let weight = storage(&[1.0, 1.0], &[1, 1, 2]);

    let output = dispatch::execute::<op::Conv1dExact, _>(
        &context,
        Conv1dAttributes {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            has_bias: false,
        },
        &[handle(&activation), handle(&weight)],
    )
    .expect("conv1d executes");
    assert_eq!(output.shape.to_vec(), vec![1, 1, 3]);
    for (index, expected) in [3.0, 5.0, 7.0].into_iter().enumerate() {
        assert_eq!(output.get(&[0, 0, index]), expected);
    }
}

#[test]
fn a_non_f32_convolution_operand_is_refused() {
    let context = context();
    let activation = storage(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let weight = CpuStorage::try_from_contiguous(CpuBuffer::F64(vec![1.0, 1.0]), vec![1, 1, 2])
        .expect("test storage must be well formed");

    let error = dispatch::execute::<op::Conv1dExact, _>(
        &context,
        Conv1dAttributes {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            has_bias: false,
        },
        &[handle(&activation), handle(&weight)],
    )
    .expect_err("an f64 weight is not a dtype this kernel honours");
    let message = format!("{error}");
    assert!(
        message.to_lowercase().contains("dtype") || message.contains("F64"),
        "the refusal must name the dtype: {message}"
    );
}
