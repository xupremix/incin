use super::*;

/// [`broadcast_shape`] over the inputs of an invocation, without materialising
/// the list of borrowed shape slices.
fn broadcast_input_shapes(
    operation: OperationKind,
    inputs: &[LogicalTensorMeta],
) -> Result<ShapeBuf, DescriptorError> {
    let mut shapes = inputs.iter().filter_map(|input| input.shape.as_deref());
    let Some(first) = shapes.next() else {
        return Ok(ShapeBuf::scalar());
    };

    // Operands that already agree broadcast to themselves. This is the ordinary
    // elementwise case, and taking it directly skips a fallible right-aligned
    // loop and its allocation for every operand after the first.
    if shapes.clone().all(|shape| shape == first) {
        validate_shape(operation, first)?;
        return Ok(ShapeBuf::from_slice(first));
    }

    let mut output: Vec<usize> = first.to_vec();
    for shape in shapes {
        output = crate::shapes::broadcast::broadcast_dim_slices(&output, shape).map_err(|_| {
            invalid(
                operation,
                "shape",
                "input shapes are not broadcast-compatible",
            )
        })?;
    }
    validate_shape(operation, &output)?;
    Ok(ShapeBuf::from_slice(&output))
}

fn broadcast_shape(
    operation: OperationKind,
    shapes: &[&[usize]],
) -> Result<ShapeBuf, DescriptorError> {
    // `broadcast_dim_slices` already returns the owned dims, so the result of
    // each step becomes the accumulator directly. The previous form copied it
    // into a `ShapeBuf` and then back out into a `Vec` on every iteration,
    // which is three allocations for the two-operand case that dominates.
    //
    // Seeding with the first shape rather than with a scalar is the same
    // computation: broadcasting rank-0 against `s` yields `s`.
    let mut output: Vec<usize> = match shapes.split_first() {
        Some((first, _)) => first.to_vec(),
        None => Vec::new(),
    };
    for shape in shapes.iter().skip(1) {
        output = crate::shapes::broadcast::broadcast_dim_slices(&output, shape).map_err(|_| {
            invalid(
                operation,
                "shape",
                "input shapes are not broadcast-compatible",
            )
        })?;
    }
    validate_shape(operation, &output)?;
    Ok(ShapeBuf::from_slice(&output))
}

fn transformed_shape<A: AttributeContract>(
    operation: OperationKind,
    attributes: &A,
    inputs: &[LogicalTensorMeta],
    output_index: usize,
) -> Result<Option<Option<ShapeBuf>>, DescriptorError> {
    let Some(transform) = attributes.shape_transform() else {
        return Ok(None);
    };
    let Some(input) = inputs.first().and_then(|input| input.shape.as_deref()) else {
        return Ok(Some(None));
    };
    fn spatial_output(
        operation: OperationKind,
        input: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<usize, DescriptorError> {
        let effective = kernel
            .checked_sub(1)
            .and_then(|value| value.checked_mul(dilation))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid(operation, "spatial", "effective kernel overflows usize"))?;
        let padded = input
            .checked_add(padding)
            .and_then(|value| value.checked_add(padding))
            .ok_or_else(|| invalid(operation, "spatial", "padded extent overflows usize"))?;
        if effective > padded {
            return Err(invalid(
                operation,
                "spatial",
                "effective kernel exceeds padded input extent",
            ));
        }
        Ok((padded - effective) / stride + 1)
    }

    fn convolution_output(
        operation: OperationKind,
        input: &[usize],
        weight: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
    ) -> Result<Vec<usize>, DescriptorError> {
        let dimensions = stride.len();
        if input.len() < dimensions + 1 || weight.len() != dimensions + 2 {
            return Err(invalid(
                operation,
                "rank",
                "convolution operand rank is invalid",
            ));
        }
        let mut output = input.to_vec();
        output[input.len() - dimensions - 1] = weight[0];
        for axis in 0..dimensions {
            output[input.len() - dimensions + axis] = spatial_output(
                operation,
                input[input.len() - dimensions + axis],
                weight[weight.len() - dimensions + axis],
                stride[axis],
                padding[axis],
                dilation[axis],
            )?;
        }
        Ok(output)
    }

    let output = match transform {
        ShapeTransform::Axis(axis) => match operation {
            OperationKind::SqueezeExact => {
                if input[axis] != 1 {
                    return Err(invalid(operation, "axis", "squeeze extent must equal one"));
                }
                let mut output = input.to_vec();
                output.remove(axis);
                output
            }
            OperationKind::UnsqueezeExact => {
                let mut output = input.to_vec();
                output.insert(axis, 1);
                output
            }
            OperationKind::StackExact => {
                let mut output = input.to_vec();
                output.insert(axis, inputs.len());
                output
            }
            OperationKind::ConcatExact => {
                let mut output = input.to_vec();
                let mut extent = 0usize;
                for other in inputs {
                    let Some(shape) = other.shape.as_deref() else {
                        return Ok(Some(None));
                    };
                    if shape.len() != input.len()
                        || shape
                            .iter()
                            .enumerate()
                            .any(|(index, value)| index != axis && *value != input[index])
                    {
                        return Err(invalid(
                            operation,
                            "shape",
                            "concat inputs must match outside the concat axis",
                        ));
                    }
                    extent = extent.checked_add(shape[axis]).ok_or_else(|| {
                        invalid(operation, "shape", "concat extent overflows usize")
                    })?;
                }
                output[axis] = extent;
                output
            }
            _ => return Ok(None),
        },
        ShapeTransform::Transpose(first, second) => {
            let mut output = input.to_vec();
            output.swap(first, second);
            output
        }
        ShapeTransform::Narrow { axis, length } => {
            let mut output = input.to_vec();
            output[axis] = length;
            output
        }
        ShapeTransform::Slice(ranges) => ranges.iter().map(|(start, end)| end - start).collect(),
        ShapeTransform::Flatten { start, end } => {
            let flattened = input[start..=end]
                .iter()
                .try_fold(1usize, |value, &extent| {
                    value.checked_mul(extent).ok_or_else(|| {
                        invalid(operation, "shape", "flattened extent overflows usize")
                    })
                })?;
            let mut output = Vec::with_capacity(input.len() - (end - start));
            output.extend_from_slice(&input[..start]);
            output.push(flattened);
            output.extend_from_slice(&input[end + 1..]);
            output
        }
        ShapeTransform::Repeat(repeats) => input
            .iter()
            .zip(repeats)
            .map(|(&extent, &repeat)| {
                extent
                    .checked_mul(repeat)
                    .ok_or_else(|| invalid(operation, "repeats", "repeated extent overflows usize"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        ShapeTransform::Pad(padding) => input
            .iter()
            .zip(padding)
            .map(|(&extent, &(before, after))| {
                extent
                    .checked_add(before)
                    .and_then(|value| value.checked_add(after))
                    .ok_or_else(|| invalid(operation, "padding", "padded extent overflows usize"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        ShapeTransform::Diagonal(offset) if operation == OperationKind::Diag => {
            let displacement = usize::try_from(offset.unsigned_abs())
                .map_err(|_| invalid(operation, "offset", "diagonal offset does not fit usize"))?;
            if input.len() == 1 {
                let extent = input[0].checked_add(displacement).ok_or_else(|| {
                    invalid(operation, "offset", "diagonal extent overflows usize")
                })?;
                vec![extent, extent]
            } else {
                let rows = input[input.len() - 2];
                let columns = input[input.len() - 1];
                let extent = if offset >= 0 {
                    rows.min(columns.saturating_sub(displacement))
                } else {
                    rows.saturating_sub(displacement).min(columns)
                };
                vec![extent]
            }
        }
        ShapeTransform::Diagonal(_) => input.to_vec(),
        ShapeTransform::Unfold { axis, size, step } => {
            let extent = input[axis];
            if size > extent {
                return Err(invalid(
                    operation,
                    "size",
                    "unfold size exceeds the selected extent",
                ));
            }
            let mut output = input.to_vec();
            output[axis] = (extent - size) / step + 1;
            output.push(size);
            output
        }
        ShapeTransform::PixelShuffle(factor) => {
            let square = factor.checked_mul(factor).ok_or_else(|| {
                invalid(
                    operation,
                    "upscale_factor",
                    "squared factor overflows usize",
                )
            })?;
            if input[1] % square != 0 {
                return Err(invalid(
                    operation,
                    "channels",
                    "channel extent must be divisible by the squared upscale factor",
                ));
            }
            vec![
                input[0],
                input[1] / square,
                input[2].checked_mul(factor).ok_or_else(|| {
                    invalid(operation, "height", "pixel-shuffle height overflows usize")
                })?,
                input[3].checked_mul(factor).ok_or_else(|| {
                    invalid(operation, "width", "pixel-shuffle width overflows usize")
                })?,
            ]
        }
        ShapeTransform::AdaptivePool2d(spatial) => {
            let mut output = input.to_vec();
            let rank = output.len();
            output[rank - 2..].copy_from_slice(&spatial);
            output
        }
        ShapeTransform::TopK { axis, k } => {
            let mut output = input.to_vec();
            output[axis] = k;
            output
        }
        ShapeTransform::Chunk { chunks, axis } => {
            let extent = input[axis];
            let chunk_size = extent.div_ceil(chunks);
            let start = output_index
                .checked_mul(chunk_size)
                .ok_or_else(|| invalid(operation, "output", "chunk offset overflows usize"))?;
            if start >= extent {
                return Err(invalid(
                    operation,
                    "output",
                    "chunk output index is invalid",
                ));
            }
            let mut output = input.to_vec();
            output[axis] = (extent - start).min(chunk_size);
            output
        }
        ShapeTransform::Split { split_size, axis } => {
            let extent = input[axis];
            let start = output_index
                .checked_mul(split_size)
                .ok_or_else(|| invalid(operation, "output", "split offset overflows usize"))?;
            if start >= extent {
                return Err(invalid(
                    operation,
                    "output",
                    "split output index is invalid",
                ));
            }
            let mut output = input.to_vec();
            output[axis] = (extent - start).min(split_size);
            output
        }
        ShapeTransform::Conv1d(attributes) => {
            let Some(weight) = inputs.get(1).and_then(|value| value.shape.as_deref()) else {
                return Ok(Some(None));
            };
            if weight.len() != 3
                || weight[1]
                    .checked_mul(attributes.groups)
                    .is_none_or(|channels| channels != input[input.len() - 2])
                || inputs
                    .get(2)
                    .and_then(|value| value.shape.as_deref())
                    .is_some_and(|bias| bias != [weight[0]])
            {
                return Err(invalid(
                    operation,
                    "channels",
                    "conv1d input, grouped weight, and optional bias channels disagree",
                ));
            }
            convolution_output(
                operation,
                input,
                weight,
                &[attributes.stride],
                &[attributes.padding],
                &[attributes.dilation],
            )?
        }
        ShapeTransform::Conv2d(attributes) => {
            let Some(weight) = inputs.get(1).and_then(|value| value.shape.as_deref()) else {
                return Ok(Some(None));
            };
            if weight.len() != 4
                || weight[1]
                    .checked_mul(attributes.groups)
                    .is_none_or(|channels| channels != input[input.len() - 3])
                || inputs
                    .get(2)
                    .and_then(|value| value.shape.as_deref())
                    .is_some_and(|bias| bias != [weight[0]])
            {
                return Err(invalid(
                    operation,
                    "channels",
                    "conv2d input, grouped weight, and optional bias channels disagree",
                ));
            }
            convolution_output(
                operation,
                input,
                weight,
                &attributes.stride,
                &attributes.padding,
                &attributes.dilation,
            )?
        }
        ShapeTransform::ConvTranspose2d(attributes) => {
            let Some(weight) = inputs.get(1).and_then(|value| value.shape.as_deref()) else {
                return Ok(Some(None));
            };
            if input.len() < 3 || weight.len() != 4 {
                return Err(invalid(
                    operation,
                    "rank",
                    "transposed convolution requires rank-three/four input and rank-four weight",
                ));
            }
            if input[input.len() - 3] != weight[0]
                || inputs
                    .get(2)
                    .and_then(|value| value.shape.as_deref())
                    .is_some_and(|bias| {
                        weight[1]
                            .checked_mul(attributes.groups)
                            .is_none_or(|channels| bias != [channels])
                    })
            {
                return Err(invalid(
                    operation,
                    "channels",
                    "transposed convolution input, weight, and optional bias channels disagree",
                ));
            }
            let mut output = input.to_vec();
            let channel_axis = input.len() - 3;
            output[channel_axis] = weight[1]
                .checked_mul(attributes.groups)
                .ok_or_else(|| invalid(operation, "groups", "output channels overflow usize"))?;
            for axis in 0..2 {
                let source = input[input.len() - 2 + axis];
                let kernel = weight[weight.len() - 2 + axis];
                let extent = source
                    .checked_sub(1)
                    .and_then(|value| value.checked_mul(attributes.stride[axis]))
                    .and_then(|value| {
                        kernel
                            .checked_sub(1)
                            .and_then(|kernel| kernel.checked_mul(attributes.dilation[axis]))
                            .and_then(|kernel| value.checked_add(kernel))
                    })
                    .and_then(|value| value.checked_add(attributes.output_padding[axis]))
                    .and_then(|value| value.checked_add(1))
                    .and_then(|value| {
                        attributes.padding[axis]
                            .checked_mul(2)
                            .and_then(|padding| value.checked_sub(padding))
                    })
                    .ok_or_else(|| {
                        invalid(
                            operation,
                            "spatial",
                            "transposed convolution output extent overflows or underflows",
                        )
                    })?;
                output[input.len() - 2 + axis] = extent;
            }
            output
        }
        ShapeTransform::Pool2d(attributes) => {
            let mut output = input.to_vec();
            for axis in 0..2 {
                output[input.len() - 2 + axis] = spatial_output(
                    operation,
                    input[input.len() - 2 + axis],
                    attributes.kernel[axis],
                    attributes.stride[axis],
                    attributes.padding[axis],
                    attributes.dilation[axis],
                )?;
            }
            output
        }
        ShapeTransform::AvgPool2d(attributes) => {
            let mut output = input.to_vec();
            for axis in 0..2 {
                output[input.len() - 2 + axis] = spatial_output(
                    operation,
                    input[input.len() - 2 + axis],
                    attributes.kernel[axis],
                    attributes.stride[axis],
                    attributes.padding[axis],
                    1,
                )?;
            }
            output
        }
        ShapeTransform::Rnn(attributes) => {
            if output_index == 0 {
                vec![input[0], input[1], attributes.hidden_size]
            } else {
                vec![input[0], attributes.hidden_size]
            }
        }
    };
    validate_shape(operation, &output)?;
    Ok(Some(Some(ShapeBuf::from_slice(&output))))
}

fn inferred_shape<A: AttributeContract>(
    operation: OperationKind,
    row: &OperationCatalogEntry,
    attributes: &A,
    inputs: &[LogicalTensorMeta],
    output_index: usize,
) -> Result<Option<Option<ShapeBuf>>, DescriptorError> {
    // Materialised lazily. Computing it up front cost an allocation on every
    // operation whose rule never reads it, which is every broadcast, matmul and
    // reduction: the overwhelming majority of a training step.
    let first = || inputs.first().and_then(|input| input.shape.clone());
    let inferred = match row.output {
        OutputRule::Created => Some(attributes.declared_shape().map(ShapeBuf::from_slice)),
        OutputRule::Preserve | OutputRule::ExplicitDType => Some(first()),
        OutputRule::ShapeAttributes => {
            if let Some(shape) = attributes.declared_shape() {
                Some(Some(ShapeBuf::from_slice(shape)))
            } else {
                transformed_shape(operation, attributes, inputs, output_index)?
            }
        }
        OutputRule::Broadcast => {
            if inputs.iter().any(|input| input.shape.is_none()) {
                Some(None)
            } else {
                // Folded over the inputs rather than collected into a
                // `Vec<&[usize]>` first: the borrowed slices were built only to
                // be walked once.
                Some(Some(broadcast_input_shapes(operation, inputs)?))
            }
        }
        OutputRule::MatMul => {
            match (
                inputs.first().and_then(|v| v.shape.as_deref()),
                inputs.get(1).and_then(|v| v.shape.as_deref()),
            ) {
                (Some(lhs), Some(rhs)) if lhs.len() >= 2 && rhs.len() >= 2 => {
                    if lhs[lhs.len() - 1] != rhs[rhs.len() - 2] {
                        return Err(invalid(
                            operation,
                            "shape",
                            "matmul contracting dimensions differ",
                        ));
                    }
                    let batch = broadcast_shape(
                        operation,
                        &[&lhs[..lhs.len() - 2], &rhs[..rhs.len() - 2]],
                    )?;
                    let mut shape = batch;
                    shape.push(lhs[lhs.len() - 2]);
                    shape.push(rhs[rhs.len() - 1]);
                    Some(Some(shape))
                }
                (None, _) | (_, None) => Some(None),
                _ => {
                    return Err(invalid(
                        operation,
                        "rank",
                        "matmul inputs require rank at least two",
                    ));
                }
            }
        }
        OutputRule::Reduction => {
            let Some(shape) = inputs.first().and_then(|input| input.shape.as_deref()) else {
                return Ok(Some(None));
            };
            if operation == OperationKind::TopK {
                return transformed_shape(operation, attributes, inputs, output_index);
            }
            if operation == OperationKind::Argsort {
                Some(Some(ShapeBuf::from_slice(shape)))
            } else if matches!(
                operation,
                OperationKind::MseLoss
                    | OperationKind::L1Loss
                    | OperationKind::BceWithLogitsLoss
                    | OperationKind::CrossEntropyLoss
            ) {
                match attributes.loss_reduction() {
                    Some(LossReduction::None) if operation == OperationKind::CrossEntropyLoss => {
                        Some(inputs.get(1).and_then(|input| input.shape.clone()))
                    }
                    Some(LossReduction::None) => Some(Some(ShapeBuf::from_slice(shape))),
                    Some(LossReduction::Mean | LossReduction::Sum) => {
                        Some(Some(ShapeBuf::scalar()))
                    }
                    None => None,
                }
            } else if matches!(
                operation,
                OperationKind::SumAll
                    | OperationKind::MeanAll
                    | OperationKind::MaxAll
                    | OperationKind::MinAll
                    | OperationKind::ProdAll
                    | OperationKind::Norm
                    | OperationKind::VarianceAll
                    | OperationKind::StdAll
            ) || matches!(operation, OperationKind::ArgMax | OperationKind::ArgMin)
                && attributes.axis().is_none()
            {
                Some(Some(ShapeBuf::scalar()))
            } else if let Some(axis) = attributes.axis() {
                if axis >= shape.len() {
                    return Err(invalid(
                        operation,
                        "axis",
                        "reduction axis is outside the input rank",
                    ));
                }
                let keep = matches!(
                    operation,
                    OperationKind::SumKeepDim
                        | OperationKind::MeanKeepDim
                        | OperationKind::MaxKeepDim
                        | OperationKind::MinKeepDim
                        | OperationKind::VarianceKeepDim
                        | OperationKind::StdKeepDim
                );
                let mut output = shape.to_vec();
                if keep {
                    output[axis] = 1;
                } else {
                    output.remove(axis);
                }
                Some(Some(ShapeBuf::from_slice(&output)))
            } else {
                None
            }
        }
        OutputRule::DataDependent | OutputRule::HostValue => Some(None),
        OutputRule::Indexing | OutputRule::TypedInference => match operation {
            OperationKind::Gather => Some(inputs.get(1).and_then(|input| input.shape.clone())),
            OperationKind::IndexSelect => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
                attributes.axis(),
            ) {
                (Some(source), Some(indices), Some(axis)) => {
                    let count =
                        crate::shapes::ShapeBuf::from_slice(indices).checked_numel(operation)?;
                    let mut output = source.to_vec();
                    output[axis] = count;
                    Some(Some(ShapeBuf::from_slice(&output)))
                }
                _ => Some(None),
            },
            OperationKind::EmbeddingExact => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
            ) {
                (Some(indices), Some(weight)) if weight.len() == 2 => {
                    let mut output = indices.to_vec();
                    output.push(weight[1]);
                    Some(Some(ShapeBuf::from_slice(&output)))
                }
                (None, _) | (_, None) => Some(None),
                _ => {
                    return Err(invalid(
                        operation,
                        "weight",
                        "embedding weight must have rank two",
                    ));
                }
            },
            OperationKind::Dot => Some(Some(ShapeBuf::scalar())),
            OperationKind::Outer => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
            ) {
                (Some(lhs), Some(rhs)) => Some(Some(ShapeBuf::from_slice(&[lhs[0], rhs[0]]))),
                _ => Some(None),
            },
            OperationKind::Addmm => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
                inputs.get(2).and_then(|input| input.shape.as_deref()),
            ) {
                (Some(addend), Some(lhs), Some(rhs)) if lhs.len() >= 2 && rhs.len() >= 2 => {
                    if lhs[lhs.len() - 1] != rhs[rhs.len() - 2] {
                        return Err(invalid(
                            operation,
                            "shape",
                            "addmm contracting dimensions differ",
                        ));
                    }
                    let mut product = broadcast_shape(
                        operation,
                        &[&lhs[..lhs.len() - 2], &rhs[..rhs.len() - 2]],
                    )?;
                    product.push(lhs[lhs.len() - 2]);
                    product.push(rhs[rhs.len() - 1]);
                    Some(Some(broadcast_shape(operation, &[addend, &product])?))
                }
                (None, _, _) | (_, None, _) | (_, _, None) => Some(None),
                _ => {
                    return Err(invalid(
                        operation,
                        "rank",
                        "addmm matrix operands require rank at least two",
                    ));
                }
            },
            OperationKind::ScaledDotProductAttention => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
                inputs.get(2).and_then(|input| input.shape.as_deref()),
            ) {
                (Some(query), Some(key), Some(value))
                    if query.len() >= 2 && key.len() >= 2 && value.len() >= 2 =>
                {
                    if query[query.len() - 1] != key[key.len() - 1]
                        || key[key.len() - 2] != value[value.len() - 2]
                    {
                        return Err(invalid(
                            operation,
                            "shape",
                            "attention query/key width or key/value sequence extents differ",
                        ));
                    }
                    let mut output = broadcast_shape(
                        operation,
                        &[
                            &query[..query.len() - 2],
                            &key[..key.len() - 2],
                            &value[..value.len() - 2],
                        ],
                    )?;
                    output.push(query[query.len() - 2]);
                    output.push(value[value.len() - 1]);
                    Some(Some(output))
                }
                (None, _, _) | (_, None, _) | (_, _, None) => Some(None),
                _ => {
                    return Err(invalid(
                        operation,
                        "rank",
                        "attention operands require rank at least two",
                    ));
                }
            },
            OperationKind::Linear => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
            ) {
                (Some(input), Some(weight)) if !input.is_empty() && weight.len() == 2 => {
                    if input[input.len() - 1] != weight[1]
                        || inputs
                            .get(2)
                            .and_then(|value| value.shape.as_deref())
                            .is_some_and(|bias| bias != [weight[0]])
                    {
                        return Err(invalid(
                            operation,
                            "shape",
                            "linear input width, weight, and optional bias disagree",
                        ));
                    }
                    let mut output = input.to_vec();
                    let last = output.len() - 1;
                    output[last] = weight[0];
                    Some(Some(ShapeBuf::from_slice(&output)))
                }
                (None, _) | (_, None) => Some(None),
                _ => {
                    return Err(invalid(
                        operation,
                        "rank",
                        "linear requires non-scalar input and rank-two weight",
                    ));
                }
            },
            OperationKind::SgdStep => Some(inputs.first().and_then(|v| v.shape.clone())),
            OperationKind::AdamStep | OperationKind::AdamWStep => {
                let source = match output_index {
                    0 => 0,
                    1 => 2,
                    _ => 3,
                };
                Some(inputs.get(source).and_then(|v| v.shape.clone()))
            }
            OperationKind::Quantize
            | OperationKind::Dequantize
            | OperationKind::QuantizedMatMul => Some(first()),
            _ => return Err(DescriptorError::MissingInference { operation }),
        },
    };
    Ok(inferred)
}

pub(super) fn verify_outputs<A: AttributeContract>(
    operation: OperationKind,
    row: &OperationCatalogEntry,
    attributes: &A,
    inputs: &[LogicalTensorMeta],
    outputs: &[LogicalTensorMeta],
    provenance: OutputProvenance,
) -> Result<(), DescriptorError> {
    let first_dtype = inputs.first().and_then(|input| input.dtype);
    let is_float = |dtype: DTypeDescriptor| dtype.is_float();
    let is_integer = |dtype: DTypeDescriptor| dtype.is_integer();

    if matches!(
        row.profile,
        SemanticProfile::BinaryBroadcast
            | SemanticProfile::Comparison
            | SemanticProfile::Logical
            | SemanticProfile::Mutation
            | SemanticProfile::MatMul
    ) {
        for input in inputs.iter().skip(1) {
            if let (Some(expected), Some(actual)) = (first_dtype, input.dtype) {
                if expected != actual {
                    return Err(invalid(
                        operation,
                        "dtype",
                        "operation inputs require the same dtype",
                    ));
                }
            }
        }
    }

    let index_input = match operation {
        OperationKind::Gather | OperationKind::Scatter | OperationKind::IndexSelect => Some(1),
        OperationKind::EmbeddingExact => Some(0),
        OperationKind::CrossEntropyLoss => Some(1),
        _ => None,
    };
    let require_float = matches!(
        row.profile,
        SemanticProfile::UnaryFloat
            | SemanticProfile::MatMul
            | SemanticProfile::Attention
            | SemanticProfile::Reduction
            | SemanticProfile::Normalization
            | SemanticProfile::Loss
            | SemanticProfile::Optimizer
    ) || matches!(
        row.profile,
        SemanticProfile::Module | SemanticProfile::Composite
    ) && operation != OperationKind::EmbeddingExact;
    if require_float {
        for (index, input) in inputs.iter().enumerate() {
            if Some(index) == index_input {
                continue;
            }
            if input.dtype.is_some_and(|dtype| !is_float(dtype)) {
                return Err(invalid(
                    operation,
                    "dtype",
                    "operation requires floating-point input metadata",
                ));
            }
        }
    }
    if let Some(index) = index_input {
        if inputs
            .get(index)
            .and_then(|input| input.dtype)
            .is_some_and(|dtype| !is_integer(dtype))
        {
            return Err(invalid(
                operation,
                "index dtype",
                "index metadata requires an integer dtype",
            ));
        }
    }
    let same_dtype_pair = match operation {
        OperationKind::WhereCond => Some((1, 2)),
        OperationKind::Scatter => Some((0, 2)),
        _ => None,
    };
    if let Some((left, right)) = same_dtype_pair {
        if let (Some(expected), Some(actual)) = (
            inputs.get(left).and_then(|input| input.dtype),
            inputs.get(right).and_then(|input| input.dtype),
        ) {
            if expected != actual {
                return Err(invalid(
                    operation,
                    "dtype",
                    "value operands require the same dtype",
                ));
            }
        }
    }
    if operation == OperationKind::EmbeddingExact
        && inputs
            .get(1)
            .and_then(|input| input.dtype)
            .is_some_and(|dtype| !is_float(dtype))
    {
        return Err(invalid(
            operation,
            "weight dtype",
            "embedding weight metadata requires a floating dtype",
        ));
    }
    match operation {
        OperationKind::Quantize => {
            if first_dtype.is_some_and(|dtype| !is_float(dtype))
                || attributes.declared_dtype() != Some(DTypeId::Q8_0.descriptor())
            {
                return Err(invalid(
                    operation,
                    "dtype",
                    "quantize requires floating input and q8_0 output metadata",
                ));
            }
        }
        OperationKind::Dequantize => {
            if first_dtype.is_some_and(|dtype| dtype != DTypeId::Q8_0.descriptor())
                || attributes
                    .declared_dtype()
                    .is_some_and(|dtype| !is_float(dtype))
            {
                return Err(invalid(
                    operation,
                    "dtype",
                    "dequantize requires q8_0 input and floating output metadata",
                ));
            }
        }
        OperationKind::QuantizedMatMul => {
            if inputs
                .iter()
                .filter_map(|input| input.dtype)
                .any(|dtype| dtype != DTypeId::Q8_0.descriptor())
            {
                return Err(invalid(
                    operation,
                    "dtype",
                    "quantized matmul requires q8_0 inputs",
                ));
            }
        }
        OperationKind::LogicalAnd | OperationKind::LogicalOr | OperationKind::LogicalNot => {
            if inputs
                .iter()
                .filter_map(|input| input.dtype)
                .any(|dtype| !dtype.is_bool())
            {
                return Err(invalid(
                    operation,
                    "dtype",
                    "logical operations require boolean inputs",
                ));
            }
        }
        OperationKind::WhereCond => {
            if inputs
                .first()
                .and_then(|input| input.dtype)
                .is_some_and(|dtype| !dtype.is_bool())
            {
                return Err(invalid(
                    operation,
                    "mask dtype",
                    "where_cond requires a boolean mask input",
                ));
            }
        }
        OperationKind::MaskedFill => {
            if inputs
                .get(1)
                .and_then(|input| input.dtype)
                .is_some_and(|dtype| !dtype.is_bool())
            {
                return Err(invalid(
                    operation,
                    "mask dtype",
                    "masked_fill requires a boolean mask input",
                ));
            }
        }
        _ => {}
    }

    for (index, output) in outputs.iter().enumerate() {
        if provenance == OutputProvenance::Derived {
            // `infer_outputs` already refused a missing inference over known
            // inputs, and produced device, dtype and shape from this very
            // function, so only the shape well-formedness check below is not
            // already implied.
            if let Some(shape) = &output.shape {
                validate_shape(operation, shape)?;
            }
            continue;
        }
        let expected = expected_output(operation, row, attributes, inputs, index)?;
        if output.device != expected.device {
            return Err(DescriptorError::MetadataMismatch {
                operation,
                output: index,
                field: "device",
            });
        }
        if output.dtype != expected.dtype {
            return Err(DescriptorError::MetadataMismatch {
                operation,
                output: index,
                field: "dtype",
            });
        }

        match expected.shape {
            Some(expected_shape) => {
                if output.shape.as_deref() != expected_shape.as_deref() {
                    return Err(DescriptorError::MetadataMismatch {
                        operation,
                        output: index,
                        field: "shape",
                    });
                }
            }
            // No inference branch produced an expectation. Accepting the
            // caller's shape here would let fully known inputs certify an
            // output that nothing checked, which is exactly the fabrication
            // this contract exists to prevent. Unknown input metadata
            // legitimately yields no expectation and stays unknown; known
            // inputs must fail closed instead.
            None => {
                if inputs_are_known(inputs) {
                    return Err(DescriptorError::MissingInference { operation });
                }
            }
        }
        if let Some(shape) = &output.shape {
            validate_shape(operation, shape)?;
        }
    }
    Ok(())
}

fn inputs_are_known(inputs: &[LogicalTensorMeta]) -> bool {
    !inputs.is_empty() && inputs.iter().all(|input| input.shape.is_some())
}

/// The output metadata the contract requires at `index`.
///
/// `shape` is `None` when no inference branch applies at all, and
/// `Some(None)` when a branch applies but the inputs it reads are unknown.
/// Verification and inference both read this one function, so an inferred
/// output can never disagree with a verified one.
/// Whether the outputs handed to validation were derived by [`infer_outputs`]
/// in the same call, or supplied from outside.
///
/// The distinction is only about *re-derivation*. `verify_outputs` compares
/// supplied outputs against `expected_output`, which is the check that stops a
/// caller fabricating output metadata. When the outputs came from
/// `infer_outputs` moments earlier, that comparison re-runs the same function
/// over the same inputs and can only ever agree - it was costing a full second
/// shape inference, and its allocations, on every operation the framework
/// executes. Every other check in `verify_outputs` runs either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OutputProvenance {
    /// Produced by `infer_outputs` in this call. Re-deriving is a tautology.
    Derived,
    /// Stated from outside. The expectation comparison is load-bearing. Only
    /// the test suite can produce this today; see `validate`.
    #[cfg_attr(not(test), allow(dead_code))]
    Supplied,
}

struct ExpectedOutput {
    device: Option<DeviceId>,
    dtype: Option<DTypeDescriptor>,
    shape: Option<Option<ShapeBuf>>,
}

fn expected_output<A: AttributeContract>(
    operation: OperationKind,
    row: &OperationCatalogEntry,
    attributes: &A,
    inputs: &[LogicalTensorMeta],
    index: usize,
) -> Result<ExpectedOutput, DescriptorError> {
    let first_dtype = inputs.first().and_then(|input| input.dtype);
    let first_device = inputs.first().and_then(|input| input.device);
    let dtype = match operation {
        OperationKind::WhereCond | OperationKind::EmbeddingExact => {
            inputs.get(1).and_then(|input| input.dtype)
        }
        OperationKind::TopK if index == 0 => first_dtype,
        OperationKind::ArgMax | OperationKind::ArgMin | OperationKind::Argsort => {
            attributes.declared_dtype()
        }
        OperationKind::TopK => attributes.declared_dtype(),
        OperationKind::QuantizedMatMul => Some(DTypeId::F32.descriptor()),
        OperationKind::CmpEq
        | OperationKind::CmpNe
        | OperationKind::CmpLt
        | OperationKind::CmpLe
        | OperationKind::CmpGt
        | OperationKind::CmpGe
        | OperationKind::LogicalAnd
        | OperationKind::LogicalOr
        | OperationKind::LogicalNot => Some(DTypeId::Bool.descriptor()),
        _ => attributes.declared_dtype().or(first_dtype),
    };
    Ok(ExpectedOutput {
        device: attributes.declared_device().or(first_device),
        dtype,
        shape: inferred_shape(operation, row, attributes, inputs, index)?,
    })
}

/// Derive the outputs an invocation must produce, instead of trusting a caller
/// to state them.
///
/// This is the entry point execution uses. A caller that never supplies output
/// metadata cannot fabricate it, so the "no output is invented" contract holds
/// by construction rather than by a comparison the caller could satisfy with a
/// lucky guess.
pub(super) fn infer_outputs<A: AttributeContract>(
    operation: OperationKind,
    row: &OperationCatalogEntry,
    attributes: &A,
    inputs: &[LogicalTensorMeta],
) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
    let count = attributes
        .expected_output_count(inputs)
        .unwrap_or(*row.output_arity.start());
    let mut outputs = Vec::with_capacity(count);
    for index in 0..count {
        let expected = expected_output(operation, row, attributes, inputs, index)?;
        let shape = match expected.shape {
            Some(shape) => shape,
            None if inputs_are_known(inputs) => {
                return Err(DescriptorError::MissingInference { operation });
            }
            None => None,
        };
        outputs.push(LogicalTensorMeta {
            // Already a `ShapeBuf`: inference builds one directly, so an
            // ordinary rank never touches the heap on the way here.
            shape,
            dtype: expected.dtype,
            device: expected.device,
        });
    }
    Ok(outputs)
}

/// The exact rank contract for one operand role.
///
/// `OperationCatalogEntry::accepted_ranks` describes the *primary* operand, the
/// activation, the value being reduced, the tensor being reshaped. Applying it
/// to every input is wrong for any operation whose operands carry different
/// contracts: a rank-one convolution bias is not a rank-four activation, and an
/// embedding table is not the index batch that addresses it.
///
/// A role listed here overrides the primary window for that position only, so
/// widening an activation's accepted ranks can never silently widen a
/// parameter's. Roles absent from this table keep the primary window, which is
/// the correct contract for genuinely homogeneous operands (broadcast
/// arithmetic, elementwise losses, comparisons).
pub(super) fn operand_ranks(
    operation: OperationKind,
    row: &OperationCatalogEntry,
    index: usize,
) -> core::ops::RangeInclusive<usize> {
    use OperationKind::*;
    let exact = |rank: usize| Some(rank..=rank);
    let override_range = match (operation, index) {
        // Convolution: the filter bank is fixed at `[c_out, c_in / groups,
        // ..spatial]`, and a bias holds one value per output channel.
        (Conv1dExact, 1) => exact(3),
        (Conv2dExact | ConvTranspose2d, 1) => exact(4),
        (Conv1dExact | Conv2dExact | ConvTranspose2d, 2) => exact(1),
        // The embedding table is always `[num_embeddings, dim]`; the indices
        // carry whatever batch geometry the caller addresses it with.
        (EmbeddingExact, 0) => Some(1..=usize::MAX),
        (EmbeddingExact, 1) => exact(2),
        // Batch-norm affine parameters and running state are per-channel
        // vectors, never activations. `BatchNormAttributes::validate` already
        // pins their extent; this pins the rank even when the extent is unknown.
        (BatchNorm, 1..=4) => exact(1),
        (RmsNorm, 1) => exact(1),
        // `Linear` weights are `[out, in]` and biases `[out]`.
        (Linear, 1) => exact(2),
        (Linear, 2) => exact(1),
        // Cross entropy consumes `[batch, classes]` logits against `[batch]`
        // class indices; the two operands are not interchangeable.
        (CrossEntropyLoss, 0) => exact(2),
        (CrossEntropyLoss, 1) => exact(1),
        // `index_select` addresses one axis with a flat index vector.
        (IndexSelect, 1) => exact(1),
        // Optimizer gradients and moment state mirror the parameter they
        // update, which the primary window already covers; the equal-shape
        // requirement is enforced separately in `validate`.
        _ => None,
    };
    override_range.unwrap_or_else(|| row.accepted_ranks.clone())
}
