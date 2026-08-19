use super::*;

impl AttributeContract for CreationAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        Some(self.dtype)
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for DataAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)?;
        match (operation, &self.payload) {
            (OperationKind::TensorFromData, CreationPayload::Typed { dtype, .. }) => {
                if *dtype != self.dtype {
                    return Err(DescriptorError::PayloadDTypeMismatch {
                        operation,
                        expected: self.dtype,
                        actual: *dtype,
                    });
                }
            }
            (OperationKind::TensorFromBytes, CreationPayload::Bytes { .. }) => {}
            (OperationKind::TensorFromData, _) => {
                return Err(DescriptorError::PayloadKind {
                    operation,
                    expected: "typed",
                });
            }
            (OperationKind::TensorFromBytes, _) => {
                return Err(DescriptorError::PayloadKind {
                    operation,
                    expected: "byte",
                });
            }
            _ => {}
        }
        let expected = self
            .dtype
            .size_bytes(
                ShapeBuf::from_slice(&self.shape).checked_numel(operation)?,
                operation,
            )
            .map_err(DescriptorError::Shape)?;
        let actual = self.payload.byte_len();
        if actual != expected {
            return Err(DescriptorError::PayloadByteLength {
                operation,
                expected,
                actual,
            });
        }
        Ok(())
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        Some(self.dtype)
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for FullAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        Some(self.dtype)
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for ArangeAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)?;
        if !self.start.is_finite() || !self.step.is_finite() || self.step == 0.0 {
            return Err(invalid(
                operation,
                "step",
                "arange requires finite start and non-zero finite step",
            ));
        }
        Ok(())
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        Some(self.dtype)
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for LinspaceAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)?;
        if !self.start.is_finite() || !self.end.is_finite() {
            return Err(invalid(
                operation,
                "bounds",
                "linspace bounds must be finite",
            ));
        }
        Ok(())
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        Some(self.dtype)
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for DistributionAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)?;
        if self.distribution.is_empty() {
            return Err(invalid(
                operation,
                "distribution",
                "distribution identity must not be empty",
            ));
        }
        Ok(())
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        Some(self.dtype)
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for ClampAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.min.is_nan() || self.max.is_nan() || self.min > self.max {
            return Err(invalid(
                operation,
                "min/max",
                "clamp requires ordered non-NaN bounds",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for ShapeAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)?;
        if let Some(input) = first_shape(inputs) {
            if operation == OperationKind::ReshapeExact {
                let source = crate::shapes::ShapeBuf::from_slice(input).checked_numel(operation)?;
                let target =
                    crate::shapes::ShapeBuf::from_slice(&self.shape).checked_numel(operation)?;
                if source != target {
                    return Err(invalid(
                        operation,
                        "shape",
                        "reshape must preserve the element count",
                    ));
                }
            } else if matches!(
                operation,
                OperationKind::BroadcastAs | OperationKind::BroadcastLeft
            ) {
                if input.len() > self.shape.len()
                    || input
                        .iter()
                        .rev()
                        .zip(self.shape.iter().rev())
                        .any(|(&source, &target)| source != target && source != 1)
                {
                    return Err(invalid(
                        operation,
                        "shape",
                        "source shape cannot broadcast to the explicit target",
                    ));
                }
            }
        }
        Ok(())
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
}
impl AttributeContract for RepeatAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            if shape.len() != self.repeats.len() {
                return Err(invalid(
                    operation,
                    "repeats",
                    "repeat count rank must equal input rank",
                ));
            }
            let mut output = Vec::with_capacity(shape.len());
            for (&dim, &repeat) in shape.iter().zip(&self.repeats) {
                output.push(dim.checked_mul(repeat).ok_or_else(|| {
                    invalid(
                        operation,
                        "repeats",
                        "repeated output dimension overflows usize",
                    )
                })?);
            }
            validate_shape(operation, &output)?;
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Repeat(&self.repeats))
    }
}
impl AttributeContract for AxisAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            let insertion = matches!(
                operation,
                OperationKind::UnsqueezeExact | OperationKind::StackExact
            );
            let limit = shape.len() + usize::from(insertion);
            if self.axis >= limit {
                return Err(invalid(
                    operation,
                    "axis",
                    "axis is outside the accepted rank",
                ));
            }
        }
        Ok(())
    }
    fn axis(&self) -> Option<usize> {
        Some(self.axis)
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Axis(self.axis))
    }
}
impl AttributeContract for IndexReductionAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_index_dtype(operation, "dtype", self.dtype)?;
        if let (Some(axis), Some(shape)) = (self.axis, first_shape(inputs)) {
            if axis >= shape.len() {
                return Err(invalid(operation, "axis", "axis is outside the input rank"));
            }
        }
        Ok(())
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        Some(self.dtype)
    }
    fn axis(&self) -> Option<usize> {
        self.axis
    }
}
impl AttributeContract for TransposeAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            if self.first >= shape.len() || self.second >= shape.len() {
                return Err(invalid(
                    operation,
                    "axis",
                    "transpose axis is outside the input rank",
                ));
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Transpose(self.first, self.second))
    }
}
impl AttributeContract for NarrowAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            let Some(&extent) = shape.get(self.axis) else {
                return Err(invalid(
                    operation,
                    "axis",
                    "narrow axis is outside the input rank",
                ));
            };
            let end = self.start.checked_add(self.length).ok_or_else(|| {
                invalid(operation, "start/length", "narrow endpoint overflows usize")
            })?;
            if end > extent {
                return Err(invalid(
                    operation,
                    "start/length",
                    "narrow range exceeds the input extent",
                ));
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Narrow {
            axis: self.axis,
            length: self.length,
        })
    }
}
impl AttributeContract for SliceAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            if shape.len() != self.ranges.len() {
                return Err(invalid(
                    operation,
                    "ranges",
                    "slice range count must equal input rank",
                ));
            }
            for ((start, end), extent) in self.ranges.iter().zip(shape) {
                if start > end || end > extent {
                    return Err(invalid(
                        operation,
                        "ranges",
                        "slice range must be ordered and within its extent",
                    ));
                }
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Slice(&self.ranges))
    }
}
impl AttributeContract for FlattenAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.start_axis > self.end_axis {
            return Err(invalid(
                operation,
                "axis range",
                "flatten start must not exceed end",
            ));
        }
        if let Some(shape) = first_shape(inputs) {
            if self.end_axis >= shape.len() {
                return Err(invalid(
                    operation,
                    "axis range",
                    "flatten end is outside the input rank",
                ));
            }
            validate_shape(operation, shape)?;
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Flatten {
            start: self.start_axis,
            end: self.end_axis,
        })
    }
}
impl AttributeContract for ScatterAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        AxisAttributes { axis: self.axis }.validate(operation, inputs)
    }
    fn axis(&self) -> Option<usize> {
        Some(self.axis)
    }
}
impl AttributeContract for PadAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            if shape.len() != self.padding.len() {
                return Err(invalid(
                    operation,
                    "padding",
                    "padding rank must equal input rank",
                ));
            }
            let mut output = Vec::with_capacity(shape.len());
            for (&dim, &(before, after)) in shape.iter().zip(&self.padding) {
                output.push(
                    dim.checked_add(before)
                        .and_then(|v| v.checked_add(after))
                        .ok_or_else(|| {
                            invalid(operation, "padding", "padded dimension overflows usize")
                        })?,
                );
            }
            validate_shape(operation, &output)?;
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Pad(&self.padding))
    }
}
impl AttributeContract for DiagonalAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            if !(1..=2).contains(&shape.len()) {
                return Err(invalid(
                    operation,
                    "rank",
                    "diagonal operations require rank one or two",
                ));
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Diagonal(self.offset))
    }
}
impl AttributeContract for ChunkAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.chunks == 0 {
            return Err(invalid(operation, "chunks", "chunk count must be non-zero"));
        }
        AxisAttributes { axis: self.axis }.validate(operation, inputs)
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Chunk {
            chunks: self.chunks,
            axis: self.axis,
        })
    }
    fn expected_output_count(&self, inputs: &[LogicalTensorMeta]) -> Option<usize> {
        let extent = first_shape(inputs)?.get(self.axis).copied()?;
        Some(self.chunks.min(extent))
    }
}
impl AttributeContract for SplitAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.split_size == 0 {
            return Err(invalid(
                operation,
                "split_size",
                "split size must be non-zero",
            ));
        }
        AxisAttributes { axis: self.axis }.validate(operation, inputs)
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Split {
            split_size: self.split_size,
            axis: self.axis,
        })
    }
    fn expected_output_count(&self, inputs: &[LogicalTensorMeta]) -> Option<usize> {
        let extent = first_shape(inputs)?.get(self.axis).copied()?;
        Some(extent.div_ceil(self.split_size))
    }
}
impl AttributeContract for AddmmAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if !self.alpha.is_finite() || !self.beta.is_finite() {
            return Err(invalid(
                operation,
                "alpha/beta",
                "addmm scaling factors must be finite",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for AttentionAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self
            .scale
            .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
        {
            return Err(invalid(
                operation,
                "scale",
                "attention scale must be positive and finite",
            ));
        }
        let expected = if self.has_mask { 4 } else { 3 };
        if inputs.len() != expected {
            return Err(invalid(
                operation,
                "has_mask",
                "attention input arity disagrees with the mask attribute",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for UnfoldAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.size == 0 || self.step == 0 {
            return Err(invalid(
                operation,
                "size/step",
                "unfold size and step must be non-zero",
            ));
        }
        AxisAttributes { axis: self.axis }.validate(operation, inputs)
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Unfold {
            axis: self.axis,
            size: self.size,
            step: self.step,
        })
    }
}
impl AttributeContract for PixelShuffleAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.upscale_factor == 0 {
            return Err(invalid(
                operation,
                "upscale_factor",
                "pixel shuffle factor must be non-zero",
            ));
        }
        if let Some(shape) = first_shape(inputs) {
            if shape.len() != 4 {
                return Err(invalid(
                    operation,
                    "rank",
                    "pixel shuffle requires rank four",
                ));
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::PixelShuffle(self.upscale_factor))
    }
}
impl AttributeContract for GroupNormAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.groups == 0 {
            return Err(invalid(operation, "groups", "group count must be non-zero"));
        }
        validate_epsilon(operation, self.epsilon)?;
        if let Some(shape) = first_shape(inputs) {
            if shape.len() < 2 || shape[1] % self.groups != 0 {
                return Err(invalid(
                    operation,
                    "groups",
                    "group norm requires a channel axis divisible by the group count",
                ));
            }
        }
        Ok(())
    }
}
fn validate_epsilon(operation: OperationKind, epsilon: f64) -> Result<(), DescriptorError> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(invalid(
            operation,
            "epsilon",
            "epsilon must be finite and non-negative",
        ));
    }
    Ok(())
}
impl AttributeContract for EpsilonAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_epsilon(operation, self.epsilon)?;
        match operation {
            OperationKind::InstanceNorm => {
                if first_shape(inputs).is_some_and(|shape| shape.len() != 4) {
                    return Err(invalid(
                        operation,
                        "rank",
                        "instance norm requires [batch, channels, height, width]",
                    ));
                }
            }
            OperationKind::RmsNorm => {
                if inputs.len() != 2 {
                    return Err(DescriptorError::Arity {
                        operation,
                        expected: 2..=2,
                        actual: inputs.len(),
                    });
                }
                if let (Some(input), Some(weight)) = (
                    first_shape(inputs),
                    inputs.get(1).and_then(|value| value.shape.as_deref()),
                ) {
                    if input.last() != weight.last() || weight.len() != 1 {
                        return Err(invalid(
                            operation,
                            "weight shape",
                            "RMS norm weight must match the final input extent",
                        ));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}
impl AttributeContract for LayerNormAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.normalized_shape)?;
        validate_epsilon(operation, self.epsilon)?;
        let expected = if self.has_bias { 3 } else { 2 };
        if inputs.len() != expected {
            return Err(invalid(
                operation,
                "has_bias",
                "layer norm input arity disagrees with the bias attribute",
            ));
        }
        if let Some(input) = first_shape(inputs) {
            if self.normalized_shape.len() > input.len()
                || input[input.len() - self.normalized_shape.len()..] != self.normalized_shape
                || inputs.get(1).and_then(|value| value.shape.as_deref())
                    != Some(self.normalized_shape.as_slice())
                || self.has_bias
                    && inputs.get(2).and_then(|value| value.shape.as_deref())
                        != Some(self.normalized_shape.as_slice())
            {
                return Err(invalid(
                    operation,
                    "normalized shape",
                    "layer norm input suffix, weight, and optional bias must match",
                ));
            }
        }
        Ok(())
    }
}
impl AttributeContract for BatchNormAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_epsilon(operation, self.epsilon)?;
        if !self.momentum.is_finite() || !(0.0..=1.0).contains(&self.momentum) {
            return Err(invalid(
                operation,
                "momentum",
                "batch norm momentum must be in [0, 1]",
            ));
        }
        let expected = 1
            + usize::from(self.has_weight)
            + usize::from(self.has_bias)
            + usize::from(self.has_running_mean)
            + usize::from(self.has_running_variance);
        if inputs.len() != expected
            || self.has_running_mean != self.has_running_variance
            || !self.training && !self.has_running_mean
        {
            return Err(invalid(
                operation,
                "optional state",
                "batch norm attributes and input arity/state are inconsistent",
            ));
        }
        if let Some(input) = first_shape(inputs) {
            if input.len() < 2 {
                return Err(invalid(
                    operation,
                    "input shape",
                    "batch norm requires a channel axis",
                ));
            }
            let channels = input[1];
            for parameter in inputs.iter().skip(1) {
                if parameter.shape.as_deref() != Some(&[channels]) {
                    return Err(invalid(
                        operation,
                        "parameter shape",
                        "batch norm affine/state tensors must match the channel extent",
                    ));
                }
            }
        }
        Ok(())
    }
}

macro_rules! spatial_contract {
    ($ty:ty, $body:expr, $transform:ident, bias) => {
        impl AttributeContract for $ty {
            fn validate(
                &self,
                operation: OperationKind,
                _: &[LogicalTensorMeta],
            ) -> Result<(), DescriptorError> {
                if $body(self) {
                    Ok(())
                } else {
                    Err(invalid(
                        operation,
                        "spatial parameters",
                        "kernel, stride, dilation, and groups where present must be non-zero",
                    ))
                }
            }
            fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
                Some(ShapeTransform::$transform(self))
            }
            fn optional_bias(&self) -> Option<bool> {
                Some(self.has_bias)
            }
        }
    };
    ($ty:ty, $body:expr, $transform:ident) => {
        impl AttributeContract for $ty {
            fn validate(
                &self,
                operation: OperationKind,
                _: &[LogicalTensorMeta],
            ) -> Result<(), DescriptorError> {
                if $body(self) {
                    Ok(())
                } else {
                    Err(invalid(
                        operation,
                        "spatial parameters",
                        "kernel, stride, dilation, and groups where present must be non-zero",
                    ))
                }
            }
            fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
                Some(ShapeTransform::$transform(self))
            }
        }
    };
}
spatial_contract!(
    Conv1dAttributes,
    |a: &Conv1dAttributes| a.stride > 0 && a.dilation > 0 && a.groups > 0,
    Conv1d,
    bias
);
spatial_contract!(
    Conv2dAttributes,
    |a: &Conv2dAttributes| a.stride.iter().all(|&v| v > 0)
        && a.dilation.iter().all(|&v| v > 0)
        && a.groups > 0,
    Conv2d,
    bias
);
spatial_contract!(
    ConvTranspose2dAttributes,
    |a: &ConvTranspose2dAttributes| a.stride.iter().all(|&v| v > 0)
        && a.dilation.iter().all(|&v| v > 0)
        && a.groups > 0,
    ConvTranspose2d,
    bias
);
spatial_contract!(
    Pool2dAttributes,
    |a: &Pool2dAttributes| a.kernel.iter().all(|&v| v > 0)
        && a.stride.iter().all(|&v| v > 0)
        && a.dilation.iter().all(|&v| v > 0),
    Pool2d
);
spatial_contract!(
    AvgPool2dAttributes,
    |a: &AvgPool2dAttributes| a.kernel.iter().all(|&v| v > 0) && a.stride.iter().all(|&v| v > 0),
    AvgPool2d
);
impl AttributeContract for AdaptivePool2dAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.output.iter().all(|&extent| extent > 0) {
            Ok(())
        } else {
            Err(invalid(
                operation,
                "output",
                "adaptive pooling output extents must be non-zero",
            ))
        }
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::AdaptivePool2d(self.output))
    }
}

impl AttributeContract for TopKAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        AxisAttributes { axis: self.axis }.validate(operation, inputs)?;
        validate_index_dtype(operation, "index_dtype", self.index_dtype)?;
        if self.k == 0 {
            return Err(invalid(
                operation,
                "k",
                "top-k requires k greater than zero",
            ));
        }
        if let Some(shape) = first_shape(inputs) {
            if self.k > shape[self.axis] {
                return Err(invalid(operation, "k", "top-k exceeds the selected extent"));
            }
        }
        Ok(())
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        Some(self.index_dtype)
    }
    fn axis(&self) -> Option<usize> {
        Some(self.axis)
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::TopK {
            axis: self.axis,
            k: self.k,
        })
    }
}
impl AttributeContract for ArgsortAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        AxisAttributes { axis: self.axis }.validate(operation, inputs)?;
        validate_index_dtype(operation, "index_dtype", self.index_dtype)
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        Some(self.index_dtype)
    }
    fn axis(&self) -> Option<usize> {
        Some(self.axis)
    }
}
impl AttributeContract for NormAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if !self.order.is_finite() || self.order <= 0.0 {
            return Err(invalid(
                operation,
                "order",
                "norm order must be positive and finite",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for VarianceAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        // `var_all`/`std_all` reduce the whole tensor, so the corrected
        // denominator is the element count.
        validate_unbiased_domain(
            operation,
            self.unbiased,
            first_shape(inputs).map(|shape| shape.iter().product()),
        )
    }
}
impl AttributeContract for AxisVarianceAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        AxisAttributes { axis: self.axis }.validate(operation, inputs)?;
        validate_unbiased_domain(
            operation,
            self.unbiased,
            first_shape(inputs).and_then(|shape| shape.get(self.axis).copied()),
        )
    }
    // Output inference reads the axis through this accessor, not through the
    // field. Without it these attributes validate an axis and then decline to
    // say what it was, so `var_dim`, `var_keepdim`, `std_dim` and `std_keepdim`
    // fell to the fail-closed arm and reported `MissingInference` for every
    // invocation, which made them undispatchable from the day they were
    // declared. `the_axis_variance_operations_infer_their_output_shape` is the
    // regression test.
    fn axis(&self) -> Option<usize> {
        Some(self.axis)
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Axis(self.axis))
    }
}
impl AttributeContract for DropoutAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if !self.probability.is_finite() || !(0.0..1.0).contains(&self.probability) {
            return Err(invalid(
                operation,
                "probability",
                "dropout probability must be in [0, 1)",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for LinearAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        let expected = if self.has_bias { 3 } else { 2 };
        if inputs.len() != expected {
            return Err(invalid(
                operation,
                "has_bias",
                "linear input arity disagrees with the bias attribute",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for RecurrentAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.input_size == 0 || self.hidden_size == 0 {
            return Err(invalid(
                operation,
                "feature size",
                "recurrent input and hidden sizes must be non-zero",
            ));
        }
        if let Some(sequence) = first_shape(inputs) {
            if sequence.len() != 3 || sequence[2] != self.input_size {
                return Err(invalid(
                    operation,
                    "input shape",
                    "recurrent sequence must be [batch, sequence, input_size]",
                ));
            }
            for state in inputs.iter().skip(1) {
                if state.shape.as_deref() != Some(&[sequence[0], self.hidden_size]) {
                    return Err(invalid(
                        operation,
                        "state shape",
                        "recurrent states must be [batch, hidden_size]",
                    ));
                }
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Rnn(self))
    }
}
impl AttributeContract for SgdAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_learning_rate(operation, self.learning_rate)
    }
}
fn validate_learning_rate(operation: OperationKind, rate: f64) -> Result<(), DescriptorError> {
    if !rate.is_finite() || rate < 0.0 {
        return Err(invalid(
            operation,
            "learning_rate",
            "learning rate must be finite and non-negative",
        ));
    }
    Ok(())
}
impl AttributeContract for AdamAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_learning_rate(operation, self.learning_rate)?;
        validate_epsilon(operation, self.epsilon)?;
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) || self.step == 0
        {
            return Err(invalid(
                operation,
                "beta/step",
                "Adam betas must be in [0, 1) and step must be non-zero",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for AdamWAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        AdamAttributes {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            step: self.step,
        }
        .validate(operation, inputs)?;
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(invalid(
                operation,
                "weight_decay",
                "weight decay must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

impl AttributeContract for DTypeAttributes {
    fn validate(&self, _: OperationKind, _: &[LogicalTensorMeta]) -> Result<(), DescriptorError> {
        Ok(())
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        Some(self.dtype)
    }
}
impl AttributeContract for DeviceAttributes {
    fn validate(&self, _: OperationKind, _: &[LogicalTensorMeta]) -> Result<(), DescriptorError> {
        Ok(())
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for QuantizationAttributes {
    fn validate(&self, _: OperationKind, _: &[LogicalTensorMeta]) -> Result<(), DescriptorError> {
        Ok(())
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        Some(self.dtype)
    }
}
impl AttributeContract for LossAttributes {
    fn validate(&self, _: OperationKind, _: &[LogicalTensorMeta]) -> Result<(), DescriptorError> {
        Ok(())
    }
    fn loss_reduction(&self) -> Option<LossReduction> {
        Some(self.reduction)
    }
}
