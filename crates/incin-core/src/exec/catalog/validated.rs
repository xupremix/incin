use super::*;

/// Opaque proof that exact input/output metadata was validated without storage.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedInvocation<O: Operation> {
    validated: crate::exec::Validated<Descriptor<O>>,
}

impl<O: Operation> ValidatedInvocation<O> {
    pub(crate) fn infer_custom_runtime(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
    ) -> Result<Self, DescriptorError> {
        let outputs = O::infer_outputs(&attributes, &inputs)?;
        Ok(Self {
            validated: crate::exec::Validated::new(
                Descriptor {
                    attributes,
                    inputs,
                    outputs,
                    identity: crate::exec::OperationIdentity::Custom(O::KEY),
                    marker: PhantomData,
                },
                crate::exec::ProofLevel::Dynamic,
            ),
        })
    }

    pub(crate) fn infer_custom_typed<S: crate::shapes::Shape>(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
        expected: &crate::shapes::ShapeValue<S>,
    ) -> Result<Self, DescriptorError> {
        let outputs = O::infer_outputs(&attributes, &inputs)?;
        let actual = match outputs.as_slice() {
            [output] => output
                .shape
                .as_ref()
                .ok_or(DescriptorError::InvalidAttribute {
                    operation: crate::shapes::error::OperationKind::Storage,
                    attribute: "outputs",
                    reason: "typed custom execution requires concrete output shape metadata",
                })?,
            _ => {
                return Err(DescriptorError::InvalidAttribute {
                    operation: crate::shapes::error::OperationKind::Storage,
                    attribute: "outputs",
                    reason: "typed custom execution requires exactly one output",
                });
            }
        };
        if actual != expected.shape_buf() {
            return Err(DescriptorError::Shape(
                crate::shapes::ShapeError::TargetShapeRejected {
                    operation: crate::shapes::error::OperationKind::Storage,
                    rank: actual.len(),
                },
            ));
        }
        Ok(Self {
            validated: crate::exec::Validated::new_with_evidence(
                Descriptor {
                    attributes,
                    inputs,
                    outputs,
                    identity: crate::exec::OperationIdentity::Custom(O::KEY),
                    marker: PhantomData,
                },
                crate::exec::ShapeEvidence::of::<S>(),
            ),
        })
    }

    pub(crate) const fn validated(&self) -> &crate::exec::Validated<Descriptor<O>> {
        &self.validated
    }

    #[must_use]
    /// Borrows the validated descriptor this handle wraps.
    pub const fn descriptor(&self) -> &Descriptor<O> {
        self.validated.descriptor()
    }

    /// The input metadata this invocation was validated against.
    ///
    /// Borrowed from the descriptor rather than held a second time. The two
    /// copies were always equal - the second was built by cloning the first -
    /// and keeping it cost a `Vec<LogicalTensorMeta>` allocation on every
    /// operation the framework executes.
    #[must_use]
    pub fn inputs(&self) -> &[LogicalTensorMeta] {
        self.validated.descriptor().inputs()
    }
}

impl<O: CanonicalOperation> ValidatedInvocation<O>
where
    O::Attributes: AttributeContract,
{
    /// Internal lowering entry point. The output is supplied by the typed
    /// frontend proof; callers outside `incin-core` cannot assert it.
    /// Validate an invocation whose outputs were stated rather than derived.
    ///
    /// Only the descriptor test suite reaches this: both production lowering
    /// paths derive their outputs from [`infer_outputs`] and cannot state them.
    /// It is kept because it is the entry point that exercises the
    /// fabrication check in `verify_outputs` - the one that proves a stated
    /// output disagreeing with the catalog's own inference is refused - and
    /// there is no other way to construct that case.
    #[cfg(test)]
    pub(crate) fn validate(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
        outputs: Vec<LogicalTensorMeta>,
        proof: crate::exec::ProofLevel,
    ) -> Result<Self, DescriptorError> {
        Self::validate_with_provenance(
            attributes,
            inputs,
            outputs,
            proof,
            OutputProvenance::Supplied,
        )
    }

    fn validate_with_provenance(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
        outputs: Vec<LogicalTensorMeta>,
        proof: crate::exec::ProofLevel,
        provenance: OutputProvenance,
    ) -> Result<Self, DescriptorError> {
        let row = catalog_entry(O::ID)
            .ok_or(DescriptorError::MissingCatalogEntry { operation: O::ID })?;
        if !row.input_arity.contains(&inputs.len()) {
            return Err(DescriptorError::Arity {
                operation: O::ID,
                expected: row.input_arity.clone(),
                actual: inputs.len(),
            });
        }
        if !row.output_arity.contains(&outputs.len()) {
            return Err(DescriptorError::OutputArity {
                operation: O::ID,
                expected: row.output_arity.clone(),
                actual: outputs.len(),
            });
        }
        attributes.validate(O::ID, &inputs)?;
        if let Some(has_bias) = attributes.optional_bias() {
            let expected = if has_bias { 3 } else { 2 };
            if inputs.len() != expected {
                return Err(DescriptorError::Arity {
                    operation: O::ID,
                    expected: expected..=expected,
                    actual: inputs.len(),
                });
            }
        }
        if row.empty == EmptyRule::RejectedWhenReductionIsEmpty {
            if let Some(shape) = inputs.first().and_then(|input| input.shape.as_deref()) {
                let reduction_is_empty = attributes
                    .axis()
                    .and_then(|axis| shape.get(axis))
                    .map_or_else(|| shape.contains(&0), |&extent| extent == 0);
                if reduction_is_empty {
                    return Err(invalid(
                        O::ID,
                        "shape",
                        "operation rejects an empty reduction domain",
                    ));
                }
            }
        }
        if let Some(expected) = attributes.expected_output_count(&inputs) {
            if outputs.len() != expected {
                return Err(DescriptorError::OutputArity {
                    operation: O::ID,
                    expected: expected..=expected,
                    actual: outputs.len(),
                });
            }
        }
        // An optimizer step reads and writes one parameter's worth of state.
        // The gradient and every moment buffer describe the same tensor, so a
        // mismatch here is a state-dictionary defect that must fail before any
        // parameter or moment is mutated rather than after a partial update.
        if matches!(
            O::ID,
            OperationKind::SgdStep | OperationKind::AdamStep | OperationKind::AdamWStep
        ) {
            if let Some(parameter) = inputs.first().and_then(|input| input.shape.as_deref()) {
                for (offset, operand) in inputs.iter().enumerate().skip(1) {
                    if let Some(shape) = operand.shape.as_deref() {
                        if shape != parameter {
                            return Err(invalid(
                                O::ID,
                                match offset {
                                    1 => "gradient shape",
                                    _ => "optimizer state shape",
                                },
                                "optimizer gradient and state must match the parameter shape",
                            ));
                        }
                    }
                }
            }
        }
        if O::ID == OperationKind::Dot {
            if let (Some(lhs), Some(rhs)) = (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
            ) {
                // Checked rather than indexed. A rank-zero operand has no
                // extent to read, and a validation path that indexes into one
                // panics in the place whose entire purpose is to return a
                // diagnostic instead.
                if lhs.is_empty() || rhs.is_empty() {
                    return Err(invalid(
                        O::ID,
                        "shape",
                        "dot requires an axis to contract over",
                    ));
                }
                if lhs[0] != rhs[0] {
                    return Err(invalid(
                        O::ID,
                        "shape",
                        "dot inputs must have equal extents",
                    ));
                }
            }
        }
        match O::ID {
            OperationKind::MaskedFill => {
                if let (Some(value), Some(mask)) = (
                    inputs.first().and_then(|input| input.shape.as_deref()),
                    inputs.get(1).and_then(|input| input.shape.as_deref()),
                ) {
                    if value != mask {
                        return Err(invalid(
                            O::ID,
                            "mask",
                            "masked_fill currently requires mask and value shapes to match",
                        ));
                    }
                }
            }
            OperationKind::Gather => {
                if let (Some(source), Some(indices), Some(axis)) = (
                    inputs.first().and_then(|input| input.shape.as_deref()),
                    inputs.get(1).and_then(|input| input.shape.as_deref()),
                    attributes.axis(),
                ) {
                    if source.len() != indices.len()
                        || indices
                            .iter()
                            .enumerate()
                            .any(|(index, &extent)| index != axis && extent > source[index])
                    {
                        return Err(invalid(
                            O::ID,
                            "index shape",
                            "gather indices must match source rank and fit non-gather extents",
                        ));
                    }
                }
            }
            OperationKind::Scatter => {
                if let (Some(target), Some(indices), Some(source), Some(axis)) = (
                    inputs.first().and_then(|input| input.shape.as_deref()),
                    inputs.get(1).and_then(|input| input.shape.as_deref()),
                    inputs.get(2).and_then(|input| input.shape.as_deref()),
                    attributes.axis(),
                ) {
                    if indices != source
                        || target.len() != indices.len()
                        || indices
                            .iter()
                            .enumerate()
                            .any(|(index, &extent)| index != axis && extent > target[index])
                    {
                        return Err(invalid(
                            O::ID,
                            "index/source shape",
                            "scatter index/source shapes must match and fit the target",
                        ));
                    }
                }
            }
            OperationKind::IndexSelect => {
                if let Some(indices) = inputs.get(1).and_then(|input| input.shape.as_deref()) {
                    if indices.len() != 1 {
                        return Err(invalid(
                            O::ID,
                            "index shape",
                            "index_select requires a rank-one index tensor",
                        ));
                    }
                }
            }
            OperationKind::EmbeddingExact => {
                if let Some(weight) = inputs.get(1).and_then(|input| input.shape.as_deref()) {
                    if weight.len() != 2 {
                        return Err(invalid(
                            O::ID,
                            "weight shape",
                            "embedding weight must have rank two",
                        ));
                    }
                }
            }
            OperationKind::MseLoss | OperationKind::L1Loss | OperationKind::BceWithLogitsLoss => {
                if let (Some(prediction), Some(target)) = (
                    inputs.first().and_then(|input| input.shape.as_deref()),
                    inputs.get(1).and_then(|input| input.shape.as_deref()),
                ) {
                    if prediction != target {
                        return Err(invalid(
                            O::ID,
                            "target shape",
                            "elementwise loss prediction and target shapes must match",
                        ));
                    }
                }
            }
            OperationKind::CrossEntropyLoss => {
                if let (Some(prediction), Some(target)) = (
                    inputs.first().and_then(|input| input.shape.as_deref()),
                    inputs.get(1).and_then(|input| input.shape.as_deref()),
                ) {
                    if prediction.len() != 2 || target.len() != 1 || target[0] != prediction[0] {
                        return Err(invalid(
                            O::ID,
                            "target shape",
                            "cross entropy requires logits [batch, classes] and targets [batch]",
                        ));
                    }
                }
            }
            _ => {}
        }
        let mut expected_device = None;
        for (index, input) in inputs.iter().enumerate() {
            if let Some(shape) = &input.shape {
                let expected = operand_ranks(O::ID, row, index);
                if !expected.contains(&shape.len()) {
                    return Err(DescriptorError::Rank {
                        operation: O::ID,
                        input: index,
                        expected,
                        actual: shape.len(),
                    });
                }
            }
            if row.same_device {
                if let Some(device) = input.device {
                    if let Some(expected) = expected_device {
                        if device != expected {
                            return Err(DescriptorError::DeviceMismatch {
                                operation: O::ID,
                                input: index,
                                expected,
                                actual: device,
                            });
                        }
                    } else {
                        expected_device = Some(device);
                    }
                }
            }
        }
        verify_outputs(O::ID, row, &attributes, &inputs, &outputs, provenance)?;
        let descriptor = Descriptor {
            attributes,
            inputs,
            outputs,
            identity: crate::exec::OperationIdentity::Builtin(O::ID),
            marker: PhantomData,
        };
        Ok(Self {
            validated: crate::exec::Validated::new(descriptor, proof),
        })
    }

    /// Validate an invocation whose outputs are derived rather than supplied.
    ///
    /// Runtime inference path: infers output metadata with ProofLevel::Dynamic.
    pub(crate) fn infer_runtime(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
    ) -> Result<Self, DescriptorError> {
        let row = catalog_entry(O::ID)
            .ok_or(DescriptorError::MissingCatalogEntry { operation: O::ID })?;
        attributes.validate(O::ID, &inputs)?;
        let outputs = infer_outputs(O::ID, row, &attributes, &inputs)?;
        Self::validate_with_provenance(
            attributes,
            inputs,
            outputs,
            crate::exec::ProofLevel::Dynamic,
            OutputProvenance::Derived,
        )
    }

    /// Typed inference path: infers output metadata, validates against the expected `ShapeValue<S>`,
    /// and only attaches `S`-derived proof after geometry equality is proven.
    pub(crate) fn infer_typed<S: crate::shapes::Shape>(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
        expected: &crate::shapes::ShapeValue<S>,
    ) -> Result<Self, DescriptorError> {
        let row = catalog_entry(O::ID)
            .ok_or(DescriptorError::MissingCatalogEntry { operation: O::ID })?;
        attributes.validate(O::ID, &inputs)?;
        let outputs = infer_outputs(O::ID, row, &attributes, &inputs)?;

        if outputs.len() != 1 {
            return Err(DescriptorError::InvalidAttribute {
                operation: O::ID,
                attribute: "shape",
                reason: "infer_typed with a single ShapeValue requires an operation with exactly one output",
            });
        }
        let first_output = &outputs[0];
        if let Some(inferred_shape) = &first_output.shape {
            // `ShapeValue::dims` allocates; `shape_buf` borrows. This runs on
            // every typed operation, which is every operation the stable tensor
            // surface performs.
            if inferred_shape.as_ref() != expected.shape_buf().as_ref() {
                return Err(DescriptorError::InvalidAttribute {
                    operation: O::ID,
                    attribute: "shape",
                    reason: "inferred output shape does not match expected typed shape",
                });
            }
        }

        let validated = Self::validate_with_provenance(
            attributes,
            inputs,
            outputs,
            expected.proof_level(),
            OutputProvenance::Derived,
        )?;
        let descriptor = validated.validated.into_descriptor();
        Ok(Self {
            validated: crate::exec::Validated::new_with_evidence(
                descriptor,
                crate::exec::ShapeEvidence::of::<S>(),
            ),
        })
    }
}
