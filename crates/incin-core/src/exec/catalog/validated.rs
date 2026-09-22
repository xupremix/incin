use super::*;

#[cfg(all(test, feature = "std"))]
std::thread_local! {
    /// How many times this test thread has run the attribute contract.
    ///
    /// The dispatch paths below must validate `attributes` exactly once: the
    /// contract depends only on `(operation, inputs)`, so a second call over
    /// the same pair is a complete duplicate validation of metadata the first
    /// call already accepted. These tests count the calls to pin that; the
    /// counter is thread-local because each `#[test]` runs on its own thread,
    /// so parallel tests never read each other's counts (a shared
    /// `AtomicUsize` would).
    static ATTRIBUTE_VALIDATE_CALLS: core::cell::Cell<usize> =
        const { core::cell::Cell::new(0) };
}

/// Records one attribute-contract validation for the counter above.
///
/// Compiles to nothing outside instrumented test builds, so production call
/// sites pay no branching and no storage.
#[inline]
fn bump_attribute_validate_count() {
    #[cfg(all(test, feature = "std"))]
    ATTRIBUTE_VALIDATE_CALLS.with(|calls| calls.set(calls.get() + 1));
}

/// Drains this thread's attribute-contract validation count.
#[cfg(all(test, feature = "std"))]
pub(super) fn take_attribute_validate_count() -> usize {
    ATTRIBUTE_VALIDATE_CALLS.with(|calls| calls.replace(0))
}

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

    pub(crate) fn infer_custom_typed<E: crate::shapes::ExpectedShapes>(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
        expected: &E,
    ) -> Result<Self, DescriptorError> {
        let outputs = O::infer_outputs(&attributes, &inputs)?;
        if outputs.len() != E::ARITY {
            return Err(DescriptorError::OutputArity {
                operation: crate::shapes::error::OperationKind::Storage,
                expected: E::ARITY..=E::ARITY,
                actual: outputs.len(),
            });
        }
        for (index, (output, expected_buf)) in outputs.iter().zip(expected.shape_bufs()).enumerate()
        {
            // A custom inference that produced no shape for an output leaves
            // nothing to check the caller against. The canonical path below
            // skips such outputs; the custom path refuses them instead, as
            // before -- a caller-held proof must be met, not waived.
            let actual = output
                .shape
                .as_ref()
                .ok_or(DescriptorError::MetadataMismatch {
                    operation: crate::shapes::error::OperationKind::Storage,
                    output: index,
                    field: "shape",
                })?;
            if actual != expected_buf {
                return Err(DescriptorError::MetadataMismatch {
                    operation: crate::shapes::error::OperationKind::Storage,
                    output: index,
                    field: "shape",
                });
            }
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
                expected.combined_evidence(),
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
        let row = Self::entry_arities(&inputs, &outputs)?;
        bump_attribute_validate_count();
        attributes.validate(O::ID, &inputs)?;
        Self::finish_validation(
            row,
            attributes,
            inputs,
            outputs,
            crate::exec::ShapeEvidence::weakened(proof),
            OutputProvenance::Supplied,
        )
    }

    /// Shared finishing path: catalog arities already checked, attribute
    /// contract already validated by whichever entry point dispatched here.
    ///
    /// Takes the [`ShapeEvidence`] to seal rather than a bare proof level:
    /// `infer_runtime` passes [`dynamic()`], the test-only `validate` passes
    /// [`weakened()`], and `infer_typed` passes the caller-held
    /// [`combined_evidence()`] directly, so the typed path no longer seals a
    /// level and then re-mints the same descriptor with full evidence.
    ///
    /// [`dynamic()`]: crate::exec::ShapeEvidence::dynamic
    /// [`weakened()`]: crate::exec::ShapeEvidence::weakened
    /// [`combined_evidence()`]: crate::shapes::ExpectedShapes::combined_evidence
    fn validate_with_provenance(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
        outputs: Vec<LogicalTensorMeta>,
        evidence: crate::exec::ShapeEvidence,
        provenance: OutputProvenance,
    ) -> Result<Self, DescriptorError> {
        let row = Self::entry_arities(&inputs, &outputs)?;
        Self::finish_validation(row, attributes, inputs, outputs, evidence, provenance)
    }

    /// Looks up the catalog row and rejects input/output arity violations.
    ///
    /// The fail-closed prefix of every validation entry, factored so the
    /// attribute contract can sit between it and [`finish_validation`]:
    /// arities are checked before `attributes.validate`, exactly as they
    /// always were, and both production entry points share one copy of the
    /// checks instead of re-walking them inside a nested call.
    fn entry_arities(
        inputs: &[LogicalTensorMeta],
        outputs: &[LogicalTensorMeta],
    ) -> Result<&'static OperationCatalogEntry, DescriptorError> {
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
        Ok(row)
    }

    /// Everything after the attribute contract: operation-specific shape
    /// rules, per-input rank/device checks, `verify_outputs`, and the seal.
    ///
    /// Deliberately does *not* call `attributes.validate`: the contract is a
    /// pure function of `(operation, inputs)` and has already run at the
    /// entry point that reached this function. Every check below either
    /// reads `outputs` (which only exists after inference) or reads state
    /// the contract does not cover, so nothing is lost by running it once.
    fn finish_validation(
        row: &'static OperationCatalogEntry,
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
        outputs: Vec<LogicalTensorMeta>,
        evidence: crate::exec::ShapeEvidence,
        provenance: OutputProvenance,
    ) -> Result<Self, DescriptorError> {
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
                    // #100 relaxed this from equality to the shared broadcast
                    // rule: the mask may be rank-deficit or carry size-1 axes
                    // as long as it broadcasts *into* the input. Resolving the
                    // pair must land back on the input's own extents - any
                    // axis on which the mask would enlarge the input, or fail
                    // to agree with it, is refused here rather than reaching a
                    // backend whose kernels index the input's geometry.
                    let resolves_to_input =
                        crate::shapes::broadcast::broadcast_dim_slices(value, mask)
                            .is_ok_and(|resolved| resolved == value);
                    if !resolves_to_input {
                        return Err(invalid(
                            O::ID,
                            "mask",
                            "mask must broadcast to the input shape",
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
            validated: crate::exec::Validated::new_with_evidence(descriptor, evidence),
        })
    }

    /// Validate an invocation whose outputs are derived rather than supplied.
    ///
    /// Runtime inference path: infers output metadata and seals it with
    /// dynamic evidence, which claims nothing the runtime field does not.
    pub(crate) fn infer_runtime(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
    ) -> Result<Self, DescriptorError> {
        let row = catalog_entry(O::ID)
            .ok_or(DescriptorError::MissingCatalogEntry { operation: O::ID })?;
        bump_attribute_validate_count();
        attributes.validate(O::ID, &inputs)?;
        let outputs = infer_outputs(O::ID, row, &attributes, &inputs)?;
        Self::validate_with_provenance(
            attributes,
            inputs,
            outputs,
            crate::exec::ShapeEvidence::dynamic(),
            OutputProvenance::Derived,
        )
    }

    /// Typed inference path: infers output metadata, validates against the
    /// caller-held shape proofs, and only attaches derived proof after
    /// geometry equality is proven element-wise.
    pub(crate) fn infer_typed<E: crate::shapes::ExpectedShapes>(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
        expected: &E,
    ) -> Result<Self, DescriptorError> {
        let row = catalog_entry(O::ID)
            .ok_or(DescriptorError::MissingCatalogEntry { operation: O::ID })?;
        bump_attribute_validate_count();
        attributes.validate(O::ID, &inputs)?;
        let outputs = infer_outputs(O::ID, row, &attributes, &inputs)?;

        if outputs.len() != E::ARITY {
            return Err(DescriptorError::OutputArity {
                operation: O::ID,
                expected: E::ARITY..=E::ARITY,
                actual: outputs.len(),
            });
        }
        for (index, (output, expected_buf)) in outputs.iter().zip(expected.shape_bufs()).enumerate()
        {
            if let Some(inferred_shape) = &output.shape {
                // `ShapeValue::dims` allocates; `shape_buf` borrows. This runs on
                // every typed operation, which is every operation the stable tensor
                // surface performs -- hence the borrow on both sides, and the
                // `ExpectedShapes` bound instead of a `Vec` of buffers.
                if inferred_shape.as_ref() != expected_buf.as_ref() {
                    return Err(DescriptorError::MetadataMismatch {
                        operation: O::ID,
                        output: index,
                        field: "shape",
                    });
                }
            }
        }

        Self::validate_with_provenance(
            attributes,
            inputs,
            outputs,
            expected.combined_evidence(),
            OutputProvenance::Derived,
        )
    }
}
