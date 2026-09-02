use super::*;
use alloc::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TestCustomOperation;

impl Operation for TestCustomOperation {
    type Attributes = NoAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("incin.test"),
        name: Cow::Borrowed("identity"),
        version: 1,
    };

    fn infer_outputs(
        _: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(inputs.first().cloned().into_iter().collect())
    }
}

#[test]
fn custom_operation_keeps_static_shape_proof() {
    let input = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2, 3])),
        dtype: None,
        device: None,
    };
    let expected =
        crate::shapes::ShapeValue::<crate::shapes::Dyn>::try_new(ShapeBuf::from_slice(&[2, 3]))
            .unwrap();
    let invocation = ValidatedInvocation::<TestCustomOperation>::infer_custom_typed(
        NoAttributes,
        vec![input],
        &expected,
    )
    .unwrap();

    assert_eq!(
        invocation.validated().descriptor().key(),
        TestCustomOperation::KEY
    );
    assert_eq!(
        invocation.validated().proof_level(),
        crate::exec::ProofLevel::of::<crate::shapes::Dyn>()
    );
}

macro_rules! custom_shape_case {
    ($name:ident, $key:literal, $outputs:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct $name;

        impl Operation for $name {
            type Attributes = NoAttributes;
            const KEY: OperationKey = OperationKey {
                namespace: Cow::Borrowed("incin.test"),
                name: Cow::Borrowed($key),
                version: 1,
            };

            fn infer_outputs(
                _: &Self::Attributes,
                _: &[LogicalTensorMeta],
            ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
                Ok($outputs)
            }
        }
    };
}

custom_shape_case!(
    NoShapeCustomOperation,
    "no-shape",
    vec![LogicalTensorMeta {
        shape: None,
        dtype: None,
        device: None,
    }]
);
custom_shape_case!(ZeroOutputCustomOperation, "zero-output", Vec::new());
custom_shape_case!(
    MultiOutputCustomOperation,
    "multi-output",
    vec![
        LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(&[2, 3])),
            dtype: None,
            device: None,
        },
        LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(&[2, 3])),
            dtype: None,
            device: None,
        },
    ]
);

#[test]
fn custom_typed_proof_requires_one_concrete_output() {
    let expected =
        crate::shapes::ShapeValue::<crate::shapes::Dyn>::try_new(ShapeBuf::from_slice(&[2, 3]))
            .unwrap();

    assert!(matches!(
        ValidatedInvocation::<NoShapeCustomOperation>::infer_custom_typed(
            NoAttributes,
            Vec::new(),
            &expected,
        ),
        Err(DescriptorError::InvalidAttribute { .. })
    ));
    assert!(matches!(
        ValidatedInvocation::<ZeroOutputCustomOperation>::infer_custom_typed(
            NoAttributes,
            Vec::new(),
            &expected,
        ),
        Err(DescriptorError::InvalidAttribute { .. })
    ));
    assert!(matches!(
        ValidatedInvocation::<MultiOutputCustomOperation>::infer_custom_typed(
            NoAttributes,
            Vec::new(),
            &expected,
        ),
        Err(DescriptorError::InvalidAttribute { .. })
    ));
}

#[test]
fn operation_key_round_trips_through_persistence() {
    let key = TestCustomOperation::KEY;
    let encoded = serde_json::to_string(&key).unwrap();
    let decoded: OperationKey = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, key);
}

#[test]
fn operation_key_persistence_accepts_runtime_owned_identity() {
    let key = OperationKey {
        namespace: Cow::Owned("external.runtime".to_owned()),
        name: Cow::Owned("custom_op".to_owned()),
        version: 7,
    };
    let encoded = serde_json::to_string(&key).unwrap();
    let decoded: OperationKey = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, key);
    assert!(matches!(decoded.namespace, Cow::Owned(_)));
    assert!(matches!(decoded.name, Cow::Owned(_)));
}

#[test]
fn identities_and_names_occur_exactly_once() {
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    for row in OPERATION_CATALOG {
        assert!(row.operation.is_exact());
        assert!(
            ids.insert(row.operation),
            "duplicate identity {}",
            row.operation
        );
        assert!(names.insert(row.name), "duplicate name {}", row.name);
        assert_eq!(row.operation.name(), row.name);
    }
}

#[test]
fn coverage_is_derived_from_catalog_rows() {
    let coverage = operation_coverage();
    assert_eq!(coverage.canonical, OPERATION_CATALOG.len());
    assert_eq!(
        coverage.backend_executable + coverage.non_backend_executable,
        coverage.canonical
    );
    assert_eq!(
        coverage
            .by_site
            .iter()
            .map(|(_, count)| count)
            .sum::<usize>(),
        coverage.canonical
    );
    assert_eq!(
        coverage.non_backend_executable,
        OPERATION_CATALOG
            .iter()
            .filter(|row| !row.site.is_backend_executable())
            .count()
    );
}

/// The execution site agrees with the arity and output rules that were
/// already in the catalog.
///
/// The site is a second description of facts the catalog partly recorded
/// already, so it can be checked against them rather than merely asserted.
/// A creation operation is exactly one that takes no operand; a host
/// readback is exactly one whose output rule is `HostValue` and whose
/// tensor output count is zero. Where the two descriptions could drift,
/// this fails.
#[test]
fn the_execution_site_agrees_with_the_arity_and_output_rules() {
    for row in OPERATION_CATALOG {
        match row.site {
            ExecutionSite::Creation => assert_eq!(
                (*row.input_arity.start(), *row.input_arity.end()),
                (0, 0),
                "{} is classified as a creation but accepts operands",
                row.name,
            ),
            ExecutionSite::HostReadback => {
                assert_eq!(
                    row.output,
                    OutputRule::HostValue,
                    "{} is classified as a host readback but does not produce a host value",
                    row.name,
                );
                assert_eq!(
                    *row.output_arity.end(),
                    0,
                    "{} is classified as a host readback but returns a tensor",
                    row.name,
                );
            }
            // The converse of the readback case: a host value must be
            // classified as one, or the report would count it as a kernel
            // that nobody has written.
            ExecutionSite::Kernel => assert_ne!(
                row.output,
                OutputRule::HostValue,
                "{} produces a host value but is classified as a kernel",
                row.name,
            ),
            ExecutionSite::Composed => assert!(
                matches!(
                    row.operation,
                    OperationKind::Sample | OperationKind::Rnn | OperationKind::Lstm
                ),
                "{} is classified as composed without a frontend composition",
                row.name,
            ),
            ExecutionSite::Mutation => assert!(
                matches!(
                    row.profile,
                    SemanticProfile::Mutation | SemanticProfile::Optimizer
                ),
                "{} is classified as a mutation but its profile is {:?}",
                row.name,
                row.profile,
            ),
            ExecutionSite::DeviceTransfer => assert!(
                !row.same_device,
                "{} moves between devices but is declared same-device",
                row.name,
            ),
            ExecutionSite::GraphState => assert_eq!(
                row.profile,
                SemanticProfile::Autograd,
                "{} is classified as graph state but its profile is not autograd",
                row.name,
            ),
        }

        assert_eq!(
            row.site.is_backend_executable(),
            row.site.blocking_reason().is_none(),
            "{} states a blocking reason inconsistent with its executability",
            row.name,
        );
    }
}

/// Every mutating and autograd operation is classified as such.
///
/// The classification defaults to `Kernel`, which is the fail-closed
/// direction for a new operation but the wrong answer for these two
/// profiles. Deriving the expectation from the profile rather than from a
/// second hand-written list means a newly declared in-place operation
/// fails here instead of being silently counted as an unwritten kernel.
#[test]
fn every_mutating_and_autograd_operation_is_classified_by_its_profile() {
    for row in OPERATION_CATALOG {
        match row.profile {
            SemanticProfile::Mutation | SemanticProfile::Optimizer => assert_eq!(
                row.site,
                ExecutionSite::Mutation,
                "{} writes through an operand but is classified as {:?}",
                row.name,
                row.site,
            ),
            SemanticProfile::Autograd => assert_eq!(
                row.site,
                ExecutionSite::GraphState,
                "{} acts on autograd state but is classified as {:?}",
                row.name,
                row.site,
            ),
            SemanticProfile::Creation => assert!(
                matches!(row.site, ExecutionSite::Creation | ExecutionSite::Composed),
                "{} creates storage but is classified as {:?}",
                row.name,
                row.site,
            ),
            _ => {}
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "document formatting is tested by ordinary test suite")]
fn generated_semantics_document_covers_every_row() {
    let document = operation_semantics_document();
    for row in OPERATION_CATALOG {
        assert!(document.contains(&alloc::format!("| `{}` |", row.name)));
    }
}

#[test]
fn every_typed_output_rule_is_fail_closed_or_exactly_inferred() {
    for row in OPERATION_CATALOG
        .iter()
        .filter(|row| row.output == OutputRule::TypedInference)
    {
        let inferred = matches!(
            row.operation,
            OperationKind::Dot
                | OperationKind::Outer
                | OperationKind::Addmm
                | OperationKind::ScaledDotProductAttention
                | OperationKind::Linear
                | OperationKind::EmbeddingExact
                | OperationKind::Quantize
                | OperationKind::Dequantize
                | OperationKind::SgdStep
                | OperationKind::AdamStep
                | OperationKind::AdamWStep
        );
        assert!(
            inferred || *row.output_arity.end() == 0,
            "{} has no typed output inference branch",
            row.operation
        );
    }
}

#[test]
fn unknown_metadata_stays_unknown() {
    let invocation = ValidatedInvocation::<op::Relu>::validate(
        NoAttributes,
        vec![LogicalTensorMeta::unknown()],
        vec![LogicalTensorMeta::unknown()],
        crate::exec::ProofLevel::Dynamic,
    )
    .unwrap();
    assert_eq!(
        invocation.descriptor().outputs(),
        &[LogicalTensorMeta::unknown()]
    );
}

#[test]
fn validation_rejects_wrong_arity_and_cross_device_inputs() {
    let cpu = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let cuda = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cuda(0)),
    };
    assert!(matches!(
        ValidatedInvocation::<op::Add>::validate(
            NoAttributes,
            vec![cpu.clone()],
            vec![cpu.clone()],
            crate::exec::ProofLevel::Dynamic
        ),
        Err(DescriptorError::Arity { .. })
    ));
    assert!(matches!(
        ValidatedInvocation::<op::Add>::validate(
            NoAttributes,
            vec![cpu.clone(), cuda],
            vec![cpu],
            crate::exec::ProofLevel::Dynamic
        ),
        Err(DescriptorError::DeviceMismatch { .. })
    ));
}

/// Helper for the per-operand rank tests: known shape, f32, CPU.
fn meta(shape: &[usize]) -> LogicalTensorMeta {
    typed_meta(shape, DTypeId::F32.descriptor())
}

/// As [`meta`], for operands whose role fixes a non-float dtype.
fn typed_meta(
    shape: &[usize],
    dtype: impl crate::tensor::arg_into::ArgInto<DTypeDescriptor>,
) -> LogicalTensorMeta {
    LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(shape)),
        dtype: Some(dtype.into_arg()),
        device: Some(DeviceId::cpu()),
    }
}

/// A rank-one bias must validate against a rank-four activation.
///
/// The catalog's `accepted_ranks` for `Conv2dExact` is the activation
/// window `3..=4`. Applying it to every operand rejected every biased
/// convolution, because a bias is `[c_out]`.
#[test]
fn convolution_bias_is_validated_by_role_not_by_the_activation_window() {
    let attributes = |has_bias| Conv2dAttributes {
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
        groups: 1,
        has_bias,
    };

    ValidatedInvocation::<op::Conv2dExact>::validate(
        attributes(false),
        vec![meta(&[1, 3, 8, 8]), meta(&[4, 3, 3, 3])],
        vec![meta(&[1, 4, 8, 8])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("an unbiased convolution validates");

    ValidatedInvocation::<op::Conv2dExact>::validate(
        attributes(true),
        vec![meta(&[1, 3, 8, 8]), meta(&[4, 3, 3, 3]), meta(&[4])],
        vec![meta(&[1, 4, 8, 8])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("a rank-one bias validates against a rank-four activation");

    // The role is exact in both directions: an activation-ranked bias is
    // still refused.
    assert!(matches!(
        ValidatedInvocation::<op::Conv2dExact>::validate(
            attributes(true),
            vec![
                meta(&[1, 3, 8, 8]),
                meta(&[4, 3, 3, 3]),
                meta(&[1, 4, 1, 1])
            ],
            vec![meta(&[1, 4, 8, 8])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::Rank { input: 2, .. })
    ));
}

#[test]
fn conv1d_and_transposed_convolution_bias_use_the_same_role_contract() {
    ValidatedInvocation::<op::Conv1dExact>::validate(
        Conv1dAttributes {
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
        },
        vec![meta(&[1, 3, 8]), meta(&[4, 3, 3]), meta(&[4])],
        vec![meta(&[1, 4, 8])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("conv1d accepts a rank-one bias");

    ValidatedInvocation::<op::ConvTranspose2d>::validate(
        ConvTranspose2dAttributes {
            stride: [1, 1],
            padding: [1, 1],
            output_padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            has_bias: true,
        },
        vec![meta(&[1, 3, 8, 8]), meta(&[3, 4, 3, 3]), meta(&[4])],
        vec![meta(&[1, 4, 8, 8])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("conv_transpose2d accepts a rank-one bias");

    // A rank-two weight is not a 2-D filter bank.
    assert!(matches!(
        ValidatedInvocation::<op::Conv2dExact>::validate(
            Conv2dAttributes {
                stride: [1, 1],
                padding: [0, 0],
                dilation: [1, 1],
                groups: 1,
                has_bias: false,
            },
            vec![meta(&[1, 3, 8, 8]), meta(&[4, 3])],
            vec![meta(&[1, 4, 6, 6])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::Rank { input: 1, .. })
    ));
}

#[test]
fn embedding_indices_and_weight_have_separate_rank_contracts() {
    let indices = typed_meta(&[2, 5], DTypeId::I64);
    ValidatedInvocation::<op::EmbeddingExact>::validate(
        NoAttributes,
        vec![indices.clone(), meta(&[10, 4])],
        vec![meta(&[2, 5, 4])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("rank-two integer indices address a rank-two table");

    // A rank-three table is not an embedding matrix. The typed weight rule
    // reports this before the role window is consulted; either way the
    // descriptor fails closed rather than inferring a table geometry.
    assert!(matches!(
        ValidatedInvocation::<op::EmbeddingExact>::validate(
            NoAttributes,
            vec![indices, meta(&[10, 4, 1])],
            vec![meta(&[2, 5, 4])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "weight shape",
            ..
        }) | Err(DescriptorError::Rank { input: 1, .. })
    ));

    // Indices must address something; a rank-zero scalar is not a batch,
    // and only the per-role window rejects it.
    assert!(matches!(
        ValidatedInvocation::<op::EmbeddingExact>::validate(
            NoAttributes,
            vec![typed_meta(&[], DTypeId::I64), meta(&[10, 4])],
            vec![meta(&[4])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::Rank { input: 0, .. })
    ));
}

#[test]
fn batch_norm_state_is_rank_one_against_a_ranked_activation() {
    let attributes = BatchNormAttributes {
        epsilon: 1e-5,
        momentum: 0.1,
        training: false,
        has_weight: true,
        has_bias: true,
        has_running_mean: true,
        has_running_variance: true,
    };
    ValidatedInvocation::<op::BatchNorm>::validate(
        attributes.clone(),
        vec![
            meta(&[2, 3, 4, 4]),
            meta(&[3]),
            meta(&[3]),
            meta(&[3]),
            meta(&[3]),
        ],
        vec![meta(&[2, 3, 4, 4])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("rank-one affine and running state validate");

    // A rank-two affine parameter is refused. The typed per-channel extent
    // rule reports it first; the role window is the backstop that keeps the
    // rank pinned even where the extent rule cannot reach.
    assert!(matches!(
        ValidatedInvocation::<op::BatchNorm>::validate(
            attributes,
            vec![
                meta(&[2, 3, 4, 4]),
                meta(&[1, 3]),
                meta(&[3]),
                meta(&[3]),
                meta(&[3]),
            ],
            vec![meta(&[2, 3, 4, 4])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "parameter shape",
            ..
        }) | Err(DescriptorError::Rank { input: 1, .. })
    ));
    assert_eq!(
        operand_ranks(
            OperationKind::BatchNorm,
            catalog_entry(OperationKind::BatchNorm).unwrap(),
            1
        ),
        1..=1
    );
}

#[test]
fn linear_weight_and_bias_have_separate_rank_contracts() {
    ValidatedInvocation::<op::Linear>::validate(
        LinearAttributes { has_bias: true },
        vec![meta(&[2, 3]), meta(&[4, 3]), meta(&[4])],
        vec![meta(&[2, 4])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("a rank-two weight and rank-one bias validate");

    assert!(matches!(
        ValidatedInvocation::<op::Linear>::validate(
            LinearAttributes { has_bias: true },
            vec![meta(&[2, 3]), meta(&[4, 3]), meta(&[1, 4])],
            vec![meta(&[2, 4])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::Rank { input: 2, .. })
    ));

    assert!(matches!(
        ValidatedInvocation::<op::Linear>::validate(
            LinearAttributes { has_bias: false },
            vec![meta(&[2, 3]), meta(&[4, 3, 1])],
            vec![meta(&[2, 4])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::Rank { input: 1, .. })
    ));
}

#[test]
fn cross_entropy_logits_and_targets_have_separate_rank_contracts() {
    ValidatedInvocation::<op::CrossEntropyLoss>::validate(
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        vec![meta(&[4, 7]), typed_meta(&[4], DTypeId::I64)],
        vec![meta(&[])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("[batch, classes] logits against [batch] integer targets");

    // A rank-two target is a one-hot encoding, which this operation does
    // not consume; it must be refused rather than reinterpreted.
    assert!(matches!(
        ValidatedInvocation::<op::CrossEntropyLoss>::validate(
            LossAttributes {
                reduction: LossReduction::Mean,
            },
            vec![meta(&[4, 7]), typed_meta(&[4, 7], DTypeId::I64)],
            vec![meta(&[])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::Rank { input: 1, .. }) | Err(DescriptorError::InvalidAttribute { .. })
    ));
}

#[test]
fn optimizer_gradient_and_state_are_validated_against_the_parameter() {
    let attributes = AdamAttributes {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        step: 1,
    };
    ValidatedInvocation::<op::AdamStep>::validate(
        attributes.clone(),
        vec![meta(&[3, 4]), meta(&[3, 4]), meta(&[3, 4]), meta(&[3, 4])],
        vec![meta(&[3, 4]), meta(&[3, 4]), meta(&[3, 4])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("parameter, gradient, and both moments agree");

    // A moment buffer from a differently shaped parameter must fail before
    // any state is mutated, not after a partial update.
    assert!(matches!(
        ValidatedInvocation::<op::AdamStep>::validate(
            attributes,
            vec![meta(&[3, 4]), meta(&[3, 4]), meta(&[3, 4]), meta(&[4, 3])],
            vec![meta(&[3, 4]), meta(&[3, 4]), meta(&[3, 4])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "optimizer state shape",
            ..
        })
    ));

    assert!(matches!(
        ValidatedInvocation::<op::SgdStep>::validate(
            SgdAttributes { learning_rate: 0.1 },
            vec![meta(&[3, 4]), meta(&[3, 5])],
            vec![meta(&[3, 4])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "gradient shape",
            ..
        })
    ));
}

/// Family defaults are defaults; an exact operation overrides an incorrect
/// one. These are the exceptions the FND-004 profile audit found.
#[test]
fn index_producing_operations_reject_a_non_integer_index_dtype() {
    // `IndexResult` is the family intent; without this guard a caller could
    // declare a float "index" dtype and have the descriptor certify it.
    assert!(matches!(
        ValidatedInvocation::<op::ArgMax>::validate(
            IndexReductionAttributes {
                axis: Some(1),
                dtype: DTypeId::F32.descriptor(),
            },
            vec![meta(&[2, 3])],
            vec![typed_meta(&[2], DTypeId::F32.descriptor())],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "dtype",
            ..
        })
    ));
    ValidatedInvocation::<op::ArgMax>::validate(
        IndexReductionAttributes {
            axis: Some(1),
            dtype: DTypeId::I64.descriptor(),
        },
        vec![meta(&[2, 3])],
        vec![typed_meta(&[2], DTypeId::I64.descriptor())],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("an integer index dtype validates");

    assert!(matches!(
        ValidatedInvocation::<op::Argsort>::validate(
            ArgsortAttributes {
                axis: 0,
                descending: false,
                index_dtype: DTypeId::F64.descriptor(),
            },
            vec![meta(&[4])],
            vec![typed_meta(&[4], DTypeId::F64.descriptor())],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "index_dtype",
            ..
        })
    ));
}

/// `topk` is a two-output exception: the values keep the input dtype while
/// the indices take the declared integer dtype. Neither is the family
/// default applied uniformly.
#[test]
fn topk_values_keep_the_input_dtype_while_indices_take_the_index_dtype() {
    ValidatedInvocation::<op::TopK>::validate(
        TopKAttributes {
            k: 2,
            axis: 0,
            largest: true,
            index_dtype: DTypeId::I64.descriptor(),
        },
        vec![meta(&[4])],
        vec![meta(&[2]), typed_meta(&[2], DTypeId::I64.descriptor())],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("values in the input dtype, indices in the index dtype");

    // Indices cannot silently adopt the value dtype.
    assert!(matches!(
        ValidatedInvocation::<op::TopK>::validate(
            TopKAttributes {
                k: 2,
                axis: 0,
                largest: true,
                index_dtype: DTypeId::I64.descriptor(),
            },
            vec![meta(&[4])],
            vec![meta(&[2]), meta(&[2])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::MetadataMismatch {
            output: 1,
            field: "dtype",
            ..
        })
    ));
}

/// The `Reduction` family only rejects an *empty* domain. An unbiased
/// estimate is also undefined on a single element, because the correction
/// divides by `n - 1`.
#[test]
fn an_unbiased_estimate_rejects_a_single_element_domain() {
    assert!(matches!(
        ValidatedInvocation::<op::VarianceAll>::validate(
            VarianceAttributes { unbiased: true },
            vec![meta(&[1])],
            vec![meta(&[])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "unbiased",
            ..
        })
    ));
    // The biased estimate over the same domain is well defined.
    ValidatedInvocation::<op::VarianceAll>::validate(
        VarianceAttributes { unbiased: false },
        vec![meta(&[1])],
        vec![meta(&[])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("a biased estimate divides by n");

    assert!(matches!(
        ValidatedInvocation::<op::StdDim>::validate(
            AxisVarianceAttributes {
                axis: 1,
                unbiased: true,
            },
            vec![meta(&[4, 1])],
            vec![meta(&[4])],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "unbiased",
            ..
        })
    ));
}

/// The axis-bearing variance operations infer an output shape.
///
/// The test above proves `AxisVarianceAttributes` validates its axis. That
/// is a weaker property than it looks: the attributes validated the axis
/// through a borrowed `AxisAttributes` and then did not expose it, so
/// inference fell to the fail-closed arm and every invocation of these four
/// operations failed with `MissingInference`. Validation alone could never
/// have caught that, which is why this checks the derived shape instead.
#[test]
fn the_axis_variance_operations_infer_their_output_shape() {
    let attributes = AxisVarianceAttributes {
        axis: 1,
        unbiased: false,
    };
    // Reducing axis one of [4, 3] drops it; the keep-dim forms collapse it
    // to one instead.
    ValidatedInvocation::<op::VarianceDim>::validate(
        attributes.clone(),
        vec![meta(&[4, 3])],
        vec![meta(&[4])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("var_dim removes the reduced axis");
    ValidatedInvocation::<op::StdDim>::validate(
        attributes.clone(),
        vec![meta(&[4, 3])],
        vec![meta(&[4])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("std_dim removes the reduced axis");
    ValidatedInvocation::<op::VarianceKeepDim>::validate(
        attributes.clone(),
        vec![meta(&[4, 3])],
        vec![meta(&[4, 1])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("var_keepdim collapses the reduced axis to one");
    ValidatedInvocation::<op::StdKeepDim>::validate(
        attributes,
        vec![meta(&[4, 3])],
        vec![meta(&[4, 1])],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("std_keepdim collapses the reduced axis to one");
}

#[test]
fn a_comparison_returns_boolean_dtype() {
    ValidatedInvocation::<op::CmpLt>::validate(
        NoAttributes,
        vec![meta(&[3]), meta(&[3])],
        vec![typed_meta(&[3], DTypeId::Bool)],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("comparison produces a boolean output");

    assert!(matches!(
        ValidatedInvocation::<op::CmpLt>::validate(
            NoAttributes,
            vec![meta(&[3]), meta(&[3])],
            vec![typed_meta(&[3], DTypeId::F32)],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::MetadataMismatch { field: "dtype", .. })
    ));
}

#[test]
fn logical_ops_require_boolean_inputs() {
    assert!(matches!(
        ValidatedInvocation::<op::LogicalAnd>::validate(
            NoAttributes,
            vec![
                typed_meta(&[3], DTypeId::F32),
                typed_meta(&[3], DTypeId::F32)
            ],
            vec![typed_meta(&[3], DTypeId::Bool)],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "dtype",
            ..
        })
    ));
}

/// Sum and product have an identity on an empty domain; mean and the
/// extrema do not. The family default is the strict one, so the identity
/// cases are the overrides.
#[test]
fn empty_domain_behaviour_splits_within_the_reduction_family() {
    for operation in [
        OperationKind::SumAll,
        OperationKind::ProdAll,
        OperationKind::SumDim,
        OperationKind::Cumsum,
    ] {
        assert_eq!(
            catalog_entry(operation).unwrap().empty,
            EmptyRule::IdentityOrDefined,
            "{operation} has an identity on an empty domain",
        );
    }
    for operation in [
        OperationKind::MeanAll,
        OperationKind::MaxAll,
        OperationKind::MinAll,
    ] {
        assert_eq!(
            catalog_entry(operation).unwrap().empty,
            EmptyRule::RejectedWhenReductionIsEmpty,
            "{operation} is undefined on an empty domain",
        );
    }
}

/// Operations whose result is not a tensor declare zero outputs, and
/// non-differentiable and nondeterministic exceptions override their
/// family default rather than inheriting it.
#[test]
fn zero_output_gradient_and_determinism_exceptions_override_their_family() {
    for operation in [
        OperationKind::ToHostFloatScalar,
        OperationKind::ToHostIntVec,
        OperationKind::TensorToBytes,
        OperationKind::Backward,
    ] {
        let row = catalog_entry(operation).unwrap();
        assert_eq!(*row.output_arity.end(), 0, "{operation} returns no tensor");
        assert_eq!(row.gradient, GradientRule::None, "{operation}");
    }

    // Piecewise-constant unary functions are float-typed but have no
    // useful gradient, unlike the rest of `UnaryFloat`.
    for operation in [
        OperationKind::Step,
        OperationKind::Sign,
        OperationKind::Floor,
        OperationKind::Round,
    ] {
        assert_eq!(
            catalog_entry(operation).unwrap().gradient,
            GradientRule::Undefined,
            "{operation}",
        );
    }

    for operation in [
        OperationKind::UniformRandom,
        OperationKind::Dropout,
        OperationKind::TopK,
        OperationKind::Argsort,
    ] {
        assert!(
            !catalog_entry(operation).unwrap().deterministic,
            "{operation} is not deterministic",
        );
    }

    // Transfer operations deliberately change device or dtype, so they are
    // the family that opts out of the same-device requirement.
    for operation in [OperationKind::ToDevice, OperationKind::ToDType] {
        assert!(
            !catalog_entry(operation).unwrap().same_device,
            "{operation} may cross devices",
        );
    }
    assert!(catalog_entry(OperationKind::Add).unwrap().same_device);
}

/// Known inputs must never certify an unchecked output shape.
///
/// `verify_outputs` used to accept whatever shape the caller supplied
/// whenever no inference branch produced an expectation. That let a fully
/// known input set certify a fabricated output. Unknown inputs still
/// legitimately produce unknown outputs; known inputs now fail closed with
/// `MissingInference`.
#[test]
fn known_inputs_never_skip_output_shape_verification() {
    // `argmax` without an axis over a *known* input has an exact answer
    // (a scalar), so it is verified rather than waved through.
    assert!(matches!(
        ValidatedInvocation::<op::ArgMax>::validate(
            IndexReductionAttributes {
                axis: None,
                dtype: DTypeId::I64.descriptor(),
            },
            vec![meta(&[2, 3])],
            vec![typed_meta(&[9, 9], DTypeId::I64.descriptor())],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::MetadataMismatch { field: "shape", .. })
    ));

    // Unknown input shape keeps the output shape unknown rather than
    // inventing one, and rather than failing. The index dtype is declared
    // by the attributes, so it stays known and is still verified.
    let output = LogicalTensorMeta {
        shape: None,
        dtype: Some(DTypeId::I64.descriptor()),
        device: None,
    };
    let unknown = ValidatedInvocation::<op::ArgMax>::validate(
        IndexReductionAttributes {
            axis: None,
            dtype: DTypeId::I64.descriptor(),
        },
        vec![LogicalTensorMeta::unknown()],
        vec![output.clone()],
        crate::exec::ProofLevel::Dynamic,
    )
    .expect("an unknown input shape stays unknown");
    assert_eq!(unknown.descriptor().outputs(), &[output]);
}

/// Every catalog row that returns a tensor must have a reachable inference
/// branch, or declare zero outputs. A row reaching neither would be one
/// whose outputs are only ever caller-asserted.
#[test]
fn every_tensor_returning_row_declares_an_inference_source() {
    for row in OPERATION_CATALOG {
        if *row.output_arity.end() == 0 {
            continue;
        }
        // `TypedInference` and `Indexing` are hand-written branches; the
        // rest derive from the declared shape, the input, or a transform.
        let has_source = match row.output {
            OutputRule::TypedInference | OutputRule::Indexing => matches!(
                row.operation,
                OperationKind::Gather
                    | OperationKind::Scatter
                    | OperationKind::IndexSelect
                    | OperationKind::EmbeddingExact
                    | OperationKind::Dot
                    | OperationKind::Outer
                    | OperationKind::Addmm
                    | OperationKind::ScaledDotProductAttention
                    | OperationKind::Linear
                    | OperationKind::Quantize
                    | OperationKind::Dequantize
                    | OperationKind::SgdStep
                    | OperationKind::AdamStep
                    | OperationKind::AdamWStep
            ),
            OutputRule::HostValue => false,
            _ => true,
        };
        assert!(
            has_source,
            "{} returns a tensor with no output inference source",
            row.operation,
        );
    }
}

#[test]
fn attribute_bearing_descriptor_round_trips_without_storage() {
    let input = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[1, 3, 8, 8])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let weight = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[4, 3, 3, 3])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let output = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[1, 4, 8, 8])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let invocation = ValidatedInvocation::<op::Conv2dExact>::validate(
        Conv2dAttributes {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
            has_bias: false,
        },
        vec![input, weight],
        vec![output],
        crate::exec::ProofLevel::Dynamic,
    )
    .unwrap();
    let json = serde_json::to_string(invocation.descriptor()).unwrap();
    let restored: Descriptor<op::Conv2dExact> = serde_json::from_str(&json).unwrap();
    assert_eq!(&restored, invocation.descriptor());
    assert_eq!(restored.operation(), OperationKind::Conv2dExact);
    assert_eq!(restored.attributes().groups, 1);
    let captured = CapturedDescriptor::capture(invocation.descriptor()).unwrap();
    let decoded = captured.decode::<op::Conv2dExact>().unwrap();
    assert_eq!(decoded, restored);
    assert!(matches!(
        captured.decode::<op::Add>(),
        Err(DescriptorCaptureError::Identity { .. })
    ));
    let mut stale_json = serde_json::to_value(&captured).unwrap();
    stale_json["schema"] = serde_json::json!(0);
    let stale: CapturedDescriptor = serde_json::from_value(stale_json).unwrap();
    assert!(matches!(
        stale.decode::<op::Conv2dExact>(),
        Err(DescriptorCaptureError::Schema { .. })
    ));

    let mut forged = invocation.descriptor().clone();
    forged.identity = crate::exec::OperationIdentity::Builtin(OperationKind::Add);
    let mut tampered = captured.clone();
    tampered.payload = postcard::to_allocvec(&forged).unwrap();
    assert!(matches!(
        tampered.decode::<op::Conv2dExact>(),
        Err(DescriptorCaptureError::Identity { .. })
            | Err(DescriptorCaptureError::CustomIdentity { .. })
            | Err(DescriptorCaptureError::Decode(_))
    ));
}

#[test]
fn typed_attributes_fail_before_storage_access() {
    let input = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2, 3])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    assert!(matches!(
        ValidatedInvocation::<op::Clamp>::validate(
            ClampAttributes { min: 2.0, max: 1.0 },
            vec![input.clone()],
            vec![input.clone()],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "min/max",
            ..
        })
    ));
    assert!(matches!(
        ValidatedInvocation::<op::Softmax>::validate(
            AxisAttributes { axis: 2 },
            vec![input.clone()],
            vec![input],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "axis",
            ..
        })
    ));
    assert!(matches!(
        ValidatedInvocation::<op::AdamStep>::validate(
            AdamAttributes {
                learning_rate: 1e-3,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                step: 0,
            },
            vec![LogicalTensorMeta::unknown(); 4],
            vec![LogicalTensorMeta::unknown(); 3],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "beta/step",
            ..
        })
    ));
}

/// The frontend's shape evidence has to survive `infer` and land on the
/// `Validated` a backend actually receives, or reading `Shape::PROOF` at
/// the typed surface buys nothing.
///
/// `dispatch::execute` passes `ProofLevel::Dynamic` because it has no
/// `S` to read; `execute_with_evidence` passes what the typed surface
/// knows. This asserts at the layer where both funnel together that the
/// supplied value is the one that arrives, rather than being replaced by a
/// constant on the way through.
#[test]
fn frontend_shape_evidence_reaches_the_validated_descriptor() {
    let created = CreationAttributes {
        shape: vec![2, 3],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
    };
    type Static23 = crate::shapes::DimCons<
        typenum::U2,
        crate::shapes::DimCons<typenum::U3, crate::shapes::Nil>,
    >;
    let sv = crate::shapes::ShapeValue::<Static23>::from_validated(
        <Static23 as crate::shapes::Shape>::resolve(((), ((), ()))).unwrap(),
    );
    let proven = ValidatedInvocation::<op::Zeros>::infer_typed(created.clone(), vec![], &sv)
        .expect("a static creation request is legal");
    assert_eq!(
        proven.validated().proof_level(),
        crate::exec::ProofLevel::Static,
    );

    // The identical request with nothing known about it must not inherit
    // the proof the previous one earned.
    let erased = ValidatedInvocation::<op::Zeros>::infer_runtime(created, vec![])
        .expect("a dynamic creation request is equally legal");
    assert_eq!(
        erased.validated().proof_level(),
        crate::exec::ProofLevel::Dynamic,
    );
}

#[test]
fn inferred_metadata_cannot_be_fabricated() {
    let lhs = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2, 1])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let rhs = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[1, 3])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let wrong = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2, 1])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    assert!(matches!(
        ValidatedInvocation::<op::Add>::validate(
            NoAttributes,
            vec![lhs, rhs],
            vec![wrong],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::MetadataMismatch { field: "shape", .. })
    ));

    let created = CreationAttributes {
        shape: vec![2, 3],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
    };
    assert!(matches!(
        ValidatedInvocation::<op::Zeros>::validate(
            created,
            vec![],
            vec![LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(&[2, 3])),
                dtype: Some(DTypeId::I64.descriptor()),
                device: Some(DeviceId::cpu()),
            }],
            crate::exec::ProofLevel::Static,
        ),
        Err(DescriptorError::MetadataMismatch { field: "dtype", .. })
    ));
}

#[test]
fn multi_output_shapes_and_counts_are_inferred_exactly() {
    let input = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2, 5])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let topk_output = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2, 3])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let topk_indices = LogicalTensorMeta {
        dtype: Some(DTypeId::I64.descriptor()),
        ..topk_output.clone()
    };
    ValidatedInvocation::<op::TopK>::validate(
        TopKAttributes {
            k: 3,
            axis: 1,
            largest: true,
            index_dtype: DTypeId::I64.descriptor(),
        },
        vec![input.clone()],
        vec![topk_output, topk_indices],
        crate::exec::ProofLevel::Dynamic,
    )
    .unwrap();

    let output = |extent| LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2, extent])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    ValidatedInvocation::<op::Chunk>::validate(
        ChunkAttributes { chunks: 2, axis: 1 },
        vec![input.clone()],
        vec![output(3), output(2)],
        crate::exec::ProofLevel::Dynamic,
    )
    .unwrap();
    assert!(matches!(
        ValidatedInvocation::<op::Chunk>::validate(
            ChunkAttributes { chunks: 2, axis: 1 },
            vec![input],
            vec![output(3)],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::OutputArity { .. })
    ));
}

#[test]
fn recurrent_and_empty_reduction_contracts_are_storage_free() {
    let tensor = |shape: Vec<usize>| LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&shape)),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    ValidatedInvocation::<op::Rnn>::validate(
        RecurrentAttributes {
            input_size: 4,
            hidden_size: 6,
            bias_ih: true,
            bias_hh: true,
        },
        vec![tensor(vec![2, 3, 4]), tensor(vec![2, 6])],
        vec![tensor(vec![2, 3, 6]), tensor(vec![2, 6])],
        crate::exec::ProofLevel::Dynamic,
    )
    .unwrap();

    let empty = tensor(vec![0, 4]);
    assert!(matches!(
        ValidatedInvocation::<op::MeanAll>::validate(
            NoAttributes,
            vec![empty.clone()],
            vec![tensor(Vec::new())],
            crate::exec::ProofLevel::Dynamic,
        ),
        Err(DescriptorError::InvalidAttribute {
            attribute: "shape",
            ..
        })
    ));
    ValidatedInvocation::<op::SumAll>::validate(
        NoAttributes,
        vec![empty],
        vec![tensor(Vec::new())],
        crate::exec::ProofLevel::Dynamic,
    )
    .unwrap();
}

#[test]
fn data_creation_rejects_non_exact_payload_byte_length() {
    let error = ValidatedInvocation::<op::TensorFromBytes>::infer_runtime(
        DataAttributes {
            shape: vec![2],
            dtype: DTypeId::F32.descriptor(),
            device: DeviceId::cpu(),
            payload: CreationPayload::Bytes { byte_len: 7 },
        },
        Vec::new(),
    )
    .expect_err("a payload shorter than the exact storage size must fail");

    assert_eq!(
        error,
        DescriptorError::PayloadByteLength {
            operation: OperationKind::TensorFromBytes,
            expected: 8,
            actual: 7,
        }
    );
}

#[test]
fn typed_data_creation_rejects_a_payload_dtype_mismatch() {
    let error = ValidatedInvocation::<op::TensorFromData>::infer_runtime(
        DataAttributes {
            shape: vec![2],
            dtype: DTypeId::F32.descriptor(),
            device: DeviceId::cpu(),
            payload: CreationPayload::Typed {
                byte_len: 8,
                dtype: DTypeId::I64.descriptor(),
            },
        },
        Vec::new(),
    )
    .expect_err("a typed payload must agree with the descriptor dtype");

    assert_eq!(
        error,
        DescriptorError::PayloadDTypeMismatch {
            operation: OperationKind::TensorFromData,
            expected: DTypeId::F32.descriptor(),
            actual: DTypeId::I64.descriptor(),
        }
    );
}

#[test]
fn data_dependent_output_shapes_are_not_inferred_from_metadata() {
    assert!(OutputRule::DataDependent.is_data_dependent());
    assert!(!OutputRule::Preserve.is_data_dependent());
}

/// A typed invocation must hand the backend a usable element count, not just a
/// proof level.
///
/// The neighbouring test pins `proof_level()`, which tells a backend how much
/// was settled. `static_numel()` is the value it can act on: the CUDA pointwise
/// path uses it to prove a packed kernel's ragged-tail branch unreachable. If
/// any link between `Shape::PROOF` and `Validated` dropped `S`, every kernel
/// would quietly take the general path and nothing would fail, so the end of the
/// chain is worth pinning separately from the middle.
#[test]
fn a_typed_invocation_carries_a_static_element_count_to_the_backend() {
    let created = CreationAttributes {
        shape: vec![2, 3],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
    };
    type Static23 = crate::shapes::DimCons<
        typenum::U2,
        crate::shapes::DimCons<typenum::U3, crate::shapes::Nil>,
    >;
    let sv = crate::shapes::ShapeValue::<Static23>::from_validated(
        <Static23 as crate::shapes::Shape>::resolve(((), ((), ()))).unwrap(),
    );

    let proven = ValidatedInvocation::<op::Zeros>::infer_typed(created.clone(), vec![], &sv)
        .expect("a static creation request is legal");
    let evidence = proven.validated().shape_evidence();
    assert_eq!(evidence.proof(), crate::exec::ProofLevel::Static);
    assert_eq!(
        evidence.static_numel(),
        Some(6),
        "2 x 3 must reach the backend as a count of 6"
    );
    assert_eq!(evidence.static_rank(), Some(2));

    // The same request with nothing known must offer no count to specialize on.
    let erased = ValidatedInvocation::<op::Zeros>::infer_runtime(created, vec![])
        .expect("a dynamic creation request is equally legal");
    let erased_evidence = erased.validated().shape_evidence();
    assert_eq!(erased_evidence.proof(), crate::exec::ProofLevel::Dynamic);
    assert_eq!(
        erased_evidence.static_numel(),
        None,
        "an unproven shape must never hand a backend a constant to bake in"
    );
}
