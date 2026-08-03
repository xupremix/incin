# FND-004 conformance summary

What this task proves about the canonical operation contract, and by which test.

## 1. One identity per operation

| Claim | Proof |
|---|---|
| Every executable identity is declared exactly once | `exec::catalog::tests::identities_and_names_occur_exactly_once` |
| No catalog row is a broad family | `operation_inventory::no_catalog_row_is_a_broad_family_identity` |
| All 142 legacy operation-family methods map to exactly one row | `operation_inventory::every_legacy_operation_method_has_exactly_one_catalog_row` |
| A renamed or deleted trait method cannot orphan a row | same test, reverse direction |
| Every stable family (creation, transfer, autograd, module, loss, optimizer, quantized, reduction, tensor) is represented | `operation_inventory::every_stable_operation_family_is_represented` |
| The reviewed inventory document matches the code | `operation_inventory::operation_inventory_document_matches_the_catalog` |
| The generated semantics document matches the code | `generated_operation_semantics::operation_semantics_matches_the_code_catalog` |

Catalog size: **174** exact operations.

## 2. Per-operand validation

The catalog's `accepted_ranks` is the *primary* operand's window.
`operand_ranks` overrides it per role, so widening an activation can never
widen a parameter.

| Role contract | Proof |
|---|---|
| Conv2d rank-1 bias against a rank-4 activation; activation-ranked bias refused | `convolution_bias_is_validated_by_role_not_by_the_activation_window` |
| Conv1d and conv_transpose2d bias; rank-2 weight refused | `conv1d_and_transposed_convolution_bias_use_the_same_role_contract` |
| Embedding rank-2 table vs. index batch; rank-0 indices refused | `embedding_indices_and_weight_have_separate_rank_contracts` |
| Batch-norm rank-1 affine/state against a ranked activation | `batch_norm_state_is_rank_one_against_a_ranked_activation` |
| Linear rank-2 weight and rank-1 bias | `linear_weight_and_bias_have_separate_rank_contracts` |
| Cross-entropy rank-2 logits vs. rank-1 integer targets | `cross_entropy_logits_and_targets_have_separate_rank_contracts` |
| Optimizer gradient and moment state validated against the parameter, before mutation | `optimizer_gradient_and_state_are_validated_against_the_parameter` |

`Conv2dExact` with a bias **could not validate at all** before this change: the
rank-1 bias was checked against the rank `3..=4` activation window.

## 3. Family defaults are defaults

Each row below overrides an incorrect family default; each is regression-tested
rather than asserted only in prose.

| Exception | Contract | Proof |
|---|---|---|
| Comparison / logical output dtype | Mask is encoded in the operand dtype; `DTypeId` has no boolean member, so `DTypeRule::Preserve` is the truthful rule and `BooleanResult` was removed as unreachable | `a_comparison_mask_keeps_the_operand_dtype` |
| Index dtype | `argmax`/`argmin`/`topk`/`argsort` must declare an **integer** index dtype | `index_producing_operations_reject_a_non_integer_index_dtype` |
| `topk` two-output split | Values keep the input dtype; indices take the index dtype | `topk_values_keep_the_input_dtype_while_indices_take_the_index_dtype` |
| Empty-domain identity | `sum`/`prod`/`cumsum` have an identity; `mean`/`max`/`min` do not | `empty_domain_behaviour_splits_within_the_reduction_family` |
| Unbiased estimator | Rejected on a domain of fewer than two elements (the correction divides by `n - 1`); the family rule only rejected an *empty* domain | `an_unbiased_estimate_rejects_a_single_element_domain` |
| Zero-output operations | Host extraction, `to_bytes`, and `backward` declare zero outputs and no gradient | `zero_output_gradient_and_determinism_exceptions_override_their_family` |
| Non-differentiable unary functions | `step`/`sign`/`floor`/`round` are `GradientRule::Undefined`, unlike the rest of `UnaryFloat` | same |
| Nondeterminism | random ops, `dropout`, `topk`, `argsort` | same |
| Device-changing transfer | `to_device`/`to_dtype` opt out of the same-device requirement | same |

## 4. Output inference is exact or fails closed

Output **device** and **dtype** are verified unconditionally against the typed
attributes and inputs.

Output **shape** now fails closed: when no inference branch produces an
expectation and the inputs are fully known, validation returns
`DescriptorError::MissingInference` instead of accepting the caller's shape.
Unknown input metadata still legitimately yields unknown output metadata.

| Claim | Proof |
|---|---|
| Known inputs never skip shape verification; unknown inputs stay unknown | `known_inputs_never_skip_output_shape_verification` |
| Every tensor-returning row has a reachable inference source | `every_tensor_returning_row_declares_an_inference_source` |
| Typed-inference rows are inferred or declare zero outputs | `every_typed_output_rule_is_fail_closed_or_exactly_inferred` |
| Metadata cannot be fabricated | `inferred_metadata_cannot_be_fabricated` |

## 5. Capability truth is exact

Family fallback is removed from `CapabilityRegistry`: a rule matches only when
`rule.operation == query.operation`.

| Claim | Proof |
|---|---|
| A family row never makes an exact query supported | `capability::tests::families_never_imply_exact_support`, and cross-backend in `capability_matrix::an_exact_query_never_resolves_through_a_broad_family_row` |
| Every advertised CPU row × layout really executes, with matching output metadata | `capability_matrix::generated_cpu_rows_match_real_execution_and_output_metadata` |
| Every advertised CPU dtype really executes | `capability_matrix::every_advertised_cpu_dtype_executes_its_registered_operation` |
| Every **unadvertised** layout returns the documented typed reason | `capability_matrix::an_unadvertised_exact_layout_returns_the_documented_typed_reason` |
| No registration resolves through a fallback implementation | `capability_matrix::every_registration_generates_supported_boundary_cases_without_fallback` |
| Every generated WGPU row matches real execution | `capability_matrix::every_generated_wgpu_row_matches_real_execution` |
| CUDA rows (hardware) | `capability_matrix::every_generated_cuda_row_matches_real_execution_on_hardware` — `#[ignore]`, no device present |

CPU strided support is **truthful**: `ReshapeExact` advertises `Contiguous`
natively plus `Strided` as `ImplementationKind::Composed`, and `MatMulExact`
advertises both. Both are proved by real execution
(`cpu_executor::reshape_descriptor_execution_materializes_a_strided_view`,
`cpu_executor::strided_view_descriptor_execution_matches_the_legacy_path`).
No capability row was narrowed to hide a gap, because no gap was found.

## 6. Backend identity coupling

All four native executors (CPU, CUDA, WGPU, Metal) and the shared binders in
`descriptor_bind.rs` now report the **exact** catalog identity in errors and in
capability queries. Previously WGPU and Metal reported the broad family, which
meant an execution refusal could not be matched against the row the descriptor
was validated against.

## 7. Storage-free capture

`CapturedDescriptor` serializes a descriptor with its identity and schema
outside the payload, so decoding as the wrong descriptor type fails closed. No
backend storage is referenced. Proof: `exec::catalog::tests::attribute_bearing_descriptor_round_trips_without_storage`
and the descriptor-schema suite.
