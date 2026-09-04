//! Typed dispatch with N outputs.
//!
//! `execute_shaped` used to take exactly one `&ShapeValue<S>`; an operation
//! inferring two or three outputs could not travel the typed path and had to
//! re-derive its geometry frontend-side (which is what `Tensor::topk` still
//! does, via untyped `execute` plus `Dense::from_parts`). `execute_shaped_n`
//! takes one expectation per output -- a single value or a tuple -- and
//! compares element-wise, so the descriptor cross-checks what inference
//! derived instead of trusting the frontend.
//!
//! The operation below declares three outputs with three dtypes (`f32`
//! values, `u32` positions, `f64` widened values) over one input, and every
//! rejection names its output index: a count mismatch, a wrong shape at
//! output 1, and a missing shape at output 1 each fail with the index
//! attached, not with the old unindexed single-output refusal.

#![cfg(feature = "cpu")]

extern crate incin_core as incin;

use std::borrow::Cow;

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
use incin_core::backend_authoring::{
    DescriptorError, Execute, ExecutionContext, ExecutionRequest, LogicalTensorMeta, Operation,
    OperationKey, ShapeBuf, execute_shaped_n,
};
use incin_core::exec::dispatch::CanonicalError;
use incin_core::prelude::{BackendError, DTypeId, DeviceId, s};
use incin_core::shapes::error::OperationKind;

type CpuBackend = CpuBackendImpl;

const N: usize = 4;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct TripleAttributes {
    count: usize,
}

#[derive(Debug, Clone)]
struct Triple;

fn invalid(attribute: &'static str, reason: &'static str) -> DescriptorError {
    DescriptorError::InvalidAttribute {
        operation: OperationKind::Pointwise,
        attribute,
        reason,
    }
}

fn expect_f32_vector(meta: &LogicalTensorMeta) -> Result<Vec<usize>, DescriptorError> {
    match meta.dtype {
        Some(actual) if actual == DTypeId::F32.descriptor() => {}
        Some(_) => return Err(invalid("values", "operand must be f32")),
        None => return Err(invalid("values", "operand element type is unknown")),
    }
    meta.shape
        .as_ref()
        .map(|shape| shape.dims().to_vec())
        .ok_or_else(|| invalid("values", "operand shape is unknown"))
}

impl Operation for Triple {
    type Attributes = TripleAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("incin.test"),
        name: Cow::Borrowed("triple"),
        version: 1,
    };

    /// One f32 input; three outputs sharing its shape with dtypes f32, u32
    /// and f64 respectively. Per-output dtypes are already representable in
    /// `infer_outputs` (TopK's own inference is indexed by position); what is
    /// new is carrying all three through the *typed* path.
    fn infer_outputs(
        attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> core::result::Result<Vec<LogicalTensorMeta>, DescriptorError> {
        if inputs.len() != 1 {
            return Err(invalid("inputs", "triple takes one operand"));
        }
        let dims = expect_f32_vector(&inputs[0])?;
        if dims.iter().product::<usize>() != attributes.count {
            return Err(invalid(
                "count",
                "attribute count must match the input size",
            ));
        }
        let output = |dtype: DTypeId| LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(&dims)),
            dtype: Some(dtype.descriptor()),
            device: Some(DeviceId::cpu()),
        };
        Ok(vec![
            output(DTypeId::F32),
            output(DTypeId::U32),
            output(DTypeId::F64),
        ])
    }
}

impl Execute<Triple> for CpuBackend {
    /// `(values, positions, widened values)`, each shaped like the input.
    type Output = (CpuStorage, CpuStorage, CpuStorage);

    fn execute(
        &self,
        request: ExecutionRequest<'_, Triple, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let input =
            request.inputs[0]
                .downcast_ref::<CpuStorage>()
                .ok_or(BackendError::InvalidInput {
                    operation: OperationKind::Pointwise,
                    reason: "triple expects CPU storage",
                })?;
        let dims = input.metadata().shape.dims().to_vec();
        let count = dims.iter().product::<usize>().max(1);
        let mut values = Vec::with_capacity(count);
        let mut positions = Vec::with_capacity(count);
        let mut widened = Vec::with_capacity(count);
        let mut index = vec![0usize; dims.len().max(1)];
        for flat in 0..count {
            let value = input.get(&index) as f32;
            values.push(value);
            positions.push(flat as u32);
            widened.push(value as f64);
            odometer(&mut index, &dims);
        }
        let storage = |buffer| {
            CpuStorage::try_from_contiguous(buffer, &dims).map_err(|error| {
                BackendError::Execution {
                    operation: OperationKind::Pointwise,
                    message: incin_core::prelude::ErrorMessage::new(error.to_string()),
                }
            })
        };
        Ok((
            storage(CpuBuffer::F32(values))?,
            storage(CpuBuffer::U32(positions))?,
            storage(CpuBuffer::F64(widened))?,
        ))
    }
}

/// Same single input, but the second inferred output carries no shape: the
/// typed custom path must refuse it naming output 1 rather than comparing
/// past it or failing without an index.
#[derive(Debug, Clone)]
struct TripleShapeless;

impl Operation for TripleShapeless {
    type Attributes = TripleAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("incin.test"),
        name: Cow::Borrowed("triple-shapeless"),
        version: 1,
    };

    fn infer_outputs(
        attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> core::result::Result<Vec<LogicalTensorMeta>, DescriptorError> {
        let mut outputs = Triple::infer_outputs(attributes, inputs)?;
        outputs[1].shape = None;
        Ok(outputs)
    }
}

impl Execute<TripleShapeless> for CpuBackend {
    type Output = (CpuStorage, CpuStorage, CpuStorage);

    fn execute(
        &self,
        _request: ExecutionRequest<'_, TripleShapeless, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        // Unreachable: inference rejects the shapeless output before any
        // backend runs. Returning an error rather than panicking keeps even
        // the impossible path structured.
        Err(BackendError::InvalidInput {
            operation: OperationKind::Pointwise,
            reason: "triple-shapeless must be refused at inference",
        })
    }
}

fn odometer(index: &mut [usize], dims: &[usize]) {
    for (i, extent) in index.iter_mut().zip(dims.iter()).rev() {
        *i += 1;
        if *i < *extent {
            return;
        }
        *i = 0;
    }
}

fn context() -> ExecutionContext<CpuBackend> {
    ExecutionContext::new(CpuBackend::default())
}

fn input_storage() -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), [N])
        .expect("test input construction")
}

fn input_handle(storage: &CpuStorage) -> incin_core::exec::TensorHandle<'_> {
    incin_core::exec::TensorHandle::from_storage::<CpuBackend, f32, _>(storage)
}

/// The three caller-held proofs, one per output. A named alias because the
/// triple spells the same shape type three times and clippy's complexity
/// lint refuses the inline form.
type TripleExpected = (
    incin_core::shapes::ShapeValue<s![4]>,
    incin_core::shapes::ShapeValue<s![4]>,
    incin_core::shapes::ShapeValue<s![4]>,
);

fn expected_triple() -> TripleExpected {
    use incin_core::shapes::ShapeValue;
    (
        ShapeValue::try_new(ShapeBuf::from_slice(&[N])).unwrap(),
        ShapeValue::try_new(ShapeBuf::from_slice(&[N])).unwrap(),
        ShapeValue::try_new(ShapeBuf::from_slice(&[N])).unwrap(),
    )
}

#[test]
fn typed_three_outputs_with_per_output_dtypes() {
    let backend = context();
    let input = input_storage();
    let handles = [input_handle(&input)];
    let expected = expected_triple();
    let (values, positions, widened) = execute_shaped_n::<Triple, CpuBackend, _>(
        &backend,
        TripleAttributes { count: N },
        &handles,
        &expected,
    )
    .expect("three-output typed dispatch");

    // Geometry and dtypes per output: output 1 is u32 and output 2 is f64,
    // not more f32.
    assert_eq!(values.metadata().shape.dims(), [N]);
    assert_eq!(positions.metadata().shape.dims(), [N]);
    assert_eq!(widened.metadata().shape.dims(), [N]);
    assert_eq!(
        values.metadata().dtype,
        DTypeId::F32.descriptor(),
        "output 0 carries f32"
    );
    assert_eq!(
        positions.metadata().dtype,
        DTypeId::U32.descriptor(),
        "output 1 carries u32"
    );
    assert_eq!(
        widened.metadata().dtype,
        DTypeId::F64.descriptor(),
        "output 2 carries f64"
    );
    for i in 0..N {
        assert_eq!(values.get(&[i]), (i + 1) as f64, "output 0 value {i}");
        assert_eq!(positions.get(&[i]), i as f64, "output 1 value {i}");
        assert_eq!(widened.get(&[i]), (i + 1) as f64, "output 2 value {i}");
    }
}

#[test]
fn typed_three_outputs_rejects_wrong_count() {
    let backend = context();
    let input = input_storage();
    let handles = [input_handle(&input)];
    // Two proofs for three inferred outputs.
    let expected = (
        incin_core::shapes::ShapeValue::<s![4]>::try_new(ShapeBuf::from_slice(&[N])).unwrap(),
        incin_core::shapes::ShapeValue::<s![4]>::try_new(ShapeBuf::from_slice(&[N])).unwrap(),
    );
    assert!(
        matches!(
            execute_shaped_n::<Triple, CpuBackend, _>(
                &backend,
                TripleAttributes { count: N },
                &handles,
                &expected
            ),
            Err(CanonicalError::Descriptor(DescriptorError::OutputArity {
                actual: 3,
                ..
            }))
        ),
        "three inferred outputs against two proofs must report the count"
    );
}

#[test]
fn typed_three_outputs_rejects_wrong_shape_at_index_1() {
    let backend = context();
    let input = input_storage();
    let handles = [input_handle(&input)];
    // Only the middle proof disagrees ([5] for an inferred [4]).
    let expected = (
        incin_core::shapes::ShapeValue::<s![4]>::try_new(ShapeBuf::from_slice(&[N])).unwrap(),
        incin_core::shapes::ShapeValue::<s![5]>::try_new(ShapeBuf::from_slice(&[5])).unwrap(),
        incin_core::shapes::ShapeValue::<s![4]>::try_new(ShapeBuf::from_slice(&[N])).unwrap(),
    );
    assert!(
        matches!(
            execute_shaped_n::<Triple, CpuBackend, _>(
                &backend,
                TripleAttributes { count: N },
                &handles,
                &expected
            ),
            Err(CanonicalError::Descriptor(
                DescriptorError::MetadataMismatch {
                    output: 1,
                    field: "shape",
                    ..
                }
            ))
        ),
        "a wrong middle shape must name output 1"
    );
}

#[test]
fn typed_three_outputs_rejects_missing_shape_at_index_1() {
    let backend = context();
    let input = input_storage();
    let handles = [input_handle(&input)];
    let expected = expected_triple();
    assert!(
        matches!(
            execute_shaped_n::<TripleShapeless, CpuBackend, _>(
                &backend,
                TripleAttributes { count: N },
                &handles,
                &expected
            ),
            Err(CanonicalError::Descriptor(
                DescriptorError::MetadataMismatch {
                    output: 1,
                    field: "shape",
                    ..
                }
            ))
        ),
        "a shapeless middle output must name output 1"
    );
}
