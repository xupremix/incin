//! Custom-operation training on WGPU: the adapter-executed twin of
//! `crates/incin-core/tests/custom_training.rs`.
//!
//! A custom `square` operation (`y = x^2`) records through
//! `wgpu::tape_record`; the forward kernel and the recipe both go through
//! host round-trips (reads via `float_to_vec1`, writes via `from_bytes`),
//! which is the honest test-only shape — production recipes stay in-kernel.
//! Requires a WGPU adapter; fails loudly without one, per the other WGPU
//! suites.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::{WgpuBackendImpl, tape_depth, tape_record, tape_record_with};
use incin_core::backend_authoring::{
    AutogradBackend, DescriptorError, Execute, ExecutionRequest, HostInterop, HostReadback,
    LogicalTensorMeta, Operation, OperationIdentity, OperationKey, SupportLevel, TapeNode,
    TapeStorage, TensorId,
};
use incin_core::exec::catalog::NoAttributes;
use incin_core::exec::{
    CapabilityQuery, ExecutionContext, GradMode, TensorHandle, UnsupportedReason,
};
use incin_core::prelude::{BackendError, DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as incin_core::backend_authoring::StorageBackend>::Storage<f32>;

fn require_wgpu() {
    assert!(
        <TestBackend as HostInterop>::from_bytes::<f32>(
            &[0u8; 4],
            &[1],
            DTypeId::F32.descriptor(),
            &DeviceId::wgpu(0),
        )
        .is_ok(),
        "no WGPU adapter, but the `wgpu` feature is enabled"
    );
}

fn upload(values: &[f32], shape: &[usize]) -> TestStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading the operand must succeed")
}

fn download(storage: &TestStorage) -> Vec<f32> {
    <TestBackend as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading back must succeed")
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

/// `y = x^2`, elementwise, `f32` only — the WGPU twin of the CPU fixture.
#[derive(Debug, Clone)]
struct WgpuSquare;

impl Operation for WgpuSquare {
    type Attributes = NoAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("wgpu_square"),
        version: 1,
    };

    fn infer_outputs(
        _attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(inputs.first().cloned().into_iter().collect())
    }
}

impl Execute<WgpuSquare> for TestBackend {
    type Output = TestStorage;

    fn supports_custom(&self, query: &CapabilityQuery) -> SupportLevel {
        assert_eq!(query.operation, OperationIdentity::Custom(WgpuSquare::KEY));
        if query.dtype != DTypeId::F32.descriptor() {
            SupportLevel::Unsupported(UnsupportedReason::CustomOperation {
                operation: WgpuSquare::KEY,
            })
        } else {
            SupportLevel::Native
        }
    }

    fn execute(
        &self,
        request: ExecutionRequest<'_, WgpuSquare, Self>,
    ) -> Result<Self::Output, BackendError> {
        use incin_core::prelude::OperationKind;
        let x = request
            .inputs
            .first()
            .and_then(|input| input.downcast_ref::<TestStorage>())
            .cloned()
            .ok_or(BackendError::InvalidInput {
                operation: OperationKind::Pointwise,
                reason: "wgpu square requires one WGPU input",
            })?;
        if x.metadata().dtype != DTypeId::F32.descriptor() {
            return Err(BackendError::InvalidInput {
                operation: OperationKind::Pointwise,
                reason: "wgpu square kernel holds f32 only",
            });
        }
        let dims = x.metadata().shape.dims().to_vec();
        let values: Vec<f32> = download(&x).into_iter().map(|v| v * v).collect();
        let out = upload(&values, &dims);
        let x_saved = x.clone();
        let node = TapeNode {
            output_id: out.id(),
            input_ids: vec![x.id()],
            backward: Box::new(move |grad_out: &TestStorage| {
                let dims = x_saved.metadata().shape.dims().to_vec();
                let grads: Vec<f32> = download(&x_saved)
                    .into_iter()
                    .zip(download(grad_out))
                    .map(|(x, g)| 2.0 * x * g)
                    .collect();
                Ok(vec![upload(&grads, &dims)])
            }),
        };
        tape_record(node);
        Ok(out)
    }
}

fn square_forward(ctx: &ExecutionContext<TestBackend>, x: &TestStorage) -> TestStorage {
    let handle = TensorHandle::from_storage::<TestBackend, f32, _>(x);
    incin_core::backend_authoring::execute::<WgpuSquare, _>(ctx, NoAttributes, &[handle])
        .expect("square executes on f32 WGPU input")
}

#[test]
fn downstream_wgpu_custom_operation_trains_end_to_end() {
    require_wgpu();
    let ctx = ExecutionContext::new(TestBackend::default());
    let x = upload(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let x_id: TensorId = x.id();

    let loss = square_forward(&ctx, &x);
    assert_eq!(download(&loss), vec![1.0, 4.0, 9.0, 16.0]);

    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs on WGPU");
    let gx = grads.get(x_id).expect("custom input receives a gradient");
    assert_eq!(download(gx), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn downstream_wgpu_custom_operation_records_nothing_under_no_grad() {
    require_wgpu();
    let ctx = ExecutionContext::new(TestBackend::default());
    let x = upload(&[1.0, 2.0], &[2]);
    let before = tape_depth();
    GradMode::Disabled.scope(|| {
        square_forward(&ctx, &x);
        tape_record_with(|| panic!("record_with built an entry under NoGrad"));
    });
    assert_eq!(
        tape_depth(),
        before,
        "a NoGrad custom forward must record nothing"
    );
}

#[test]
fn downstream_wgpu_custom_operation_refuses_dtypes_it_does_not_support() {
    require_wgpu();
    // `supports_custom` decides on the query descriptor before any kernel
    // runs; drive it directly since WGPU cannot even store f16.
    let query = CapabilityQuery {
        operation: OperationIdentity::Custom(WgpuSquare::KEY),
        dtype: DTypeId::F16.descriptor(),
        layout: incin_core::exec::LayoutClass::Contiguous,
        rank: 1,
        training: false,
        math_mode: incin_core::exec::MathMode::Precise,
    };
    assert!(matches!(
        <TestBackend as Execute<WgpuSquare>>::supports_custom(&TestBackend::default(), &query),
        SupportLevel::Unsupported(UnsupportedReason::CustomOperation { .. })
    ));
}
