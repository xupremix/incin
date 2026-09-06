//! Custom-operation training on CUDA: the hardware-executed twin of the CPU
//! and WGPU fixtures (#82 tracks the execution runner).
//!
//! Every test is `#[ignore]`d: reaching one is an explicit request for the
//! hardware run, and the `require_cuda` assert fails loudly without a device
//! rather than skipping green. A custom `square` operation records through
//! `cuda::tape_record`; forward and recipe go through host round-trips,
//! which is the honest test-only shape — production recipes stay in-kernel.
#![cfg(feature = "cuda")]

extern crate incin_core as incin;

use incin_backends::cuda::{CudaBackendImpl, tape_depth, tape_record, tape_record_with};
use incin_core::backend_authoring::{
    AutogradBackend, DescriptorError, Execute, ExecutionRequest, HostInterop, LogicalTensorMeta,
    Operation, OperationIdentity, OperationKey, SupportLevel, TapeNode, TapeStorage, TensorId,
};
use incin_core::exec::catalog::NoAttributes;
use incin_core::exec::{
    CapabilityQuery, ExecutionContext, GradMode, TensorHandle, UnsupportedReason,
};
use incin_core::prelude::{BackendError, CudaN, DTypeId, DeviceId};
use incin_core::typenum::U0;

type TestBackend = CudaBackendImpl<CudaN<U0>>;
type TestStorage = <TestBackend as incin_core::backend_authoring::StorageBackend>::Storage<f32>;

/// Aborts unless a CUDA device is present.
///
/// Every caller is `#[ignore]`d, so reaching one is a deliberate request for
/// the hardware run, and returning early there reports `ok` for a test that
/// launched nothing.
fn require_cuda() {
    assert!(
        TestBackend::from_bytes::<f32>(
            bytemuck::cast_slice(&[1.0f32]),
            &[1],
            DTypeId::F32.into(),
            &DeviceId::cuda(0),
        )
        .is_ok(),
        "no CUDA device, but this test is #[ignore]d -- running it is an explicit request for hardware."
    );
}

fn upload(values: &[f32], shape: &[usize]) -> TestStorage {
    TestBackend::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32.into(),
        &DeviceId::cuda(0),
    )
    .expect("uploading the operand must succeed")
}

fn download(storage: &TestStorage) -> Vec<f32> {
    let bytes = TestBackend::to_bytes::<f32>(storage).expect("readback must succeed");
    bytemuck::cast_slice(&bytes).to_vec()
}

/// `y = x^2`, elementwise, `f32` only — the CUDA twin of the CPU fixture.
#[derive(Debug, Clone)]
struct CudaSquare;

impl Operation for CudaSquare {
    type Attributes = NoAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("cuda_square"),
        version: 1,
    };

    fn infer_outputs(
        _attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(inputs.first().cloned().into_iter().collect())
    }
}

impl Execute<CudaSquare> for TestBackend {
    type Output = TestStorage;

    fn supports_custom(&self, query: &incin_core::exec::CapabilityQuery) -> SupportLevel {
        assert_eq!(query.operation, OperationIdentity::Custom(CudaSquare::KEY));
        if query.dtype != DTypeId::F32.descriptor() {
            SupportLevel::Unsupported(UnsupportedReason::CustomOperation {
                operation: CudaSquare::KEY,
            })
        } else {
            SupportLevel::Native
        }
    }

    fn execute(
        &self,
        request: ExecutionRequest<'_, CudaSquare, Self>,
    ) -> Result<Self::Output, BackendError> {
        use incin_core::prelude::OperationKind;
        let x = request
            .inputs
            .first()
            .and_then(|input| input.downcast_ref::<TestStorage>())
            .cloned()
            .ok_or(BackendError::InvalidInput {
                operation: OperationKind::Pointwise,
                reason: "cuda square requires one CUDA input",
            })?;
        if x.metadata().dtype != DTypeId::F32.descriptor() {
            return Err(BackendError::InvalidInput {
                operation: OperationKind::Pointwise,
                reason: "cuda square kernel holds f32 only",
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
                    .zip(download(grad_out).into_iter())
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
    incin_core::backend_authoring::execute::<CudaSquare, _>(ctx, NoAttributes, &[handle])
        .expect("square executes on f32 CUDA input")
}

#[test]
#[ignore = "requires CUDA hardware"]
fn downstream_cuda_custom_operation_trains_end_to_end() {
    require_cuda();
    let ctx = ExecutionContext::new(TestBackend::new());
    let x = upload(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let x_id: TensorId = x.id();

    let loss = square_forward(&ctx, &x);
    assert_eq!(download(&loss), vec![1.0, 4.0, 9.0, 16.0]);

    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs on CUDA");
    let gx = grads.get(x_id).expect("custom input receives a gradient");
    assert_eq!(download(gx), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn downstream_cuda_custom_operation_records_nothing_under_no_grad() {
    require_cuda();
    let ctx = ExecutionContext::new(TestBackend::new());
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
