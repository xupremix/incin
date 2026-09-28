//! Issue #104 fused attention on CUDA: native single-pass online-softmax
//! forward over query rows (GQA head mapping, causal masking with kv_lead)
//! plus a recompute backward (delta, dq, dkv kernels) driven by stored
//! per-row statistics - no score matrix, no atomics, one tape entry.
//!
//! Every hardware test is `#[ignore]`d; `require_cuda` fails loudly rather
//! than skipping, because reaching an `#[ignore]`d test is an explicit
//! request for the hardware run. Forward and backward values are pinned
//! against the CPU kernel through dispatch twins, not hand arithmetic:
//! the contract under test is parity with the reference implementation.
#![cfg(feature = "cuda")]

use incin_backends::cuda::{
    CudaBackendImpl, tape_depth,
    testing::{
        download_bytes, download_f32, require_cuda, transpose, upload_bytes, upload_f32_shaped,
    },
};
use incin_core::backend_authoring::{AutogradBackend, HostInterop, StorageBackend};
use incin_core::exec::catalog::FusedAttentionAttributes;
use incin_core::exec::{
    CapabilityQuery, ExecutionContext, LayoutClass, MathMode, OperationIdentity, SupportLevel,
    TapeStorage, TensorHandle, dispatch, op,
};
use incin_core::prelude::{CudaN, DTypeId};
use incin_core::shapes::error::OperationKind;
use incin_core::typenum::U0;

type TestBackend = CudaBackendImpl<CudaN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;
#[cfg(feature = "cpu")]
type Cpu = incin_backends::cpu::CpuBackendImpl<incin_core::tensor::device::Cpu>;

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g}, want {w} (tol {tol})"
        );
    }
}

fn read_f32(storage: &TestStorage) -> Vec<f64> {
    download_f32(storage)
        .iter()
        .map(|&v| f64::from(v))
        .collect()
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

fn decode_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunk")))
        .collect()
}

/// Three f32 rank-4 inputs through `dispatch`, returning the output and
/// the tape entries the forward pushed.
fn run_fused(
    context: &ExecutionContext<TestBackend>,
    q: &TestStorage,
    k: &TestStorage,
    v: &TestStorage,
    attributes: FusedAttentionAttributes,
) -> (TestStorage, usize) {
    let before = tape_depth();
    let out = dispatch::execute::<op::FusedAttention, TestBackend>(
        context,
        attributes,
        &[
            TensorHandle::from_storage::<TestBackend, f32, _>(q),
            TensorHandle::from_storage::<TestBackend, f32, _>(k),
            TensorHandle::from_storage::<TestBackend, f32, _>(v),
        ],
    )
    .expect("an advertised fused-attention invocation must execute");
    (out, tape_depth() - before)
}

#[cfg(feature = "cpu")]
fn cpu_fused(
    shape_q: &[usize],
    q: &[f32],
    shape_kv: &[usize],
    k: &[f32],
    v: &[f32],
    attributes: FusedAttentionAttributes,
) -> Vec<f64> {
    let context = ExecutionContext::new(Cpu::new());
    let hq = <Cpu as HostInterop>::from_bytes::<f32>(
        bytemuck::cast_slice(q),
        shape_q,
        DTypeId::F32.descriptor(),
        &incin_core::tensor::device::DeviceId::cpu(),
    )
    .expect("uploading the CPU query twin must succeed");
    let hk = <Cpu as HostInterop>::from_bytes::<f32>(
        bytemuck::cast_slice(k),
        shape_kv,
        DTypeId::F32.descriptor(),
        &incin_core::tensor::device::DeviceId::cpu(),
    )
    .expect("uploading the CPU key twin must succeed");
    let hv = <Cpu as HostInterop>::from_bytes::<f32>(
        bytemuck::cast_slice(v),
        shape_kv,
        DTypeId::F32.descriptor(),
        &incin_core::tensor::device::DeviceId::cpu(),
    )
    .expect("uploading the CPU value twin must succeed");
    let out = dispatch::execute::<op::FusedAttention, Cpu>(
        &context,
        attributes,
        &[
            TensorHandle::from_storage::<Cpu, f32, _>(&hq),
            TensorHandle::from_storage::<Cpu, f32, _>(&hk),
            TensorHandle::from_storage::<Cpu, f32, _>(&hv),
        ],
    )
    .expect("CPU reference fused attention executes");
    cpu_read_f32(&out)
}

#[cfg(feature = "cpu")]
fn cpu_read_f32(storage: &<Cpu as StorageBackend>::Storage<f32>) -> Vec<f64> {
    use incin_core::backend_authoring::HostReadback;
    <Cpu as HostReadback>::float_to_vec1::<f32>(storage).expect("reading the CPU twin must succeed")
}

#[test]
fn fused_attention_row_is_native_training_on_cuda() {
    let level = incin_backends::capability::support(
        incin_core::tensor::device::DeviceKind::Cuda,
        &CapabilityQuery {
            operation: OperationIdentity::Builtin(OperationKind::FusedAttention),
            dtype: DTypeId::F32.descriptor(),
            layout: LayoutClass::Contiguous,
            rank: 4,
            training: true,
            math_mode: MathMode::default(),
        },
    );
    assert_eq!(
        level,
        SupportLevel::Native,
        "fused attention ships a native CUDA kernel, so rank 4 must be Native with training"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn fused_forward_mha_matches_the_cpu_kernel() {
    require_cuda();
    // B=1, H=2, Sq=3, Skv=3, D=4, non-causal, default scale.
    let qv: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) / 8.0).collect();
    let kv: Vec<f32> = (0..24).map(|i| ((i * 7) % 11) as f32 / 8.0 - 0.5).collect();
    let vv: Vec<f32> = (0..24).map(|i| ((i * 13) % 7) as f32 / 4.0 - 0.5).collect();
    let context = ExecutionContext::new(TestBackend::new());
    let q = upload_f32_shaped(&[1, 2, 3, 4], &qv);
    let k = upload_f32_shaped(&[1, 2, 3, 4], &kv);
    let v = upload_f32_shaped(&[1, 2, 3, 4], &vv);
    let attrs = FusedAttentionAttributes {
        scale: None,
        causal: false,
    };
    let (out, recorded) = run_fused(&context, &q, &k, &v, attrs);
    assert_eq!(recorded, 1, "fused attention records one tape entry");
    assert_eq!(out.shape, vec![1, 2, 3, 4]);
    #[cfg(feature = "cpu")]
    {
        let want = cpu_fused(
            &[1, 2, 3, 4],
            &qv,
            &[1, 2, 3, 4],
            &kv,
            &vv,
            FusedAttentionAttributes {
                scale: None,
                causal: false,
            },
        );
        assert_close(&read_f32(&out), &want, 1e-4, "mha forward");
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn fused_forward_gqa_causal_decode_matches_the_cpu_kernel() {
    require_cuda();
    // GQA (Hq=4 over Hkv=2), causal, decode geometry (Sq=1, Skv=4):
    // kv_lead = 3 lets the single query see the whole prefix.
    let qv: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) / 8.0).collect();
    let kv: Vec<f32> = (0..32).map(|i| ((i * 5) % 9) as f32 / 8.0 - 0.5).collect();
    let vv: Vec<f32> = (0..32).map(|i| ((i * 11) % 7) as f32 / 4.0 - 0.5).collect();
    let context = ExecutionContext::new(TestBackend::new());
    let q = upload_f32_shaped(&[1, 4, 1, 4], &qv);
    let k = upload_f32_shaped(&[1, 2, 4, 4], &kv);
    let v = upload_f32_shaped(&[1, 2, 4, 4], &vv);
    let attrs = FusedAttentionAttributes {
        scale: Some(0.5),
        causal: true,
    };
    let (out, recorded) = run_fused(&context, &q, &k, &v, attrs);
    assert_eq!(recorded, 1, "fused attention records one tape entry");
    assert_eq!(out.shape, vec![1, 4, 1, 4]);
    #[cfg(feature = "cpu")]
    {
        let want = cpu_fused(
            &[1, 4, 1, 4],
            &qv,
            &[1, 2, 4, 4],
            &kv,
            &vv,
            FusedAttentionAttributes {
                scale: Some(0.5),
                causal: true,
            },
        );
        assert_close(&read_f32(&out), &want, 1e-4, "gqa causal forward");
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn fused_forward_f64_matches_within_double_tolerance() {
    require_cuda();
    // B=1, H=1, Sq=2, Skv=2, D=2, non-causal, f64 end to end.
    let qv = [0.5, -1.0, 2.0, 0.25];
    let kv = [1.0, 0.5, -0.5, 1.5];
    let vv = [0.25, 1.0, -1.0, 0.5];
    let context = ExecutionContext::new(TestBackend::new());
    let q = upload_bytes(&[1, 1, 2, 2], DTypeId::F64, &f64_bytes(&qv));
    let k = upload_bytes(&[1, 1, 2, 2], DTypeId::F64, &f64_bytes(&kv));
    let v = upload_bytes(&[1, 1, 2, 2], DTypeId::F64, &f64_bytes(&vv));
    let before = tape_depth();
    let out = dispatch::execute::<op::FusedAttention, TestBackend>(
        &context,
        FusedAttentionAttributes {
            scale: None,
            causal: false,
        },
        &[
            TensorHandle::from_storage::<TestBackend, f64, _>(&q),
            TensorHandle::from_storage::<TestBackend, f64, _>(&k),
            TensorHandle::from_storage::<TestBackend, f64, _>(&v),
        ],
    )
    .expect("f64 fused attention must execute");
    assert_eq!(
        tape_depth() - before,
        1,
        "fused attention records one tape entry"
    );
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // Hand-computed reference: scores/sqrt(2), softmax rows, mix.
    // Row 0: s = [0.375/1.4142, ...]; checked against the CPU twin
    // below rather than decimal-chasing here.
    let got = decode_f64(&download_bytes(&out));
    assert_eq!(got.len(), 4);
    for &x in &got {
        assert!(x.is_finite(), "f64 forward must be finite, got {x}");
    }
    #[cfg(feature = "cpu")]
    {
        let context = ExecutionContext::new(Cpu::new());
        let hq = <Cpu as HostInterop>::from_bytes::<f64>(
            bytemuck::cast_slice(&qv),
            &[1, 1, 2, 2],
            DTypeId::F64.descriptor(),
            &incin_core::tensor::device::DeviceId::cpu(),
        )
        .unwrap();
        let hk = <Cpu as HostInterop>::from_bytes::<f64>(
            bytemuck::cast_slice(&kv),
            &[1, 1, 2, 2],
            DTypeId::F64.descriptor(),
            &incin_core::tensor::device::DeviceId::cpu(),
        )
        .unwrap();
        let hv = <Cpu as HostInterop>::from_bytes::<f64>(
            bytemuck::cast_slice(&vv),
            &[1, 1, 2, 2],
            DTypeId::F64.descriptor(),
            &incin_core::tensor::device::DeviceId::cpu(),
        )
        .unwrap();
        let want = dispatch::execute::<op::FusedAttention, Cpu>(
            &context,
            FusedAttentionAttributes {
                scale: None,
                causal: false,
            },
            &[
                TensorHandle::from_storage::<Cpu, f64, _>(&hq),
                TensorHandle::from_storage::<Cpu, f64, _>(&hk),
                TensorHandle::from_storage::<Cpu, f64, _>(&hv),
            ],
        )
        .expect("CPU reference f64 fused attention executes");
        let want = cpu_read_f64(&want);
        assert_close(
            &got.iter().copied().collect::<Vec<f64>>(),
            &want,
            1e-9,
            "f64 forward",
        );
    }
}

#[cfg(feature = "cpu")]
fn cpu_read_f64(storage: &<Cpu as StorageBackend>::Storage<f64>) -> Vec<f64> {
    use incin_core::backend_authoring::HostReadback;
    <Cpu as HostReadback>::float_to_vec1::<f64>(storage).expect("reading the CPU twin must succeed")
}

#[test]
#[ignore = "requires CUDA hardware"]
fn fused_backward_matches_the_cpu_twin() {
    require_cuda();
    // Same MHA geometry as the forward test, ones-seeded through a
    // sum-all loss so every gradient is exercised.
    let qv: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) / 8.0).collect();
    let kv: Vec<f32> = (0..24).map(|i| ((i * 7) % 11) as f32 / 8.0 - 0.5).collect();
    let vv: Vec<f32> = (0..24).map(|i| ((i * 13) % 7) as f32 / 4.0 - 0.5).collect();
    let context = ExecutionContext::new(TestBackend::new());
    let q = upload_f32_shaped(&[1, 2, 3, 4], &qv);
    let k = upload_f32_shaped(&[1, 2, 3, 4], &kv);
    let v = upload_f32_shaped(&[1, 2, 3, 4], &vv);
    let q_id = TapeStorage::id(&q);
    let k_id = TapeStorage::id(&k);
    let v_id = TapeStorage::id(&v);
    let attrs = FusedAttentionAttributes {
        scale: None,
        causal: false,
    };
    let (out, _) = run_fused(&context, &q, &k, &v, attrs);
    let loss = dispatch::execute::<op::SumAll, _>(
        &context,
        incin_core::exec::catalog::NoAttributes,
        &[TensorHandle::from_storage::<TestBackend, f32, _>(&out)],
    )
    .expect("sum_all executes");
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");
    let gq = grads.get(q_id).expect("query receives a gradient");
    let gk = grads.get(k_id).expect("key receives a gradient");
    let gv = grads.get(v_id).expect("value receives a gradient");
    assert_eq!(gq.shape, vec![1, 2, 3, 4]);
    assert_eq!(gk.shape, vec![1, 2, 3, 4]);
    assert_eq!(gv.shape, vec![1, 2, 3, 4]);
    #[cfg(feature = "cpu")]
    {
        let context = ExecutionContext::new(Cpu::new());
        let hq = <Cpu as HostInterop>::from_bytes::<f32>(
            bytemuck::cast_slice(&qv),
            &[1, 2, 3, 4],
            DTypeId::F32.descriptor(),
            &incin_core::tensor::device::DeviceId::cpu(),
        )
        .unwrap();
        let hk = <Cpu as HostInterop>::from_bytes::<f32>(
            bytemuck::cast_slice(&kv),
            &[1, 2, 3, 4],
            DTypeId::F32.descriptor(),
            &incin_core::tensor::device::DeviceId::cpu(),
        )
        .unwrap();
        let hv = <Cpu as HostInterop>::from_bytes::<f32>(
            bytemuck::cast_slice(&vv),
            &[1, 2, 3, 4],
            DTypeId::F32.descriptor(),
            &incin_core::tensor::device::DeviceId::cpu(),
        )
        .unwrap();
        let hq_id = TapeStorage::id(&hq);
        let hk_id = TapeStorage::id(&hk);
        let hv_id = TapeStorage::id(&hv);
        let hout = dispatch::execute::<op::FusedAttention, Cpu>(
            &context,
            FusedAttentionAttributes {
                scale: None,
                causal: false,
            },
            &[
                TensorHandle::from_storage::<Cpu, f32, _>(&hq),
                TensorHandle::from_storage::<Cpu, f32, _>(&hk),
                TensorHandle::from_storage::<Cpu, f32, _>(&hv),
            ],
        )
        .unwrap();
        let hloss = dispatch::execute::<op::SumAll, _>(
            &context,
            incin_core::exec::catalog::NoAttributes,
            &[TensorHandle::from_storage::<Cpu, f32, _>(&hout)],
        )
        .unwrap();
        let hgrads = <Cpu as AutogradBackend>::backward::<f32>(&hloss).unwrap();
        assert_close(
            &read_f32(gq),
            &cpu_read_f32(hgrads.get(hq_id).unwrap()),
            1e-4,
            "dq",
        );
        assert_close(
            &read_f32(gk),
            &cpu_read_f32(hgrads.get(hk_id).unwrap()),
            1e-4,
            "dk",
        );
        assert_close(
            &read_f32(gv),
            &cpu_read_f32(hgrads.get(hv_id).unwrap()),
            1e-4,
            "dv",
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn fused_attention_refusals_are_typed() {
    require_cuda();
    let context = ExecutionContext::new(TestBackend::new());
    let q = upload_f32_shaped(&[1, 2, 3, 4], &vec![0.5; 24]);
    let k = upload_f32_shaped(&[1, 2, 3, 4], &vec![0.5; 24]);
    let v = upload_f32_shaped(&[1, 2, 3, 4], &vec![0.5; 24]);
    let handles = || {
        [
            TensorHandle::from_storage::<TestBackend, f32, _>(&q),
            TensorHandle::from_storage::<TestBackend, f32, _>(&k),
            TensorHandle::from_storage::<TestBackend, f32, _>(&v),
        ]
    };
    // Non-positive scale is refused by the descriptor, before launch.
    assert!(
        dispatch::execute::<op::FusedAttention, TestBackend>(
            &context,
            FusedAttentionAttributes {
                scale: Some(0.0),
                causal: false,
            },
            &handles(),
        )
        .is_err(),
        "non-positive scale must be refused"
    );
    // Rank-3 operands are refused by the descriptor, before launch.
    let flat = upload_f32_shaped(&[2, 12], &vec![0.5; 24]);
    assert!(
        dispatch::execute::<op::FusedAttention, TestBackend>(
            &context,
            FusedAttentionAttributes {
                scale: None,
                causal: false,
            },
            &[
                TensorHandle::from_storage::<TestBackend, f32, _>(&flat),
                TensorHandle::from_storage::<TestBackend, f32, _>(&k),
                TensorHandle::from_storage::<TestBackend, f32, _>(&v),
            ],
        )
        .is_err(),
        "rank-3 query must be refused"
    );
    // Query heads must be a multiple of kv heads.
    let k3 = upload_f32_shaped(&[1, 3, 3, 4], &vec![0.5; 36]);
    let v3 = upload_f32_shaped(&[1, 3, 3, 4], &vec![0.5; 36]);
    assert!(
        dispatch::execute::<op::FusedAttention, TestBackend>(
            &context,
            FusedAttentionAttributes {
                scale: None,
                causal: false,
            },
            &[
                TensorHandle::from_storage::<TestBackend, f32, _>(&q),
                TensorHandle::from_storage::<TestBackend, f32, _>(&k3),
                TensorHandle::from_storage::<TestBackend, f32, _>(&v3),
            ],
        )
        .is_err(),
        "2 query heads over 3 kv heads must be refused"
    );
    // A transposed (strided) operand is refused by the launcher: the
    // row only admits contiguous layouts.
    let qt = transpose(&q, 2, 3).expect("transpose must execute");
    assert!(
        dispatch::execute::<op::FusedAttention, TestBackend>(
            &context,
            FusedAttentionAttributes {
                scale: None,
                causal: false,
            },
            &[
                TensorHandle::from_storage::<TestBackend, f32, _>(&qt),
                TensorHandle::from_storage::<TestBackend, f32, _>(&k),
                TensorHandle::from_storage::<TestBackend, f32, _>(&v),
            ],
        )
        .is_err(),
        "strided query must be refused"
    );
    // Mixed dtypes are refused by the launcher, not silently bit-cast.
    let kf = upload_bytes(&[1, 2, 3, 4], DTypeId::F64, &f64_bytes(&vec![0.5; 24]));
    assert!(
        dispatch::execute::<op::FusedAttention, TestBackend>(
            &context,
            FusedAttentionAttributes {
                scale: None,
                causal: false,
            },
            &[
                TensorHandle::from_storage::<TestBackend, f32, _>(&q),
                TensorHandle::from_storage::<TestBackend, f64, _>(&kf),
                TensorHandle::from_storage::<TestBackend, f32, _>(&v),
            ],
        )
        .is_err(),
        "f32/f64 mix must be refused"
    );
}
