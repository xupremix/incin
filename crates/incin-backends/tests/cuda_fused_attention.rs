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

use half::{bf16, f16};
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
use incin_core::tensor::dtype::DType;
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

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}

fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
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
    for dtype in [DTypeId::F32, DTypeId::F64, DTypeId::F16, DTypeId::BF16] {
        let level = incin_backends::capability::support(
            incin_core::tensor::device::DeviceKind::Cuda,
            &CapabilityQuery {
                operation: OperationIdentity::Builtin(OperationKind::FusedAttention),
                dtype: dtype.descriptor(),
                layout: LayoutClass::Contiguous,
                rank: 4,
                training: true,
                math_mode: MathMode::default(),
            },
        );
        assert_eq!(
            level,
            SupportLevel::Native,
            "{dtype:?} fused attention ships a native CUDA kernel, so rank 4 must be Native with training"
        );
    }
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

/// Three half-dtype inputs through `dispatch`: the handle element type
/// carries the storage dtype, so f16 and bf16 need one monomorphic call
/// each.
fn run_half<K: DType>(
    context: &ExecutionContext<TestBackend>,
    q: &TestStorage,
    k: &TestStorage,
    v: &TestStorage,
) -> (
    Result<TestStorage, incin_core::exec::dispatch::CanonicalError>,
    usize,
) {
    let before = tape_depth();
    let out = dispatch::execute::<op::FusedAttention, TestBackend>(
        context,
        FusedAttentionAttributes {
            scale: None,
            causal: false,
        },
        &[
            TensorHandle::from_storage::<TestBackend, K, _>(q),
            TensorHandle::from_storage::<TestBackend, K, _>(k),
            TensorHandle::from_storage::<TestBackend, K, _>(v),
        ],
    );
    (out, tape_depth() - before)
}

#[test]
#[ignore = "requires CUDA hardware"]
fn fused_forward_f16_bf16_match_f32_within_half_tolerance() {
    require_cuda();
    // B=1, H=1, Sq=2, Skv=2, D=2, non-causal. Values are exactly
    // representable in both half formats, so the only error is the
    // half rounding of intermediates against the f32 twin.
    let qv = [0.5, -1.0, 2.0, 0.25];
    let kv = [1.0, 0.5, -0.5, 1.5];
    let vv = [0.25, 1.0, -1.0, 0.5];
    let context = ExecutionContext::new(TestBackend::new());
    for (dtype, bytes) in [
        (
            DTypeId::F16,
            (f16_bytes(&qv), f16_bytes(&kv), f16_bytes(&vv)),
        ),
        (
            DTypeId::BF16,
            (bf16_bytes(&qv), bf16_bytes(&kv), bf16_bytes(&vv)),
        ),
    ] {
        let (qb, kb, vb) = bytes;
        let q = upload_bytes(&[1, 1, 2, 2], dtype, &qb);
        let k = upload_bytes(&[1, 1, 2, 2], dtype, &kb);
        let v = upload_bytes(&[1, 1, 2, 2], dtype, &vb);
        let before = tape_depth();
        let out = if dtype == DTypeId::F16 {
            run_half::<f16>(&context, &q, &k, &v)
        } else {
            run_half::<bf16>(&context, &q, &k, &v)
        };
        let (out, _recorded) = (out.0.expect("half fused attention must execute"), out.1);
        assert_eq!(
            tape_depth() - before,
            1,
            "fused attention records one tape entry"
        );
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        let got_bytes = download_bytes(&out);
        let got: Vec<f64> = if dtype == DTypeId::F16 {
            decode_f16(&got_bytes)
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        } else {
            decode_bf16(&got_bytes)
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        };
        #[cfg(feature = "cpu")]
        {
            let context = ExecutionContext::new(Cpu::new());
            let hq = <Cpu as HostInterop>::from_bytes::<f32>(
                bytemuck::cast_slice(&qv),
                &[1, 1, 2, 2],
                DTypeId::F32.descriptor(),
                &incin_core::tensor::device::DeviceId::cpu(),
            )
            .unwrap();
            let hk = <Cpu as HostInterop>::from_bytes::<f32>(
                bytemuck::cast_slice(&kv),
                &[1, 1, 2, 2],
                DTypeId::F32.descriptor(),
                &incin_core::tensor::device::DeviceId::cpu(),
            )
            .unwrap();
            let hv = <Cpu as HostInterop>::from_bytes::<f32>(
                bytemuck::cast_slice(&vv),
                &[1, 1, 2, 2],
                DTypeId::F32.descriptor(),
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
                    TensorHandle::from_storage::<Cpu, f32, _>(&hq),
                    TensorHandle::from_storage::<Cpu, f32, _>(&hk),
                    TensorHandle::from_storage::<Cpu, f32, _>(&hv),
                ],
            )
            .expect("CPU reference f32 fused attention executes");
            let want = cpu_read_f32(&want);
            let gotf: Vec<f64> = got;
            // Observed maxima on GTX 1650S: F16 1.4e-4, BF16 8.4e-4;
            // the 1e-2 bound leaves an order of magnitude for
            // driver/hardware variance.
            assert_close(&gotf, &want, 1e-2, &format!("{dtype:?} forward"));
        }
    }
}
/// Deterministic pseudo-random values in `[-1, 1)`: the parity geometries
/// below are too large to hand-write, and a fixed LCG keeps the CPU twin
/// and the GPU run on identical inputs.
fn lcg(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / 2147483648.0) - 1.0
        })
        .collect()
}

/// Query lengths that exercise the tiled forward: an exact multiple of the
/// tile, a ragged tail of 2 live rows, a ragged tail of 1, and the
/// smallest length the launcher is allowed to tile.
const TILED_SEQ_Q: [usize; 4] = [8, 10, 9, 8];

#[test]
#[ignore = "requires CUDA hardware"]
fn tiled_forward_matches_the_cpu_kernel_across_tile_tails() {
    require_cuda();
    // B=2, Hq=4 over Hkv=2 (GQA), D=8. Every listed query length is at
    // least two tiles, so the launcher must pick the tiled kernel; the
    // ragged ones (10 -> 2+2+... tiles of 4, 9 -> 4+4+1) are the cases a
    // tile decomposition gets wrong.
    let (b, hq, hkv, d) = (2usize, 4usize, 2usize, 8usize);
    for &sq in &TILED_SEQ_Q {
        let skv = sq;
        let attributes = FusedAttentionAttributes {
            scale: None,
            causal: true,
        };
        let qv = lcg(0x51ed_0000 + sq as u64, b * hq * sq * d);
        let kv = lcg(0x51ed_1000 + sq as u64, b * hkv * skv * d);
        let vv = lcg(0x51ed_2000 + sq as u64, b * hkv * skv * d);
        let context = ExecutionContext::new(TestBackend::new());
        let q = upload_f32_shaped(&[b, hq, sq, d], &qv);
        let k = upload_f32_shaped(&[b, hkv, skv, d], &kv);
        let v = upload_f32_shaped(&[b, hkv, skv, d], &vv);
        let (out, recorded) = run_fused(&context, &q, &k, &v, attributes);
        assert_eq!(recorded, 1, "tiled fused attention records one tape entry");
        assert_eq!(out.shape, vec![b, hq, sq, d]);
        #[cfg(feature = "cpu")]
        {
            let want = cpu_fused(
                &[b, hq, sq, d],
                &qv,
                &[b, hkv, skv, d],
                &kv,
                &vv,
                FusedAttentionAttributes {
                    scale: None,
                    causal: true,
                },
            );
            assert_close(
                &read_f32(&out),
                &want,
                1e-4,
                &format!("tiled causal forward sq={sq}"),
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn tiled_forward_decode_geometry_matches_the_cpu_kernel() {
    require_cuda();
    // A short prefill over a longer cached prefix: Skv > Sq, so
    // kv_lead = Skv - Sq > 0 and every query row sees keys past itself.
    // Tiled, because the query length is.
    let (b, hq, hkv, sq, skv, d) = (1usize, 2usize, 2usize, 8usize, 20usize, 4usize);
    let attributes = FusedAttentionAttributes {
        scale: Some(0.25),
        causal: true,
    };
    let qv = lcg(0xdec0_de01, b * hq * sq * d);
    let kv = lcg(0xdec0_de02, b * hkv * skv * d);
    let vv = lcg(0xdec0_de03, b * hkv * skv * d);
    let context = ExecutionContext::new(TestBackend::new());
    let q = upload_f32_shaped(&[b, hq, sq, d], &qv);
    let k = upload_f32_shaped(&[b, hkv, skv, d], &kv);
    let v = upload_f32_shaped(&[b, hkv, skv, d], &vv);
    let (out, _) = run_fused(&context, &q, &k, &v, attributes);
    assert_eq!(out.shape, vec![b, hq, sq, d]);
    #[cfg(feature = "cpu")]
    {
        let want = cpu_fused(
            &[b, hq, sq, d],
            &qv,
            &[b, hkv, skv, d],
            &kv,
            &vv,
            FusedAttentionAttributes {
                scale: Some(0.25),
                causal: true,
            },
        );
        assert_close(&read_f32(&out), &want, 1e-4, "tiled prefill with kv_lead");
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn tiled_backward_matches_the_cpu_twin() {
    require_cuda();
    // Same tiled geometry, gradients included: the forward tiling must not
    // change the statistics the recompute backward reads back.
    let (b, hq, hkv, sq, skv, d) = (2usize, 4usize, 2usize, 10usize, 10usize, 8usize);
    let attributes = FusedAttentionAttributes {
        scale: None,
        causal: true,
    };
    let qv = lcg(0xba5e_0001, b * hq * sq * d);
    let kv = lcg(0xba5e_0002, b * hkv * skv * d);
    let vv = lcg(0xba5e_0003, b * hkv * skv * d);
    let context = ExecutionContext::new(TestBackend::new());
    let q = upload_f32_shaped(&[b, hq, sq, d], &qv);
    let k = upload_f32_shaped(&[b, hkv, skv, d], &kv);
    let v = upload_f32_shaped(&[b, hkv, skv, d], &vv);
    let (q_id, k_id, v_id) = (
        TapeStorage::id(&q),
        TapeStorage::id(&k),
        TapeStorage::id(&v),
    );
    let (out, _) = run_fused(&context, &q, &k, &v, attributes);
    let loss = dispatch::execute::<op::SumAll, _>(
        &context,
        incin_core::exec::catalog::NoAttributes,
        &[TensorHandle::from_storage::<TestBackend, f32, _>(&out)],
    )
    .expect("sum_all executes");
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");
    #[cfg(feature = "cpu")]
    {
        let hq_in = <Cpu as HostInterop>::from_bytes::<f32>(
            bytemuck::cast_slice(&qv),
            &[b, hq, sq, d],
            DTypeId::F32.descriptor(),
            &incin_core::tensor::device::DeviceId::cpu(),
        )
        .unwrap();
        let hk_in = <Cpu as HostInterop>::from_bytes::<f32>(
            bytemuck::cast_slice(&kv),
            &[b, hkv, skv, d],
            DTypeId::F32.descriptor(),
            &incin_core::tensor::device::DeviceId::cpu(),
        )
        .unwrap();
        let hv_in = <Cpu as HostInterop>::from_bytes::<f32>(
            bytemuck::cast_slice(&vv),
            &[b, hkv, skv, d],
            DTypeId::F32.descriptor(),
            &incin_core::tensor::device::DeviceId::cpu(),
        )
        .unwrap();
        let (hq_id, hk_id, hv_id) = (
            TapeStorage::id(&hq_in),
            TapeStorage::id(&hk_in),
            TapeStorage::id(&hv_in),
        );
        let context = ExecutionContext::new(Cpu::new());
        let hout = dispatch::execute::<op::FusedAttention, Cpu>(
            &context,
            FusedAttentionAttributes {
                scale: None,
                causal: true,
            },
            &[
                TensorHandle::from_storage::<Cpu, f32, _>(&hq_in),
                TensorHandle::from_storage::<Cpu, f32, _>(&hk_in),
                TensorHandle::from_storage::<Cpu, f32, _>(&hv_in),
            ],
        )
        .expect("CPU reference tiled fused attention executes");
        let hloss = dispatch::execute::<op::SumAll, _>(
            &context,
            incin_core::exec::catalog::NoAttributes,
            &[TensorHandle::from_storage::<Cpu, f32, _>(&hout)],
        )
        .expect("CPU sum_all executes");
        let hgrads = <Cpu as AutogradBackend>::backward::<f32>(&hloss).expect("CPU backward runs");
        assert_close(
            &read_f32(grads.get(q_id).expect("query receives a gradient")),
            &cpu_read_f32(hgrads.get(hq_id).unwrap()),
            1e-4,
            "tiled dq",
        );
        assert_close(
            &read_f32(grads.get(k_id).expect("key receives a gradient")),
            &cpu_read_f32(hgrads.get(hk_id).unwrap()),
            1e-4,
            "tiled dk",
        );
        assert_close(
            &read_f32(grads.get(v_id).expect("value receives a gradient")),
            &cpu_read_f32(hgrads.get(hv_id).unwrap()),
            1e-4,
            "tiled dv",
        );
    }
}

/// Wall-clock for the forward at transformer-shaped geometries, tiled.
/// Prints rather than asserts: the useful question is how the tiled
/// kernel compares with the one-row-per-block kernel at the same
/// geometry, and a threshold in a test would only encode this machine's
/// clocks. The `wide` row is the interesting one - its key/value working
/// set is larger than L2, so the 4x cut in streamed bytes shows up; the
/// `square` row is small enough to sit in cache, where it cannot.
#[test]
#[ignore = "requires CUDA hardware"]
fn tiled_forward_timing_across_geometries() {
    require_cuda();
    const REPEATS: u32 = 20;
    let attributes = FusedAttentionAttributes {
        scale: None,
        causal: false,
    };
    for (label, (b, hq, sq, skv, d)) in [
        ("square", (1usize, 8usize, 256usize, 256usize, 64usize)),
        ("wide", (1usize, 4usize, 512usize, 4096usize, 64usize)),
    ] {
        let qv = lcg(0x71_0001, b * hq * sq * d);
        let kv = lcg(0x71_0002, b * hq * skv * d);
        let vv = lcg(0x71_0003, b * hq * skv * d);
        let context = ExecutionContext::new(TestBackend::new());
        let q = upload_f32_shaped(&[b, hq, sq, d], &qv);
        let k = upload_f32_shaped(&[b, hq, skv, d], &kv);
        let v = upload_f32_shaped(&[b, hq, skv, d], &vv);
        // Warm up: module load, JIT, and the first allocation are not the
        // steady state being measured.
        let (out, _) = run_fused(&context, &q, &k, &v, attributes.clone());
        assert_eq!(out.shape, vec![b, hq, sq, d]);
        std::hint::black_box(download_bytes(&out));
        let start = std::time::Instant::now();
        for _ in 0..REPEATS {
            let (out, _) = run_fused(&context, &q, &k, &v, attributes.clone());
            std::hint::black_box(out);
        }
        // The readback is on the same stream, so it is also the sync point.
        let last = run_fused(&context, &q, &k, &v, attributes.clone()).0;
        std::hint::black_box(download_bytes(&last));
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_secs_f64() / f64::from(REPEATS);
        // Every key row is D floats, streamed once per query row without
        // tiling and once per query *block* with it.
        let row_bytes = (b * hq * skv * d * std::mem::size_of::<f32>()) as f64;
        println!(
            "{label} forward B={b} Hq={hq} Sq={sq} Skv={skv} D={d}: {per_iter:.3} ms/iter \
             ({REPEATS} iters in {:.3} s); K+V bytes per query row {row_bytes:.0}, \
             per query tile {:.0}",
            elapsed.as_secs_f64(),
            row_bytes / 4.0,
        );
        assert!(
            per_iter.is_finite() && per_iter > 0.0,
            "a timing run must produce a positive duration, got {per_iter}"
        );
    }
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
