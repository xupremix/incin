//! Issue #85 batched CUDA GEMM end to end: the native grid-z kernel (one
//! launch, explicit slice strides, stride-0 batch broadcasts) and the
//! `cuda-vendor` strided-batched cuBLASLt path, plus the composed-loop
//! guarantee for shapes neither admits.
//!
//! Forwards are checked against a naive f64 host GEMM with right-aligned
//! batch broadcasting; gradients are checked against the CPU twin under
//! the same dispatch; the vendor path is checked against the native one on
//! the shapes both claim. Every hardware test is `#[ignore]`d;
//! `require_cuda` fails loudly rather than skipping, because reaching an
//! `#[ignore]`d test is an explicit request for the hardware run.
#![cfg(feature = "cuda")]

use incin_backends::cuda::{
    CudaBackendImpl, tape_depth,
    testing::{
        batched_matmul, batched_matmul_native, download_f32, require_cuda, upload_f32_shaped,
    },
};
use incin_core::backend_authoring::{
    AutogradBackend, Execute, HostInterop, HostReadback, StorageBackend,
};
use incin_core::exec::catalog::{LinearAttributes, NoAttributes};
use incin_core::exec::{
    CanonicalOperation, ExecutionContext, TapeStorage, TensorHandle, dispatch, op,
};
use incin_core::prelude::{CudaN, DTypeId, DeviceId};
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

fn as_f64(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

fn read_f64(storage: &TestStorage) -> Vec<f64> {
    as_f64(&download_f32(storage))
}

/// Row-major naive f64 GEMM with right-aligned batch broadcasting: the
/// host reference every forward here compares against. `out[batch, i, j]`
/// sums `lhs[batch, i, p] * rhs[batch, p, j]`, with an operand reading one
/// matrix for the whole batch when it has no batch axis or its extent is 1
/// - the same broadcasts the flat kernel expresses as stride 0.
fn naive_batched_gemm(
    lhs: &[f64],
    lhs_shape: &[usize],
    rhs: &[f64],
    rhs_shape: &[usize],
) -> (Vec<f64>, Vec<usize>) {
    let (lr, rr) = (lhs_shape.len(), rhs_shape.len());
    assert!(lr >= 2 && rr >= 2, "operands must have rank at least 2");
    let (m, k) = (lhs_shape[lr - 2], lhs_shape[lr - 1]);
    let (rhs_k, n) = (rhs_shape[rr - 2], rhs_shape[rr - 1]);
    assert_eq!(k, rhs_k, "inner dimensions must match");
    let lb = &lhs_shape[..lr - 2];
    let rb = &rhs_shape[..rr - 2];
    let out_rank = lb.len().max(rb.len());
    let pad = |batch: &[usize]| -> Vec<usize> {
        let mut padded = vec![1usize; out_rank - batch.len()];
        padded.extend_from_slice(batch);
        padded
    };
    let (lb_pad, rb_pad) = (pad(lb), pad(rb));
    let mut out_batch = Vec::with_capacity(out_rank);
    for (&a, &b) in lb_pad.iter().zip(&rb_pad) {
        assert!(
            a == b || a == 1 || b == 1,
            "shapes must broadcast: {lhs_shape:?} vs {rhs_shape:?}"
        );
        out_batch.push(a.max(b));
    }
    let batch_total: usize = out_batch.iter().product();
    let mut out = vec![0.0; batch_total * m * n];
    let operand_flat = |batch: &[usize], coords_tail: &[usize]| -> usize {
        let mut flat = 0;
        for (i, &dim) in batch.iter().enumerate() {
            let coord = if dim == 1 { 0 } else { coords_tail[i] };
            flat = flat * dim + coord;
        }
        flat
    };
    for flat_out in 0..batch_total {
        let mut rem = flat_out;
        let mut coords = vec![0usize; out_rank];
        for d in (0..out_rank).rev() {
            coords[d] = rem % out_batch[d];
            rem /= out_batch[d];
        }
        let lhs_tail = &coords[out_rank - lb.len()..];
        let rhs_tail = &coords[out_rank - rb.len()..];
        let lhs_base = operand_flat(lb, lhs_tail) * m * k;
        let rhs_base = operand_flat(rb, rhs_tail) * k * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += lhs[lhs_base + i * k + p] * rhs[rhs_base + p * n + j];
                }
                out[flat_out * m * n + i * n + j] = acc;
            }
        }
    }
    let mut out_shape = out_batch;
    out_shape.extend_from_slice(&[m, n]);
    (out, out_shape)
}

/// Two f32 inputs through `dispatch`, returning the output and the tape
/// entries the forward pushed.
fn run2<O, A>(
    context: &ExecutionContext<TestBackend>,
    first: &TestStorage,
    second: &TestStorage,
    attributes: A,
) -> (TestStorage, usize)
where
    O: CanonicalOperation<Attributes = A>,
    TestBackend: Execute<O, Output = TestStorage>,
{
    let before = tape_depth();
    let out = dispatch::execute::<O, TestBackend>(
        context,
        attributes,
        &[
            TensorHandle::from_storage::<TestBackend, f32, _>(first),
            TensorHandle::from_storage::<TestBackend, f32, _>(second),
        ],
    )
    .expect("an advertised two-input CUDA operation must execute");
    (out, tape_depth() - before)
}

/// Three f32 inputs through `dispatch` (`linear`'s input, weight, bias).
fn run3<O, A>(
    context: &ExecutionContext<TestBackend>,
    first: &TestStorage,
    second: &TestStorage,
    third: &TestStorage,
    attributes: A,
) -> (TestStorage, usize)
where
    O: CanonicalOperation<Attributes = A>,
    TestBackend: Execute<O, Output = TestStorage>,
{
    let before = tape_depth();
    let out = dispatch::execute::<O, TestBackend>(
        context,
        attributes,
        &[
            TensorHandle::from_storage::<TestBackend, f32, _>(first),
            TensorHandle::from_storage::<TestBackend, f32, _>(second),
            TensorHandle::from_storage::<TestBackend, f32, _>(third),
        ],
    )
    .expect("an advertised three-input CUDA operation must execute");
    (out, tape_depth() - before)
}

#[test]
#[ignore = "requires CUDA hardware"]
fn rank_three_batched_matmul_matches_the_host_naive_gemm_in_one_tape_entry() {
    // [2,2,3] x [2,3,2] -> [2,2,2]. The production seam must admit the
    // contiguous pair (vendor first under `cuda-vendor`, then native) and
    // the forward must land as a *single* tape entry: the composed loop
    // would push a reshape/narrow/concat stack per slice, so `recorded == 1`
    // is what distinguishes one batched launch from the old composition.
    require_cuda();
    let lhs_shape = [2, 2, 3];
    let rhs_shape = [2, 3, 2];
    let lhs_vals: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let rhs_vals: Vec<f32> = (1..=12).map(|v| v as f32).collect();

    let (want, want_shape) = naive_batched_gemm(
        &as_f64(&lhs_vals),
        &lhs_shape,
        &as_f64(&rhs_vals),
        &rhs_shape,
    );

    let lhs = upload_f32_shaped(&lhs_shape, &lhs_vals);
    let rhs = upload_f32_shaped(&rhs_shape, &rhs_vals);
    let product = batched_matmul(&lhs, &rhs)
        .expect("the batched orchestrator must not fail on a plain request")
        .expect("a contiguous rank-3 f32 pair must be admitted by a batched path");
    assert_eq!(product.shape, want_shape, "product shape");
    assert_close(&read_f64(&product), &want, 1e-4, "rank-3 batched product");

    // The same request through `op::MatMulExact` - the route
    // `Tensor::matmul` takes - must push exactly one tape entry, which is
    // only true when a batched path (not the composed loop) served it.
    let context = ExecutionContext::new(TestBackend::new());
    let (out, recorded) = run2::<op::MatMulExact, _>(&context, &lhs, &rhs, NoAttributes);
    assert_eq!(
        recorded, 1,
        "one batched launch pushes one tape entry; the composed loop would push more"
    );
    assert_eq!(out.shape, want_shape, "MatMulExact product shape");
    assert_close(&read_f64(&out), &want, 1e-4, "MatMulExact rank-3 product");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn rank_four_batched_matmul_matches_the_host_naive_gemm() {
    // [2,2,2,3] x [2,2,3,2] -> [2,2,2,2]: two batch axes sharing one
    // affine constant (the flat slice stride is m * k, not the axis-0
    // stride), which is the case a host-side launch loop could not have
    // expressed without materializing copies.
    require_cuda();
    let lhs_shape = [2, 2, 2, 3];
    let rhs_shape = [2, 2, 3, 2];
    let lhs_vals: Vec<f32> = (1..=24).map(|v| (v % 7) as f32 - 3.0).collect();
    let rhs_vals: Vec<f32> = (1..=24).map(|v| (v % 5) as f32 - 2.0).collect();

    let (want, want_shape) = naive_batched_gemm(
        &as_f64(&lhs_vals),
        &lhs_shape,
        &as_f64(&rhs_vals),
        &rhs_shape,
    );

    let lhs = upload_f32_shaped(&lhs_shape, &lhs_vals);
    let rhs = upload_f32_shaped(&rhs_shape, &rhs_vals);
    let product = batched_matmul(&lhs, &rhs)
        .expect("the batched orchestrator must not fail on a plain request")
        .expect("a contiguous rank-4 f32 pair must be admitted by the native plan");
    assert_eq!(product.shape, want_shape, "product shape");
    assert_close(&read_f64(&product), &want, 1e-4, "rank-4 batched product");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn batch_broadcast_operands_are_served_with_stride_zero_reads() {
    // The broadcasts the flat kernel expresses directly: a rank-2 lhs
    // (no batch axis at all) and a size-1 leading lhs axis, each reading
    // one matrix for every slice of the rhs batch. Neither may materialize
    // a broadcast copy - the plan admits them precisely because stride 0
    // carries the broadcast.
    require_cuda();

    // [2,3] x [4,3,2] -> [4,2,2]: the lhs has no batch axis.
    let lhs_shape = [2, 3];
    let rhs_shape = [4, 3, 2];
    let lhs_vals: Vec<f32> = (1..=6).map(|v| v as f32).collect();
    let rhs_vals: Vec<f32> = (1..=24).map(|v| (v % 9) as f32 - 4.0).collect();
    let (want, want_shape) = naive_batched_gemm(
        &as_f64(&lhs_vals),
        &lhs_shape,
        &as_f64(&rhs_vals),
        &rhs_shape,
    );
    let lhs = upload_f32_shaped(&lhs_shape, &lhs_vals);
    let rhs = upload_f32_shaped(&rhs_shape, &rhs_vals);
    let native = batched_matmul_native(&lhs, &rhs)
        .expect("native batched launch must not fail")
        .expect("a rank-two lhs broadcasts as stride 0");
    assert_eq!(native.shape, want_shape, "broadcast product shape");
    assert_close(
        &read_f64(&native),
        &want,
        1e-4,
        "rank-two lhs broadcast product",
    );

    // [1,2,3] x [4,3,2] -> [4,2,2]: a size-1 leading axis is vacuous on
    // the lhs side of the broadcast, so it must not constrain the stride.
    let lhs_shape = [1, 2, 3];
    let lhs_vals: Vec<f32> = (1..=6).map(|v| v as f32).collect();
    let (want, want_shape) = naive_batched_gemm(
        &as_f64(&lhs_vals),
        &lhs_shape,
        &as_f64(&rhs_vals),
        &rhs_shape,
    );
    let lhs = upload_f32_shaped(&lhs_shape, &lhs_vals);
    let native = batched_matmul_native(&lhs, &rhs)
        .expect("native batched launch must not fail")
        .expect("a size-one leading axis must stay admissible");
    assert_eq!(native.shape, want_shape, "vacuous-axis product shape");
    assert_close(
        &read_f64(&native),
        &want,
        1e-4,
        "size-one leading axis product",
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn non_affine_batch_broadcast_is_refused_by_the_flat_paths_and_served_composed() {
    // [2,1,2,3] x [2,4,3,2] -> [2,4,2,2]: the lhs varies along axis 0
    // with stride m*k = 6 while axis 1 must read stride 0 (extent 1); one
    // flat slice stride cannot satisfy both, so the native plan and the
    // vendor policy (rank-3 only) both refuse. Correctness must survive
    // through the composed loop - refusing is a missed fast path, never a
    // wrong value.
    require_cuda();
    let lhs_shape = [2, 1, 2, 3];
    let rhs_shape = [2, 4, 3, 2];
    let lhs_vals: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let rhs_vals: Vec<f32> = (1..=48).map(|v| (v % 11) as f32 - 5.0).collect();
    let (want, want_shape) = naive_batched_gemm(
        &as_f64(&lhs_vals),
        &lhs_shape,
        &as_f64(&rhs_vals),
        &rhs_shape,
    );

    let lhs = upload_f32_shaped(&lhs_shape, &lhs_vals);
    let rhs = upload_f32_shaped(&rhs_shape, &rhs_vals);
    assert!(
        batched_matmul_native(&lhs, &rhs)
            .expect("the native attempt must not fail")
            .is_none(),
        "a non-affine batch broadcast must be refused by the flat-stride plan"
    );
    assert!(
        batched_matmul(&lhs, &rhs)
            .expect("the orchestrator must not fail on a plain request")
            .is_none(),
        "no batched path claims a non-affine stride pattern"
    );

    // The composed loop through `op::MatMulExact` still computes the
    // product (and, unlike the batched paths, pushes its per-slice tape
    // stack - any positive entry count is enough to show it ran).
    let context = ExecutionContext::new(TestBackend::new());
    let (out, recorded) = run2::<op::MatMulExact, _>(&context, &lhs, &rhs, NoAttributes);
    assert!(
        recorded >= 1,
        "the composed fallback is tape-tracked, recorded {recorded}"
    );
    assert_eq!(out.shape, want_shape, "composed product shape");
    assert_close(
        &read_f64(&out),
        &want,
        1e-4,
        "composed non-affine batch product",
    );
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn batched_matmul_backward_matches_the_cpu_twin() {
    // The native path's single tape entry carries a hand-written backward
    // (transpose -> batched product -> unbroadcast per gradient), so it is
    // checked against the CPU composition under the same dispatch rather
    // than against itself.
    require_cuda();
    let lhs_shape = [2, 2, 3];
    let rhs_shape = [2, 3, 2];
    let lhs_vals: Vec<f32> = (1..=12).map(|v| (v % 5) as f32 - 2.0).collect();
    let rhs_vals: Vec<f32> = (1..=12).map(|v| (v % 7) as f32 - 3.0).collect();

    let context = ExecutionContext::new(TestBackend::new());
    let lhs = upload_f32_shaped(&lhs_shape, &lhs_vals);
    let rhs = upload_f32_shaped(&rhs_shape, &rhs_vals);
    let (lhs_id, rhs_id) = (TapeStorage::id(&lhs), TapeStorage::id(&rhs));
    let (out, recorded) = run2::<op::MatMulExact, _>(&context, &lhs, &rhs, NoAttributes);
    assert_eq!(recorded, 1, "the batched forward pushes one tape entry");
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&out)
        .expect("batched matmul backward on CUDA");
    let gl = grads
        .get(lhs_id)
        .expect("batched matmul lhs receives a gradient");
    let gr = grads
        .get(rhs_id)
        .expect("batched matmul rhs receives a gradient");

    let cpu_lhs = <Cpu as HostInterop>::from_bytes::<f32>(
        bytemuck::cast_slice(&lhs_vals),
        &lhs_shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .expect("uploading the CPU twin lhs must succeed");
    let cpu_rhs = <Cpu as HostInterop>::from_bytes::<f32>(
        bytemuck::cast_slice(&rhs_vals),
        &rhs_shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .expect("uploading the CPU twin rhs must succeed");
    let cpu_context = ExecutionContext::new(Cpu::new());
    let cpu_out = dispatch::execute::<op::MatMulExact, _>(
        &cpu_context,
        NoAttributes,
        &[
            TensorHandle::from_storage::<Cpu, f32, _>(&cpu_lhs),
            TensorHandle::from_storage::<Cpu, f32, _>(&cpu_rhs),
        ],
    )
    .expect("CPU reference batched matmul executes");
    let cpu_grads = <Cpu as AutogradBackend>::backward::<f32>(&cpu_out)
        .expect("batched matmul backward on CPU");
    let cpu_gl = cpu_grads
        .get(TapeStorage::id(&cpu_lhs))
        .expect("CPU reference is missing dL/dlhs");
    let cpu_gr = cpu_grads
        .get(TapeStorage::id(&cpu_rhs))
        .expect("CPU reference is missing dL/drhs");

    let want_fwd = <Cpu as HostReadback>::float_to_vec1::<f32>(&cpu_out)
        .expect("reading the CPU forward back");
    assert_close(&read_f64(&out), &want_fwd, 1e-5, "batched forward vs CPU");
    let want_gl =
        <Cpu as HostReadback>::float_to_vec1::<f32>(cpu_gl).expect("reading the CPU dL/dlhs back");
    assert_close(&read_f64(gl), &want_gl, 1e-5, "batched dL/dlhs vs CPU");
    let want_gr =
        <Cpu as HostReadback>::float_to_vec1::<f32>(cpu_gr).expect("reading the CPU dL/drhs back");
    assert_close(&read_f64(gr), &want_gr, 1e-5, "batched dL/drhs vs CPU");
}

#[cfg(feature = "cuda-vendor")]
#[test]
#[ignore = "requires CUDA hardware"]
fn vendor_and_native_batched_paths_agree_on_the_shape_both_claim() {
    // The vendor policy claims exactly the contiguous equal-batch rank-3
    // f32 pair; on that shape both paths must compute the same values
    // (cuBLASLt may pick a different algorithm than the tiled kernel, so
    // this is a tolerance comparison, not bitwise equality). Running both
    // directly also proves each path is reachable on its own - the
    // orchestrator hides which one served a production request.
    use incin_backends::cuda::testing::batched_matmul_vendor;

    require_cuda();
    let lhs_shape = [3, 4, 5];
    let rhs_shape = [3, 5, 6];
    let lhs_vals: Vec<f32> = (1..=60).map(|v| (v % 13) as f32 - 6.0).collect();
    let rhs_vals: Vec<f32> = (1..=90).map(|v| (v % 17) as f32 - 8.0).collect();
    let lhs = upload_f32_shaped(&lhs_shape, &lhs_vals);
    let rhs = upload_f32_shaped(&rhs_shape, &rhs_vals);

    let native = batched_matmul_native(&lhs, &rhs)
        .expect("the native attempt must not fail")
        .expect("the native plan must admit the contiguous rank-3 pair");
    let vendor = batched_matmul_vendor(&lhs, &rhs)
        .expect("the vendor attempt must not fail on an admitted request")
        .expect("the vendor policy must admit the contiguous rank-3 f32 pair");

    assert_eq!(
        native.shape, vendor.shape,
        "both paths must produce the product's shape"
    );
    assert_close(
        &read_f64(&vendor),
        &read_f64(&native),
        1e-4,
        "vendor vs native batched product",
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn rank_three_linear_projects_through_the_batched_dispatcher() {
    // `op::Linear` with a leading batch on the input: before issue #85
    // this failed inside `matmul`'s unbatched-2D check. It must now reach
    // the batched dispatcher (one launch when the plan admits the pair),
    // add the bias by broadcast, and match the host computation
    // `out[b,t,o] = sum_i x[b,t,i] * w[o,i] + b[o]`.
    require_cuda();
    let x_shape = [2, 3, 3];
    let w_shape = [2, 3];
    let x_vals: Vec<f32> = (1..=18).map(|v| (v % 7) as f32 - 3.0).collect();
    let w_vals: Vec<f32> = vec![0.5, -0.25, 0.75, 0.1, 0.2, -0.4];
    let b_vals: Vec<f32> = vec![0.125, -0.375];

    let (batch, rows, cols) = (x_shape[0], x_shape[1], x_shape[2]);
    let out_dim = w_shape[0];
    let mut want = vec![0.0f64; batch * rows * out_dim];
    for b in 0..batch {
        for t in 0..rows {
            for o in 0..out_dim {
                let mut acc = f64::from(b_vals[o]);
                for i in 0..cols {
                    acc += f64::from(x_vals[b * rows * cols + t * cols + i])
                        * f64::from(w_vals[o * cols + i]);
                }
                want[(b * rows + t) * out_dim + o] = acc;
            }
        }
    }

    let context = ExecutionContext::new(TestBackend::new());
    let x = upload_f32_shaped(&x_shape, &x_vals);
    let w = upload_f32_shaped(&w_shape, &w_vals);
    let b = upload_f32_shaped(&[out_dim], &b_vals);
    let (logits, recorded) =
        run3::<op::Linear, _>(&context, &x, &w, &b, LinearAttributes { has_bias: true });
    assert_eq!(
        recorded, 3,
        "rank-three Linear records exactly transpose(weight) + one batched \
         matmul + one bias add; the composed loop would push a \
         reshape/narrow/concat stack for the matmul alone (recorded {recorded})"
    );
    assert_eq!(
        logits.shape,
        vec![batch, rows, out_dim],
        "rank-three Linear output shape"
    );
    assert_close(
        &read_f64(&logits),
        &want,
        1e-4,
        "rank-three linear projection",
    );
}
