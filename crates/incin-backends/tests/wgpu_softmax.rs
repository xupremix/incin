//! `softmax` on real WGPU hardware, against the CPU reference.
//!
//! WGPU advertises `softmax` as `Composed` with `training = true`. Both halves
//! of that row are claims this file checks.
//!
//! `Composed` means the answer comes from rewriting into `max_keepdim`, `sub`,
//! `exp`, `sum_keepdim` and `log` rather than from a kernel of its own. That
//! rewrite is the numerically stable form -- subtracting the row max before
//! exponentiating -- and it is the same one CPU and `candle-nn` use, so the
//! two backends should agree closely rather than merely both being plausible.
//!
//! `training = true` means the operation records enough for a backward pass.
//! A composed forward that pushed nothing would still produce these exact
//! numbers and silently drop the gradient, so the tape depth is checked too,
//! and checked as a range rather than an equality: six primitives contribute,
//! but which of them coalesce is an implementation detail this test should not
//! freeze.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{HostInterop, HostReadback, StorageBackend, op};
use incin_core::exec::catalog::AxisAttributes;
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

/// Two rows of four. The second row is deliberately wide (`-8.0` next to
/// `8.0`), because that is the input where the unstable `exp(x) / sum(exp(x))`
/// spelling overflows and the stable one does not.
const VALUES: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -8.0, 0.0, 8.0, 0.0];
const ROWS: usize = 2;
const COLS: usize = 4;

fn upload(values: &[f32], shape: &[usize]) -> TestStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading the operand must succeed")
}

fn read(storage: &TestStorage) -> Vec<f64> {
    <TestBackend as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading a contiguous f32 buffer back must succeed")
}

/// Runs `softmax` over `axis` and reports the tape entries it added.
fn softmax(input: &TestStorage, axis: usize) -> (TestStorage, usize) {
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(input)];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::Softmax, _>(
        &context,
        AxisAttributes { axis },
        &inputs,
    )
    .expect("an advertised operation must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

/// The reference, computed on the host in the stable form.
fn reference(values: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; values.len()];
    for row in 0..rows {
        let slice = &values[row * cols..(row + 1) * cols];
        let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f64> = slice
            .iter()
            .map(|value| f64::from(*value - max).exp())
            .collect();
        let total: f64 = exponentials.iter().sum();
        for (column, value) in exponentials.iter().enumerate() {
            out[row * cols + column] = value / total;
        }
    }
    out
}

#[test]
fn softmax_matches_the_reference_and_records_a_gradient() {
    let input = upload(&VALUES, &[ROWS, COLS]);
    let (out, recorded) = softmax(&input, 1);
    let got = read(&out);
    let want = reference(&VALUES, ROWS, COLS);

    for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "element {index}: got {got}, reference {want}"
        );
    }

    assert!(
        recorded >= 2,
        "softmax advertises `training = true` and is composed from six \
         tape-tracked primitives, so it must leave entries behind. A fused \
         forward that recorded none would return these same numbers and \
         silently drop the gradient, which is the failure this asserts against."
    );
}

/// The defining property, and the one a wrong axis or a broadcasting mistake
/// breaks first: each row sums to exactly one.
#[test]
fn every_row_sums_to_one() {
    let input = upload(&VALUES, &[ROWS, COLS]);
    let (out, _) = softmax(&input, 1);
    let got = read(&out);

    for row in 0..ROWS {
        let total: f64 = got[row * COLS..(row + 1) * COLS].iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "row {row} sums to {total}, not 1.0"
        );
    }
}

/// Reducing the other axis has to give a different, also-normalised answer.
/// A kernel that ignored the attribute and always reduced the last axis would
/// pass every assertion above and fail this one.
#[test]
fn the_axis_attribute_is_honoured() {
    let input = upload(&VALUES, &[ROWS, COLS]);
    let (down_columns, _) = softmax(&input, 0);
    let (across_rows, _) = softmax(&input, 1);

    assert_ne!(
        read(&down_columns),
        read(&across_rows),
        "axis 0 and axis 1 must not produce the same tensor"
    );

    let got = read(&down_columns);
    for column in 0..COLS {
        let total: f64 = (0..ROWS).map(|row| got[row * COLS + column]).sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "column {column} sums to {total}, not 1.0"
        );
    }
}

/// The stability claim, stated as a test. `exp(8 - (-8))` is about 8.9e6 and
/// survives; the unstable spelling would evaluate `exp(8)/(exp(-8)+..)` with
/// no shift and drift or overflow on a wider row than this one.
#[test]
fn a_wide_row_stays_finite_and_normalised() {
    const WIDE: [f32; 4] = [-100.0, 0.0, 100.0, 50.0];
    let input = upload(&WIDE, &[1, 4]);
    let (out, _) = softmax(&input, 1);
    let got = read(&out);

    assert!(
        got.iter().all(|value| value.is_finite()),
        "a wide row must not overflow to inf or NaN: {got:?}"
    );
    let total: f64 = got.iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "wide row sums to {total}");
    assert!(
        got[2] > 0.99,
        "the largest element should carry almost all the mass: {got:?}"
    );
}
