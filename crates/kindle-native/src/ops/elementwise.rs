//! `NumericOps` (`add`/`sub`/`mul`/`div`) and `FloatOps::{add_scalar_float,
//! mul_scalar_float}` for `NativeBackend<T, D>`.
//!
//! Every op here resolves the broadcast output shape via
//! `stride::broadcast_shape`, then iterates the OUTPUT shape's logical index
//! space, resolving each operand's own index through its own strides with
//! wraparound (stride-0-equivalent) logic on right-aligned/expanded
//! dimensions — it never pre-materializes a broadcast copy of either operand
//! (the anti-pattern flagged in RESEARCH.md). Every op pushes a `TapeEntry`
//! whose backward closure calls `tape::unbroadcast` on the ORIGINAL
//! (pre-broadcast) operand shapes.

use kindle_core::prelude::{Backend, DType, FloatOps, NumericOps, Result};

use crate::NativeBackend;
use crate::storage::{NativeBuffer, NativeStorage};
use crate::tape::{self, TapeEntry};

/// Resolve the per-operand logical index for a given output logical index,
/// right-aligning `operand_shape` against `out_shape` (numpy/Candle-style
/// broadcast): any leading dim the operand doesn't have is dropped, and any
/// dim of size 1 in the operand wraps to index 0 regardless of the output's
/// index at that axis.
fn broadcast_index(out_idx: &[usize], out_shape: &[usize], operand_shape: &[usize]) -> Vec<usize> {
    let offset = out_shape.len() - operand_shape.len();
    operand_shape
        .iter()
        .enumerate()
        .map(|(i, &dim)| if dim == 1 { 0 } else { out_idx[i + offset] })
        .collect()
}

/// Increment a row-major multi-index in place (odometer-style), matching
/// `storage.rs`/`tape.rs`'s own iteration order.
pub(crate) fn increment_index(idx: &mut [usize], shape: &[usize]) {
    for i in (0..idx.len()).rev() {
        idx[i] += 1;
        if idx[i] < shape[i] {
            return;
        }
        idx[i] = 0;
    }
}

/// Read `storage` as `f64` at `storage`'s own (already-resolved) logical
/// index, given the output-space index and shape.
fn read_broadcast(storage: &NativeStorage, out_idx: &[usize], out_shape: &[usize]) -> f64 {
    let idx = broadcast_index(out_idx, out_shape, &storage.shape);
    storage.get(&idx)
}

/// Build a contiguous `NativeStorage` by applying `f(lhs_val, rhs_val)` over
/// every logical index in `out_shape`, reading each operand through its own
/// broadcast-resolved index (no pre-materialized broadcast copy).
fn elementwise_binary(
    lhs: &NativeStorage,
    rhs: &NativeStorage,
    out_shape: &[usize],
    f: impl Fn(f64, f64) -> f64,
) -> NativeStorage {
    let total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..total {
        let a = read_broadcast(lhs, &idx, out_shape);
        let b = read_broadcast(rhs, &idx, out_shape);
        out.push(f(a, b) as f32);
        if !out_shape.is_empty() {
            increment_index(&mut idx, out_shape);
        }
    }
    NativeStorage::from_contiguous(NativeBuffer::F32(out), out_shape.to_vec())
}

/// Elementwise negate (used by `sub`'s backward rule: rhs receives the
/// negated incoming gradient before unbroadcasting).
fn negate(t: &NativeStorage) -> NativeStorage {
    let total: usize = t.shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; t.shape.len()];
    for _ in 0..total {
        out.push(-t.get(&idx) as f32);
        if !t.shape.is_empty() {
            increment_index(&mut idx, &t.shape);
        }
    }
    NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone())
}

/// Elementwise multiply of two ALREADY-shape-matching (or one of them
/// broadcastable to the other's shape) storages — used by `mul`'s backward
/// rule to compute `grad_out * other_operand`, aligned to `grad_out`'s shape.
fn mul_elementwise_broadcast(grad_out: &NativeStorage, other: &NativeStorage) -> NativeStorage {
    elementwise_binary(grad_out, other, &grad_out.shape, |a, b| a * b)
}

impl<T: DType, D: kindle_core::prelude::Device> NumericOps<Self> for NativeBackend<T, D> {
    fn add<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary(lhs, rhs, &out_shape, |a, b| a + b);

        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![
                    tape::unbroadcast(grad_out, &lhs_shape).expect("unbroadcast lhs (add)"),
                    tape::unbroadcast(grad_out, &rhs_shape).expect("unbroadcast rhs (add)"),
                ]
            }),
        });
        Ok(out)
    }

    fn sub<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary(lhs, rhs, &out_shape, |a, b| a - b);

        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![
                    tape::unbroadcast(grad_out, &lhs_shape).expect("unbroadcast lhs (sub)"),
                    tape::unbroadcast(&negate(grad_out), &rhs_shape)
                        .expect("unbroadcast rhs (sub)"),
                ]
            }),
        });
        Ok(out)
    }

    fn mul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary(lhs, rhs, &out_shape, |a, b| a * b);

        // Capture cloned copies of lhs/rhs's NativeStorage (cheap, Rc-backed)
        // since the backward closure needs their VALUES, not just shapes.
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let grad_lhs = mul_elementwise_broadcast(grad_out, &rhs_capture);
                let grad_rhs = mul_elementwise_broadcast(grad_out, &lhs_capture);
                vec![
                    tape::unbroadcast(&grad_lhs, &lhs_shape).expect("unbroadcast lhs (mul)"),
                    tape::unbroadcast(&grad_rhs, &rhs_shape).expect("unbroadcast rhs (mul)"),
                ]
            }),
        });
        Ok(out)
    }

    fn div<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary(lhs, rhs, &out_shape, |a, b| a / b);

        // Per Assumption A2 (RESEARCH.md): implemented for trait-completeness
        // via the standard quotient rule (1/rhs, -lhs/rhs^2), each
        // unbroadcast — best-effort correctness, not exercised by this
        // phase's example/tests.
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                // d(lhs/rhs)/dlhs = 1/rhs -> grad_lhs = grad_out / rhs
                let grad_lhs =
                    elementwise_binary(grad_out, &rhs_capture, &grad_out.shape, |g, r| g / r);
                // d(lhs/rhs)/drhs = -lhs/rhs^2 -> grad_rhs = grad_out * (-lhs/rhs^2)
                let grad_rhs = elementwise_binary(
                    grad_out,
                    &elementwise_binary(&lhs_capture, &rhs_capture, &grad_out.shape, |l, r| {
                        -l / (r * r)
                    }),
                    &grad_out.shape,
                    |g, dr| g * dr,
                );
                vec![
                    tape::unbroadcast(&grad_lhs, &lhs_shape).expect("unbroadcast lhs (div)"),
                    tape::unbroadcast(&grad_rhs, &rhs_shape).expect("unbroadcast rhs (div)"),
                ]
            }),
        });
        Ok(out)
    }
}

impl<T: DType, D: kindle_core::prelude::Device> FloatOps<Self> for NativeBackend<T, D> {
    fn add_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            out.push((t.get(&idx) + scalar) as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            // Gradient passes through unchanged (same shape, no unbroadcast
            // needed — scalar ops don't change shape).
            backward: Box::new(move |grad_out: &NativeStorage| vec![grad_out.clone()]),
        });
        Ok(out)
    }

    fn mul_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            out.push((t.get(&idx) * scalar) as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            // Gradient scales by the same constant (same shape, no
            // unbroadcast needed).
            backward: Box::new(move |grad_out: &NativeStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut scaled = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    scaled.push((grad_out.get(&idx) * scalar) as f32);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(scaled),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    fn relu<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("relu not implemented for NativeBackend")
    }
    fn gelu<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("gelu not implemented for NativeBackend")
    }
    fn abs<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("abs not implemented for NativeBackend")
    }
    fn exp<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("exp not implemented for NativeBackend")
    }
    fn neg<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("neg not implemented for NativeBackend")
    }
    fn sqrt<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("sqrt not implemented for NativeBackend")
    }
    fn log<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("log not implemented for NativeBackend")
    }
    fn tanh<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("tanh not implemented for NativeBackend")
    }
    fn sigmoid<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("sigmoid not implemented for NativeBackend")
    }
    fn swish<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("swish not implemented for NativeBackend")
    }
    fn softmax<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("softmax not implemented for NativeBackend")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape;

    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![rows, cols])
    }

    fn vector(v: Vec<f32>) -> NativeStorage {
        let len = v.len();
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![len])
    }

    fn f32_vec(s: &NativeStorage) -> Vec<f32> {
        match &*s.buffer {
            NativeBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    type TestBackend = NativeBackend<f32, kindle_core::prelude::Cpu>;

    #[test]
    fn add_broadcasts_forward_correctly() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![10.0, 20.0, 30.0]);
        let out = TestBackend::add::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(f32_vec(&out), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn add_backward_unbroadcasts_correctly_for_bias_vector_case() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![10.0, 20.0, 30.0]);
        let out = TestBackend::add::<f32>(&lhs, &rhs).unwrap();

        let grads = tape::backward(&out).unwrap();
        let lhs_grad = grads.get(lhs.id).expect("lhs should have a gradient");
        let rhs_grad = grads.get(rhs.id).expect("rhs should have a gradient");

        // lhs's grad: ones_like(out) unbroadcast to [2,3] = all ones.
        assert_eq!(lhs_grad.shape, vec![2, 3]);
        assert_eq!(f32_vec(lhs_grad), vec![1.0; 6]);

        // rhs's grad: ones_like(out) [2,3] unbroadcast (summed) to [3] = [2,2,2].
        assert_eq!(rhs_grad.shape, vec![3]);
        assert_eq!(f32_vec(rhs_grad), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn sub_forward_computes_elementwise_difference_with_broadcast() {
        let lhs = matrix(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3);
        let rhs = vector(vec![1.0, 2.0, 3.0]);
        let out = TestBackend::sub::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(f32_vec(&out), vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);
    }

    #[test]
    fn sub_backward_negates_rhs_contribution() {
        let lhs = vector(vec![10.0, 20.0, 30.0]);
        let rhs = vector(vec![1.0, 2.0, 3.0]);
        let out = TestBackend::sub::<f32>(&lhs, &rhs).unwrap();

        let grads = tape::backward(&out).unwrap();
        let lhs_grad = grads.get(lhs.id).unwrap();
        let rhs_grad = grads.get(rhs.id).unwrap();

        assert_eq!(f32_vec(lhs_grad), vec![1.0, 1.0, 1.0]);
        assert_eq!(f32_vec(rhs_grad), vec![-1.0, -1.0, -1.0]);
    }

    #[test]
    fn mul_forward_computes_elementwise_product_with_broadcast() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![2.0, 3.0, 4.0]);
        let out = TestBackend::mul::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(f32_vec(&out), vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0]);
    }

    #[test]
    fn mul_backward_uses_other_operands_real_values() {
        // d(a*b)/da = b, d(a*b)/db = a — verify the retrieved gradient
        // equals a manually-computed expected value (not merely "some
        // gradient exists").
        let a = vector(vec![2.0, 3.0, 4.0]);
        let b = vector(vec![5.0, 6.0, 7.0]);
        let out = TestBackend::mul::<f32>(&a, &b).unwrap();

        let grads = tape::backward(&out).unwrap();
        let a_grad = grads.get(a.id).unwrap();
        let b_grad = grads.get(b.id).unwrap();

        // grad_out is ones_like(out) = [1,1,1].
        // da = grad_out * b = [5,6,7]
        assert_eq!(f32_vec(a_grad), vec![5.0, 6.0, 7.0]);
        // db = grad_out * a = [2,3,4]
        assert_eq!(f32_vec(b_grad), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn mul_backward_with_broadcast_bias_vector_case() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![2.0, 3.0, 4.0]);
        let out = TestBackend::mul::<f32>(&lhs, &rhs).unwrap();

        let grads = tape::backward(&out).unwrap();
        let lhs_grad = grads.get(lhs.id).unwrap();
        let rhs_grad = grads.get(rhs.id).unwrap();

        // grad_out = ones_like(out) = [[1,1,1],[1,1,1]]
        // d(lhs*rhs)/dlhs = rhs broadcast -> unbroadcast to lhs shape [2,3]
        // (no reduction needed since lhs shape == out shape): [[2,3,4],[2,3,4]]
        assert_eq!(lhs_grad.shape, vec![2, 3]);
        assert_eq!(f32_vec(lhs_grad), vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]);

        // d(lhs*rhs)/drhs = lhs broadcast, summed (unbroadcast) to rhs shape [3]:
        // col sums of lhs = [1+4, 2+5, 3+6] = [5,7,9]
        assert_eq!(rhs_grad.shape, vec![3]);
        assert_eq!(f32_vec(rhs_grad), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn add_scalar_float_forward_and_backward() {
        let t = vector(vec![1.0, 2.0, 3.0]);
        let out = TestBackend::add_scalar_float::<f32>(&t, 1.0).unwrap();
        assert_eq!(f32_vec(&out), vec![2.0, 3.0, 4.0]);

        let grads = tape::backward(&out).unwrap();
        let t_grad = grads.get(t.id).unwrap();
        // Gradient passes through unchanged.
        assert_eq!(f32_vec(t_grad), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn mul_scalar_float_forward_and_backward() {
        let t = vector(vec![1.0, 2.0, 3.0]);
        let out = TestBackend::mul_scalar_float::<f32>(&t, 2.5).unwrap();
        assert_eq!(f32_vec(&out), vec![2.5, 5.0, 7.5]);

        let grads = tape::backward(&out).unwrap();
        let t_grad = grads.get(t.id).unwrap();
        // Gradient scales by the same constant.
        assert_eq!(f32_vec(t_grad), vec![2.5, 2.5, 2.5]);
    }
}
