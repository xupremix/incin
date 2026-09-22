use alloc::vec::Vec;
use core::cell::RefCell;

use incin_core::error::Result;
use incin_core::exec::tape;
use incin_core::exec::{GradientMap, Tape, TapeNode, TapeStorage};
use incin_core::shapes::OperationKind;
use incin_core::tensor::dtype::DTypeId;

use crate::wgpu::dispatch;
use crate::wgpu::storage::{TensorId, WgpuBuffer, WgpuStorage};

/// One recorded operation, as the core defines it.
pub(crate) type TapeEntry = TapeNode<WgpuStorage>;

thread_local! {
    static TAPE: RefCell<Tape<WgpuStorage>> = const { RefCell::new(Tape::new()) };
}

/// The three things a reverse walk needs of WGPU storage.
impl TapeStorage for WgpuStorage {
    fn id(&self) -> TensorId {
        self.id
    }

    fn ones_like(&self) -> Result<Self> {
        let n = crate::wgpu::backend::num_elements(&self.shape)?;
        let data: Vec<f32> = vec![1.0; n];
        let buf = WgpuBuffer::from_slice(&data);
        Ok(WgpuStorage::new(buf, self.shape.to_vec()))
    }

    fn accumulate(&self, contribution: &Self) -> Result<Self> {
        add_wgpu_storage(self, contribution)
    }

    fn has_non_finite(&self) -> Result<bool> {
        let data: Vec<f32> = self.buffer.to_vec()?;
        Ok(data.iter().any(|x| x.is_nan() || x.is_infinite()))
    }
}

/// Number of entries currently on the tape.
#[must_use]
pub fn depth() -> usize {
    TAPE.with(|t| t.borrow().depth())
}

#[cfg(feature = "telemetry")]
thread_local! {
    static BACKWARD_STEP: RefCell<usize> = const { RefCell::new(0) };
}

/// Record one operation, building the entry only if it will be kept.
///
/// [`push`] already discards entries when the effective `GradMode` does not
/// record, but by then the entry exists: its saved shapes, its input-id vector
/// and its boxed backward closure have all been allocated for a value nothing
/// can read. See `cpu::tape::push_with`, which this mirrors.
pub(crate) fn push_with(entry: impl FnOnce() -> TapeEntry) {
    if !incin_core::exec::GradMode::current().records() {
        return;
    }
    push(entry());
}

/// Push a `TapeEntry` unless `GradMode` forbids recording.
pub fn push(entry: TapeEntry) {
    TAPE.with(|t| t.borrow_mut().push(entry));
    #[cfg(feature = "telemetry")]
    {
        let depth = depth() as f64;
        let step = BACKWARD_STEP.with(|s| *s.borrow());
        crate::telemetry::emit_scalar(step, "tape/depth", depth);
    }
}

/// Record a custom operation's backward recipe on this thread's tape.
///
/// The downstream half of the custom-training contract, mirroring
/// `cpu::tape_record`: a foreign `Execute` implementation runs its forward
/// kernel, then calls this with a `TapeNode` whose recipe maps one output
/// gradient to one gradient per input. The node joins the same tape the
/// built-in kernels record on, under the same `GradMode` gate, so mixed
/// graphs walk as one graph. Recipes should stay in-kernel (broadcast,
/// scale, elementwise launches): every host value access is a readback.
pub fn record(entry: TapeNode<WgpuStorage>) {
    push(entry);
}

/// Record a custom operation's backward recipe, building it only if kept.
///
/// The lazy form of [`record`](self::record): the entry closure runs only
/// when the ambient `GradMode` records.
pub fn record_with(entry: impl FnOnce() -> TapeNode<WgpuStorage>) {
    push_with(entry);
}

impl<D, K> incin_core::backend_authoring::RecordingBackend<K> for super::WgpuBackendImpl<D>
where
    D: incin_core::tensor::device::Device,
    K: incin_core::tensor::dtype::DType,
{
    fn record_custom(node: TapeNode<WgpuStorage>) {
        push(node);
    }
}

pub struct WgpuGrads {
    pub(crate) grads: GradientMap<WgpuStorage>,
}

impl WgpuGrads {
    /// Look up the accumulated gradient for a given tensor id, if any.
    /// Replace the gradient recorded for `id`.
    ///
    /// A replacement rather than an accumulation, which is why it is spelled
    /// differently from anything the reverse walk calls. See
    /// `AutogradBackend::set_grad`.
    pub fn set(&mut self, id: TensorId, value: WgpuStorage) {
        self.grads.insert(id, value);
    }

    pub fn get(&self, id: TensorId) -> Option<&WgpuStorage> {
        self.grads.get(id)
    }

    /// How many tensors the backward pass reached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Whether it reached none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }
}

pub fn backward(loss: &WgpuStorage) -> Result<WgpuGrads> {
    #[cfg(feature = "telemetry")]
    let n_ops = depth();

    let nodes = TAPE.with(|t| t.borrow_mut().drain_reachable(loss.id()));
    let grads = incin_core::exec::GradMode::Disabled.scope(|| tape::backward(nodes, loss))?;

    #[cfg(feature = "telemetry")]
    {
        let step = BACKWARD_STEP.with(|s| {
            let cur = *s.borrow();
            *s.borrow_mut() += 1;
            cur
        });
        emit_backward_telemetry(step, n_ops);
    }

    Ok(WgpuGrads { grads })
}

/// Walk the reachable graph with an explicit output cotangent.
pub fn backward_with(loss: &WgpuStorage, seed: &WgpuStorage) -> Result<WgpuGrads> {
    let nodes = TAPE.with(|t| t.borrow_mut().drain_reachable(loss.id()));
    let grads = incin_core::exec::GradMode::Disabled
        .scope(|| tape::backward_with_seed(nodes, loss, seed))?;
    Ok(WgpuGrads { grads })
}

#[cfg(feature = "telemetry")]
fn emit_backward_telemetry(step: usize, n_ops: usize) {
    crate::telemetry::emit_scalar(step, "tape/ops", n_ops as f64);
    #[cfg(feature = "std")]
    {
        if let Some(g) = incin_core::backend_authoring::tracing_graph_snapshot() {
            crate::telemetry::emit_graph_snapshot(g);
        }
    }
}

fn add_wgpu_storage(a: &WgpuStorage, b: &WgpuStorage) -> Result<WgpuStorage> {
    debug_assert_eq!(
        a.shape, b.shape,
        "tape accumulation requires matching shapes"
    );
    let n = crate::wgpu::backend::num_elements(&a.shape)?;
    let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, n, OperationKind::Storage)?;
    let params = [
        0,
        u32::try_from(n).map_err(|_| {
            incin_core::error::Error::Msg("WGPU launch element count exceeds u32".into())
        })?,
    ]; // op_mode 0=add
    dispatch::dispatch_binary(&a.buffer, &b.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, a.shape.to_vec()))
}

/// `pub(crate)`, as on CPU and CUDA.
///
/// It was the one of the four that was public, and the four are not one
/// contract: they differ in how a reduced-all-the-way scalar seed is expanded
/// back and in which reduce kernel they reach for. Exporting them as if they
/// were one API is how a downstream recipe comes to depend on this backend's
/// edge cases and finds another's. If un-broadcasting is ever offered
/// downstream it belongs on a trait beside `TapeStorage`, written once.
pub(crate) fn unbroadcast(grad: &WgpuStorage, target_shape: &[usize]) -> Result<WgpuStorage> {
    if grad.shape == target_shape {
        return Ok(grad.clone());
    }

    let ndim_diff = grad.shape.len().saturating_sub(target_shape.len());
    let mut result = grad.clone();

    // Reduce leading dims
    for _ in 0..ndim_diff {
        result = sum_dim_squeeze(&result, 0)?;
    }

    // Reduce keepdim dims
    for (i, &t_dim) in target_shape.iter().enumerate() {
        if t_dim == 1 && result.shape[i] != 1 {
            result = sum_dim_keepdim(&result, i)?;
        }
    }

    if result.shape[..] == target_shape[..] {
        return Ok(result);
    }

    // Expand what reduction left smaller: a reduced-all-the-way scalar seed
    // reaches here with fewer elements than its target, and the kernels
    // downstream do not broadcast scalars implicitly the way the CPU ones
    // do, so handing the scalar on produces a shape the next launch refuses.
    // `broadcast_shape` checks compatibility first, because the materializer
    // assumes a legal target and would otherwise read out of bounds on a
    // genuinely incompatible shape. Mirrors the CUDA tail (`cuda/tape.rs`).
    crate::layout::broadcast_shape(&result.shape, target_shape)?;
    crate::wgpu::backend::broadcast_storage(&result, target_shape)
}

fn sum_dim_squeeze(storage: &WgpuStorage, axis: usize) -> Result<WgpuStorage> {
    let reduced = sum_dim_keepdim(storage, axis)?;
    let mut new_shape = reduced.shape.to_vec();
    new_shape.remove(axis);
    Ok(WgpuStorage::new(reduced.buffer.clone(), new_shape))
}

fn sum_dim_keepdim(storage: &WgpuStorage, axis: usize) -> Result<WgpuStorage> {
    let mut out_shape = storage.shape.to_vec();
    out_shape[axis] = 1;
    let total: usize = incin_core::shapes::ShapeBuf::from_slice(&(out_shape))
        .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;
    let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, total, OperationKind::Storage)?;

    let inner_stride: usize =
        incin_core::shapes::ShapeBuf::from_slice(&(storage.shape[axis + 1..]))
            .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;

    let axis_len = u32::try_from(storage.shape[axis]).map_err(|_| {
        incin_core::error::Error::Msg("WGPU reduction axis length exceeds u32".into())
    })?;
    let inner_stride = u32::try_from(inner_stride)
        .map_err(|_| incin_core::error::Error::Msg("WGPU reduction stride exceeds u32".into()))?;
    let total = u32::try_from(total).map_err(|_| {
        incin_core::error::Error::Msg("WGPU reduction output length exceeds u32".into())
    })?;
    dispatch::dispatch_reduce_dim(
        &storage.buffer,
        &out_buf,
        0, // sum
        axis_len,
        inner_stride,
        total,
    );
    Ok(WgpuStorage::new(out_buf, out_shape))
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;

    /// `storage`.
    fn storage(values: &[f32], shape: &[usize]) -> WgpuStorage {
        let buffer =
            WgpuBuffer::try_from_slice(values).expect("a WGPU adapter is available for tape tests");
        WgpuStorage::new(buffer, shape.to_vec())
    }

    /// `scalar`.
    fn scalar(v: f32) -> WgpuStorage {
        storage(&[v], &[])
    }

    /// `vector`.
    fn vector(v: &[f32]) -> WgpuStorage {
        storage(v, &[v.len()])
    }

    /// `matrix`.
    fn matrix(v: &[f32], rows: usize, cols: usize) -> WgpuStorage {
        storage(v, &[rows, cols])
    }

    /// `read`.
    fn read(storage: &WgpuStorage) -> Vec<f32> {
        storage
            .buffer
            .to_vec::<f32>()
            .expect("reading a contiguous f32 buffer back must succeed")
    }

    // --- unbroadcast tail tests (#121) ---

    #[test]
    /// `unbroadcast_scalar_seed_is_materialized_to_full_width`.
    fn unbroadcast_scalar_seed_is_materialized_to_full_width() {
        // The #121 tail: a reduced-all-the-way scalar seed for a `[3]`
        // target must be materialized to full width here, because the WGPU
        // kernels do not broadcast scalars implicitly the way the CPU ones
        // do -- handing the scalar on produces a shape the next launch
        // refuses (`iteration_plan: expected [], got [3]`).
        let grad = scalar(2.0);
        let result = unbroadcast(&grad, &[3]).expect("a compatible scalar seed expands");
        assert_eq!(result.shape, vec![3]);
        assert_eq!(read(&result), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    /// `unbroadcast_incompatible_shapes_are_refused`.
    fn unbroadcast_incompatible_shapes_are_refused() {
        // A grad that no broadcast could have produced for this target must
        // refuse rather than hand a wrong-shaped gradient on to accumulation
        // (mirrors the CPU/CUDA refusal tests; see #121).
        let grad = vector(&[1.0, 2.0, 3.0]);
        assert!(unbroadcast(&grad, &[4]).is_err());
        let grad = matrix(&[1.0; 6], 2, 3);
        assert!(unbroadcast(&grad, &[4]).is_err());
    }

    #[test]
    /// `unbroadcast_bias_vector_b_n_to_n`.
    fn unbroadcast_bias_vector_b_n_to_n() {
        // The reduction half of unbroadcast, which the tail must not
        // disturb: grad shape [4,3] (B=4, N=3), summed back to [3].
        let grad = matrix(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            4,
            3,
        );
        let result = unbroadcast(&grad, &[3]).expect("the bias reduction runs");
        assert_eq!(result.shape, vec![3]);
        // Column sums: col0 = 1+4+7+10=22, col1 = 2+5+8+11=26, col2 = 3+6+9+12=30
        assert_eq!(read(&result), vec![22.0, 26.0, 30.0]);
    }
}
