use alloc::sync::Arc;
use wgpu::ComputePipeline;

use crate::device::get_device_state;
use crate::storage::WgpuBuffer;
use crate::pipeline::get_or_create_pipeline;

const WG_SIZE: u32 = 256;

/// Run a simple 1D dispatch with 3 storage bindings: lhs, rhs, out, plus a u32 params buffer.
pub(crate) fn dispatch_binary(
    lhs: &WgpuBuffer,
    rhs: &WgpuBuffer,
    out: &Arc<WgpuBuffer>,
    params_data: &[u32],
) {
    let state = get_device_state();
    let shader = include_str!("shaders/binary.wgsl");
    let pipeline = get_or_create_pipeline("binary", shader, "main");

    let params_buf = WgpuBuffer::from_slice(params_data);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Binary BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: lhs.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: rhs.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.buffer.as_entire_binding() },
        ],
    });
    let n = params_data[1];
    let wg = (n + WG_SIZE - 1) / WG_SIZE;
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Binary");
}

/// Run a 1D unary dispatch: inp, out, params
pub(crate) fn dispatch_unary(inp: &WgpuBuffer, out: &Arc<WgpuBuffer>, params_data: &[u32]) {
    let state = get_device_state();
    let shader = include_str!("shaders/unary.wgsl");
    let pipeline = get_or_create_pipeline("unary", shader, "main");

    let params_buf = WgpuBuffer::from_slice(params_data);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Unary BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.buffer.as_entire_binding() },
        ],
    });
    let n = params_data[1];
    let wg = (n + WG_SIZE - 1) / WG_SIZE;
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Unary");
}

/// Run a scalar op: inp, out, params (op_mode, n, scalar_bits)
pub(crate) fn dispatch_scalar(inp: &WgpuBuffer, out: &Arc<WgpuBuffer>, params_data: &[u32]) {
    let state = get_device_state();
    let shader = include_str!("shaders/scalar.wgsl");
    let pipeline = get_or_create_pipeline("scalar", shader, "main");

    let params_buf = WgpuBuffer::from_slice(params_data);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Scalar BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.buffer.as_entire_binding() },
        ],
    });
    let n = params_data[1];
    let wg = (n + WG_SIZE - 1) / WG_SIZE;
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Scalar");
}

/// Full reduction over `n` elements. Returns a scalar WgpuStorage (shape=[1]).
/// reduce_mode: 0=sum, 1=max, 2=min
pub(crate) fn dispatch_reduce_all(inp: &WgpuBuffer, n: u32, reduce_mode: u32) -> Arc<WgpuBuffer> {
    let state = get_device_state();
    let shader = include_str!("shaders/reduce.wgsl");
    let pipeline = get_or_create_pipeline("reduce", shader, "main");

    // First pass: reduce into n_wg partial results
    let n_wg = (n + WG_SIZE - 1) / WG_SIZE;
    let partial_buf = WgpuBuffer::new_zeros(n_wg as usize * 4);
    let params = [n, reduce_mode];
    let params_buf = WgpuBuffer::from_slice(&params);

    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Reduce BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: partial_buf.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.buffer.as_entire_binding() },
        ],
    });
    run_pipeline(&state, &pipeline, &bg, n_wg, 1, 1, "Reduce-pass1");

    // If already a single element, done
    if n_wg == 1 {
        return partial_buf;
    }

    // Second pass: reduce partials
    dispatch_reduce_all(&partial_buf, n_wg, reduce_mode)
}

/// Dispatch softmax: shape [batch, n]
pub(crate) fn dispatch_softmax(inp: &WgpuBuffer, out: &Arc<WgpuBuffer>, batch: u32, n: u32) {
    let state = get_device_state();
    let shader = include_str!("shaders/softmax.wgsl");
    let pipeline = get_or_create_pipeline("softmax", shader, "main");

    let params = [batch, n];
    let params_buf = WgpuBuffer::from_slice(&params);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Softmax BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.buffer.as_entire_binding() },
        ],
    });
    // Each workgroup handles one row
    run_pipeline(&state, &pipeline, &bg, batch, 1, 1, "Softmax");
}

/// Dispatch tiled 2D transpose
pub(crate) fn dispatch_transpose(inp: &WgpuBuffer, out: &Arc<WgpuBuffer>, rows: u32, cols: u32) {
    let state = get_device_state();
    let shader = include_str!("shaders/transpose.wgsl");
    let pipeline = get_or_create_pipeline("transpose", shader, "main");

    let params = [rows, cols];
    let params_buf = WgpuBuffer::from_slice(&params);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Transpose BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.buffer.as_entire_binding() },
        ],
    });
    let wg_x = (cols + 15) / 16;
    let wg_y = (rows + 15) / 16;
    run_pipeline(&state, &pipeline, &bg, wg_x, wg_y, 1, "Transpose");
}

fn run_pipeline(
    state: &crate::device::WgpuDeviceState,
    pipeline: &ComputePipeline,
    bg: &wgpu::BindGroup,
    x: u32, y: u32, z: u32,
    label: &str,
) {
    let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(label),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bg, &[]);
        cpass.dispatch_workgroups(x, y, z);
    }
    state.queue.submit(core::iter::once(encoder.finish()));
    // Block until the GPU finishes so that any immediately-following CPU
    // readback (`to_vec`) sees the results of this dispatch.
    state.device.poll(wgpu::Maintain::Wait);
}

/// im2col: inp [N,C,H,W] -> col [N, C*Kh*Kw, H_out*W_out]
/// params: [N, C, H_in, W_in, H_out, W_out, Kh, Kw, sh, sw, ph, pw, dh, dw]
pub(crate) fn dispatch_im2col(inp: &WgpuBuffer, col: &WgpuBuffer, params_data: &[u32]) {
    let state = get_device_state();
    let shader = include_str!("shaders/im2col.wgsl");
    let pipeline = get_or_create_pipeline("im2col", shader, "main");

    let params_buf = WgpuBuffer::from_slice(params_data);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Im2Col BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: col.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.buffer.as_entire_binding() },
        ],
    });
    // total threads = N * C*Kh*Kw * H_out*W_out
    let n       = params_data[0] as u64;
    let c_in    = params_data[1] as u64;
    let h_out   = params_data[4] as u64;
    let w_out   = params_data[5] as u64;
    let kh      = params_data[6] as u64;
    let kw      = params_data[7] as u64;
    let total   = n * c_in * kh * kw * h_out * w_out;
    let wg = ((total + 255) / 256) as u32;
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Im2Col");
}

/// Fused GPU AdamW.
/// param, grad, m, v must all be length-N f32 buffers.
/// meta: [N, lr_bits, beta1_bits, beta2_bits, eps_bits, wd_bits, bc1_bits, bc2_bits]
pub(crate) fn dispatch_adamw(
    param: &WgpuBuffer,
    grad:  &WgpuBuffer,
    m:     &WgpuBuffer,
    v:     &WgpuBuffer,
    meta:  &[u32],
) {
    let state = get_device_state();
    let shader = include_str!("shaders/adamw.wgsl");
    let pipeline = get_or_create_pipeline("adamw", shader, "main");

    let meta_buf = WgpuBuffer::from_slice(meta);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("AdamW BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: param.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: grad.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: m.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: meta_buf.buffer.as_entire_binding() },
        ],
    });
    let n = meta[0];
    let wg = (n + 255) / 256;
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "AdamW");
}

