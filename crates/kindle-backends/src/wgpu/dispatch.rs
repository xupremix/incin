use alloc::sync::Arc;
use wgpu::ComputePipeline;

use crate::wgpu::device::get_device_state;
use crate::wgpu::pipeline::get_or_create_pipeline;
use crate::wgpu::storage::WgpuBuffer;

/// Core abstraction for `WG_SIZE` within the Kindle framework..
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let n = params_data[1];
    let wg = n.div_ceil(WG_SIZE);
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let n = params_data[1];
    let wg = n.div_ceil(WG_SIZE);
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let n = params_data[1];
    let wg = n.div_ceil(WG_SIZE);
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Scalar");
}

/// Full reduction over `n` elements. Returns a scalar WgpuStorage (shape=[1]).
/// reduce_mode: 0=sum, 1=max, 2=min
pub(crate) fn dispatch_reduce_all(inp: &WgpuBuffer, n: u32, reduce_mode: u32) -> Arc<WgpuBuffer> {
    let state = get_device_state();
    let shader = include_str!("shaders/reduce.wgsl");
    let pipeline = get_or_create_pipeline("reduce", shader, "main");

    // First pass: reduce into n_wg partial results
    let n_wg = n.div_ceil(WG_SIZE);
    let partial_buf = WgpuBuffer::new_zeros(n_wg as usize * 4);
    let params = [n, reduce_mode];
    let params_buf = WgpuBuffer::from_slice(&params);

    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Reduce BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: partial_buf.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.buffer.as_entire_binding(),
            },
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
#[allow(dead_code)]
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    // Each workgroup handles one row
    run_pipeline(&state, &pipeline, &bg, batch, 1, 1, "Softmax");
}

/// Core abstraction for `run_pipeline` within the Kindle framework..
fn run_pipeline(
    state: &crate::wgpu::device::WgpuDeviceState,
    pipeline: &ComputePipeline,
    bg: &wgpu::BindGroup,
    x: u32,
    y: u32,
    z: u32,
    label: &str,
) {
    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: col.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    // total threads = N * C*Kh*Kw * H_out*W_out
    let n = params_data[0] as u64;
    let c_in = params_data[1] as u64;
    let h_out = params_data[4] as u64;
    let w_out = params_data[5] as u64;
    let kh = params_data[6] as u64;
    let kw = params_data[7] as u64;
    let total = n * c_in * kh * kw * h_out * w_out;
    let wg = total.div_ceil(256) as u32;
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Im2Col");
}

/// Fused GPU AdamW.
/// param, grad, m, v must all be length-N f32 buffers.
/// meta: [N, lr_bits, beta1_bits, beta2_bits, eps_bits, wd_bits, bc1_bits, bc2_bits]
pub(crate) fn dispatch_adamw(
    param: &WgpuBuffer,
    grad: &WgpuBuffer,
    m: &WgpuBuffer,
    v: &WgpuBuffer,
    meta: &[u32],
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grad.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: m.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: meta_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let n = meta[0];
    let wg = n.div_ceil(256);
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "AdamW");
}

/// Dispatch transposed 2D convolution
pub(crate) fn dispatch_conv_transpose2d(
    inp: &WgpuBuffer,
    weight: &WgpuBuffer,
    out: &WgpuBuffer,
    params_data: &[u32],
) {
    let state = get_device_state();
    let shader = include_str!("shaders/conv_transpose.wgsl");
    let pipeline = get_or_create_pipeline("conv_transpose", shader, "main");

    let params_buf = WgpuBuffer::from_slice(params_data);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ConvTranspose BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });

    let total_out = params_data[0] * params_data[2] * params_data[5] * params_data[6];
    let wg = total_out.div_ceil(64);
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "ConvTranspose");
}

pub(crate) fn dispatch_shape(inp: &WgpuBuffer, out: &Arc<WgpuBuffer>, params_data: &[u32; 21]) {
    let state = get_device_state();
    let shader = include_str!("shaders/shape.wgsl");
    let pipeline = get_or_create_pipeline("shape", shader, "main");

    let params_buf = WgpuBuffer::from_slice(params_data);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Shape BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let n = params_data[2];
    let wg = n.div_ceil(WG_SIZE);
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Shape");
}

pub(crate) fn prepare_shape_params(
    op_mode: u32,
    n_elements: u32,
    out_shape: &[usize],
    inp_shape: &[usize],
    aux: &[usize],
) -> [u32; 21] {
    let rank = core::cmp::max(out_shape.len(), inp_shape.len()) as u32;
    let mut params = [0u32; 21];
    params[0] = op_mode;
    params[1] = rank;
    params[2] = n_elements;

    let pad_out = 6 - out_shape.len();
    for (i, &s) in out_shape.iter().enumerate() {
        params[3 + pad_out + i] = s as u32;
    }
    for i in 0..pad_out {
        params[3 + i] = 1;
    }

    let pad_inp = 6 - inp_shape.len();
    for (i, &s) in inp_shape.iter().enumerate() {
        params[9 + pad_inp + i] = s as u32;
    }
    for i in 0..pad_inp {
        params[9 + i] = 1;
    }

    let pad_aux = 6 - aux.len();
    for (i, &s) in aux.iter().enumerate() {
        let mut val = s as u32;
        if op_mode == 2 {
            val += pad_out as u32;
        }
        params[15 + pad_aux + i] = val;
    }
    for i in 0..pad_aux {
        params[15 + i] = 0;
    }

    params
}

pub(crate) fn dispatch_reduce_dim(
    inp: &WgpuBuffer,
    out: &Arc<WgpuBuffer>,
    op_mode: u32,
    dim_size: u32,
    inner_stride: u32,
    out_n: u32,
) {
    let state = get_device_state();
    let shader = include_str!("shaders/reduce_dim.wgsl");
    let pipeline = get_or_create_pipeline("reduce_dim", shader, "main");

    let params_buf = WgpuBuffer::from_slice(&[op_mode, dim_size, inner_stride, out_n]);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ReduceDim BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let wg = out_n.div_ceil(WG_SIZE);
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "ReduceDim");
}

pub(crate) fn dispatch_embedding(
    indices: &WgpuBuffer,
    weight: &WgpuBuffer,
    out: &Arc<WgpuBuffer>,
    seq_len: u32,
    embed_dim: u32,
    vocab_size: u32,
) {
    let state = get_device_state();
    let shader = include_str!("shaders/embedding.wgsl");
    let pipeline = get_or_create_pipeline("embedding", shader, "main");
    let params_buf = WgpuBuffer::from_slice(&[seq_len, embed_dim, vocab_size]);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Embedding BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: indices.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let total = seq_len * embed_dim;
    let wg = total.div_ceil(WG_SIZE);
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Embedding");
}

#[allow(dead_code)]
pub(crate) fn dispatch_layer_norm(
    inp: &WgpuBuffer,
    gamma: &WgpuBuffer,
    beta: &WgpuBuffer,
    out: &Arc<WgpuBuffer>,
    eps: f32,
    norm_size: u32,
    has_bias: f32,
    batch_size: u32,
) {
    let state = get_device_state();
    let shader = include_str!("shaders/layer_norm.wgsl");
    let pipeline = get_or_create_pipeline("layer_norm", shader, "main");
    let params_buf = WgpuBuffer::from_slice(&[eps, norm_size as f32, has_bias, batch_size as f32]);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("LayerNorm BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gamma.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: beta.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let wg = batch_size.div_ceil(64); // workgroup size is 64
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "LayerNorm");
}

#[allow(dead_code)]
pub(crate) fn dispatch_batch_norm(
    inp: &WgpuBuffer,
    gamma: &WgpuBuffer,
    beta: &WgpuBuffer,
    rm: &WgpuBuffer,
    rv: &WgpuBuffer,
    out: &Arc<WgpuBuffer>,
    eps: f32,
    channels: u32,
    spatial: u32,
    batch: u32,
    has_gamma: f32,
    has_beta: f32,
    has_rm_rv: f32,
) {
    let state = get_device_state();
    let shader = include_str!("shaders/batch_norm.wgsl");
    let pipeline = get_or_create_pipeline("batch_norm", shader, "main");
    let params = [
        eps,
        channels as f32,
        spatial as f32,
        batch as f32,
        has_gamma,
        has_beta,
        has_rm_rv,
    ];
    let params_buf = WgpuBuffer::from_slice(&params);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BatchNorm BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gamma.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: beta.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: rm.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: rv.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let wg = channels.div_ceil(64); // workgroup size is 64
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "BatchNorm");
}

pub(crate) fn dispatch_pool2d(
    inp: &WgpuBuffer,
    out: &Arc<WgpuBuffer>,
    mode: u32,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
) {
    let state = get_device_state();
    let shader = include_str!("shaders/pool2d.wgsl");
    let pipeline = get_or_create_pipeline("pool2d", shader, "main");
    let params = [mode, n, c, h, w, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw];
    let params_buf = WgpuBuffer::from_slice(&params);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Pool2D BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let total = n * c * oh * ow;
    let wg = total.div_ceil(WG_SIZE);
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Pool2D");
}

pub(crate) fn dispatch_bias_add(
    t: &Arc<WgpuBuffer>,
    bias: &WgpuBuffer,
    batch: u32,
    channels: u32,
    spatial: u32,
) {
    let state = get_device_state();
    let shader = include_str!("shaders/bias_add.wgsl");
    let pipeline = get_or_create_pipeline("bias_add", shader, "main");
    let total = batch * channels * spatial;
    let params_buf = WgpuBuffer::from_slice(&[channels, spatial, total]);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BiasAdd BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: t.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bias.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let wg = total.div_ceil(WG_SIZE);
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "BiasAdd");
}

pub(crate) fn dispatch_conv2d_direct(
    inp: &WgpuBuffer,
    weight: &WgpuBuffer,
    out: &Arc<WgpuBuffer>,
    params: &[u32; 16],
) {
    let state = get_device_state();
    let shader = include_str!("shaders/conv2d_direct.wgsl");
    let pipeline = get_or_create_pipeline("conv2d_direct", shader, "main");
    let params_buf = WgpuBuffer::from_slice(params);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Conv2DDirect BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inp.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let total = params[0] * params[4] * params[5] * params[6];
    let wg = total.div_ceil(WG_SIZE);
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Conv2DDirect");
}

#[allow(dead_code)]
pub(crate) fn dispatch_nll_loss(
    log_sm: &WgpuBuffer,
    target: &WgpuBuffer,
    out: &Arc<WgpuBuffer>,
    batch: u32,
    n_classes: u32,
) {
    let state = get_device_state();
    let shader = include_str!("shaders/nll_loss.wgsl");
    let pipeline = get_or_create_pipeline("nll_loss", shader, "main");
    let params_buf = WgpuBuffer::from_slice(&[batch, n_classes]);
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("NLLLoss BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: log_sm.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.buffer.as_entire_binding(),
            },
        ],
    });
    let wg = batch.div_ceil(WG_SIZE);
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "NLLLoss");
}
