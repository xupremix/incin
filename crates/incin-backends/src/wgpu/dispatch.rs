use alloc::sync::Arc;
use wgpu::ComputePipeline;

use crate::wgpu::device::get_device_state;
use crate::wgpu::pipeline::get_or_create_pipeline;
use incin_core::error::Result;
use incin_core::shapes::{OperationKind, ShapeError};
use incin_core::tensor::dtype::DTypeId;

use crate::wgpu::storage::WgpuBuffer;

/// `WG_SIZE`.
const WG_SIZE: u32 = 256;

fn checked_workgroups(
    factors: &[u32],
    workgroup_size: u32,
    expression: &'static str,
) -> Result<u32> {
    let total = factors.iter().try_fold(1u64, |total, &factor| {
        total
            .checked_mul(u64::from(factor))
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Storage,
                expression,
            })
    })?;
    u32::try_from(total.div_ceil(u64::from(workgroup_size))).map_err(|_| {
        ShapeError::ArithmeticOverflow {
            operation: OperationKind::Storage,
            expression,
        }
        .into()
    })
}

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
pub(crate) fn dispatch_reduce_all(
    inp: &WgpuBuffer,
    n: u32,
    reduce_mode: u32,
) -> Result<Arc<WgpuBuffer>> {
    let state = get_device_state();
    let shader = include_str!("shaders/reduce.wgsl");
    let pipeline = get_or_create_pipeline("reduce", shader, "main");

    // First pass: reduce into n_wg partial results
    let n_wg = n.div_ceil(WG_SIZE);
    let partial_buf =
        WgpuBuffer::new_zeros_for(DTypeId::F32, n_wg as usize, OperationKind::Reduction)?;
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
        return Ok(partial_buf);
    }

    // Second pass: reduce partials
    dispatch_reduce_all(&partial_buf, n_wg, reduce_mode)
}

/// `run_pipeline`.
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
pub(crate) fn dispatch_im2col(
    inp: &WgpuBuffer,
    col: &WgpuBuffer,
    params_data: &[u32],
) -> Result<()> {
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
    let wg = checked_workgroups(
        &[
            params_data[0],
            params_data[1],
            params_data[4],
            params_data[5],
            params_data[6],
            params_data[7],
        ],
        256,
        "WGPU im2col dispatch size",
    )?;
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Im2Col");
    Ok(())
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
) -> Result<[u32; 21]> {
    if out_shape.len() > 6 || inp_shape.len() > 6 || aux.len() > 6 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Storage,
            parameter: "WGPU shape-kernel rank",
            value: core::cmp::max(out_shape.len(), core::cmp::max(inp_shape.len(), aux.len())),
        }
        .into());
    }
    let rank = crate::wgpu::backend::checked_u32(
        core::cmp::max(out_shape.len(), inp_shape.len()),
        "WGPU shape-kernel rank",
    )?;
    let mut params = [0u32; 21];
    params[0] = op_mode;
    params[1] = rank;
    params[2] = n_elements;

    let pad_out = 6 - out_shape.len();
    for (i, &s) in out_shape.iter().enumerate() {
        params[3 + pad_out + i] = crate::wgpu::backend::checked_u32(s, "WGPU output dimension")?;
    }
    for i in 0..pad_out {
        params[3 + i] = 1;
    }

    let pad_inp = 6 - inp_shape.len();
    for (i, &s) in inp_shape.iter().enumerate() {
        params[9 + pad_inp + i] = crate::wgpu::backend::checked_u32(s, "WGPU input dimension")?;
    }
    for i in 0..pad_inp {
        params[9 + i] = 1;
    }

    let pad_aux = 6 - aux.len();
    for (i, &s) in aux.iter().enumerate() {
        let mut val = crate::wgpu::backend::checked_u32(s, "WGPU shape auxiliary value")?;
        if op_mode == 2 {
            val = val
                .checked_add(crate::wgpu::backend::checked_u32(
                    pad_out,
                    "WGPU transpose padding",
                )?)
                .ok_or(ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Transpose,
                    expression: "WGPU transpose axis plus padding",
                })?;
        }
        params[15 + pad_aux + i] = val;
    }
    for i in 0..pad_aux {
        params[15 + i] = 0;
    }

    Ok(params)
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

#[allow(clippy::too_many_arguments)]
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
) -> Result<()> {
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
    let wg = checked_workgroups(&[n, c, oh, ow], WG_SIZE, "WGPU pooling dispatch size")?;
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Pool2D");
    Ok(())
}

pub(crate) fn dispatch_conv2d_direct(
    inp: &WgpuBuffer,
    weight: &WgpuBuffer,
    out: &Arc<WgpuBuffer>,
    params: &[u32; 16],
) -> Result<()> {
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
    let wg = checked_workgroups(
        &[params[0], params[4], params[5], params[6]],
        WG_SIZE,
        "WGPU convolution dispatch size",
    )?;
    run_pipeline(&state, &pipeline, &bg, wg, 1, 1, "Conv2DDirect");
    Ok(())
}
