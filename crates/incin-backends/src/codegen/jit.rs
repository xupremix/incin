//! JIT compilation and dynamic kernel execution engine for IncinIR on CUDA and CPU.
//!
//! Compiles, caches, and executes custom operations represented by `KernelDefinition` on the fly,
//! providing zero-overhead execution for user-defined forward expressions and symbolically derived
//! backward autograd passes.

use super::ir::KernelDefinition;
#[cfg(feature = "cuda")]
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
#[cfg(feature = "cuda")]
use incin_core::shapes::OperationKind;

#[cfg(feature = "cuda")]
use crate::cuda::storage::{CudaBuffer, CudaStorage};

/// CUDA JIT compiled kernel instance.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaJitKernel {
    /// Kernel definition.
    pub definition: KernelDefinition,
    /// Device ID on which the kernel is loaded.
    pub device_id: usize,
    /// Cache module key.
    pub module_name: String,
}

#[cfg(feature = "cuda")]
impl CudaJitKernel {
    /// Compiles and loads a kernel definition on the specified CUDA device.
    pub fn compile(definition: KernelDefinition, device_id: usize) -> Result<Self> {
        let module_name = alloc::format!("jit_{}_{:?}", definition.name, definition.dtype);
        if crate::cuda::gpu::cuda_cache::get_module(device_id, &module_name).is_none() {
            let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
            let mut full_cuda_source = definition.render_forward_cuda();
            for i in 0..definition.input_arity {
                if let Some(bwd_src) = definition.render_backward_cuda(i) {
                    full_cuda_source.push_str("\n\n");
                    full_cuda_source.push_str(&bwd_src);
                }
            }
            dispatcher.compile_and_load_kernel(
                &definition.name,
                &full_cuda_source,
                &module_name,
            )?;
        }

        Ok(Self {
            definition,
            device_id,
            module_name,
        })
    }

    /// Executes the forward pass on the given input CUDA storages.
    pub fn launch_forward(&self, inputs: &[&CudaStorage]) -> Result<CudaStorage> {
        if inputs.len() != self.definition.input_arity {
            return Err(Error::Msg(alloc::format!(
                "JIT kernel {} expects {} inputs, got {}",
                self.definition.name,
                self.definition.input_arity,
                inputs.len()
            )));
        }
        let first_buf = &*inputs[0].buffer;
        let numel = crate::bytes::checked_numel(&inputs[0].shape)?;
        let stream = first_buf.device.default_stream();

        let mut output = CudaBuffer {
            len: numel,
            dtype: self.definition.dtype.into(),
            data: Arc::new(
                stream
                    .alloc_zeros::<u8>(crate::bytes::byte_len(
                        self.definition.dtype,
                        numel,
                        OperationKind::Pointwise,
                    )?)
                    .map_err(|e| Error::Msg(alloc::format!("CUDA JIT alloc failed: {e:?}")))?,
            ),
            device: first_buf.device.clone(),
            device_id: self.device_id,
        };

        if numel == 0 {
            return Ok(CudaStorage::new(Arc::new(output), inputs[0].shape.to_vec()));
        }

        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(self.device_id)?;
        let entry_point = alloc::format!("{}_forward", self.definition.name);
        let function = dispatcher.get_function(&self.module_name, &entry_point)?;

        // SAFETY: Inputs and output are validated contiguous buffers bounded by checked numel.
        unsafe {
            let output_u8 = Arc::get_mut(&mut output.data).ok_or_else(|| {
                Error::Msg("fresh CUDA JIT output was unexpectedly shared".into())
            })?;
            use cudarc::driver::PushKernelArg;
            let block_size = 256u32;
            let grid_size = (numel as u32).div_ceil(block_size);
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            let mut builder = stream.launch_builder(&function);
            for input in inputs {
                builder.arg(&*input.buffer.data);
            }
            builder.arg(&mut *output_u8);
            let numel_i32 =
                i32::try_from(numel).map_err(|_| Error::Msg("numel overflow".into()))?;
            builder.arg(&numel_i32);
            builder
                .launch(config)
                .map_err(|e| Error::Msg(alloc::format!("CUDA JIT forward launch failed: {e:?}")))?;
        }

        Ok(CudaStorage::new(Arc::new(output), inputs[0].shape.to_vec()))
    }

    /// Executes the symbolically generated backward pass for a given input tensor index.
    pub fn launch_backward(
        &self,
        grad_out: &CudaStorage,
        inputs: &[&CudaStorage],
        input_idx: usize,
    ) -> Result<CudaStorage> {
        if input_idx >= self.definition.input_arity {
            return Err(Error::Msg(alloc::format!(
                "JIT backward input_idx {input_idx} out of range (arity {})",
                self.definition.input_arity
            )));
        }
        let first_buf = &*inputs[0].buffer;
        let numel = crate::bytes::checked_numel(&inputs[0].shape)?;
        let stream = first_buf.device.default_stream();

        let mut grad_in = CudaBuffer {
            len: numel,
            dtype: self.definition.dtype.into(),
            data: Arc::new(
                stream
                    .alloc_zeros::<u8>(crate::bytes::byte_len(
                        self.definition.dtype,
                        numel,
                        OperationKind::Pointwise,
                    )?)
                    .map_err(|e| Error::Msg(alloc::format!("CUDA JIT alloc failed: {e:?}")))?,
            ),
            device: first_buf.device.clone(),
            device_id: self.device_id,
        };

        if numel == 0 {
            return Ok(CudaStorage::new(
                Arc::new(grad_in),
                inputs[input_idx].shape.to_vec(),
            ));
        }

        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(self.device_id)?;
        let entry_point = alloc::format!("{}_backward_{input_idx}", self.definition.name);
        let function = dispatcher.get_function(&self.module_name, &entry_point)?;

        // SAFETY: Checked tensor shapes and dimensions; unique output buffer allocation.
        unsafe {
            let grad_in_u8 = Arc::get_mut(&mut grad_in.data).ok_or_else(|| {
                Error::Msg("fresh CUDA JIT output was unexpectedly shared".into())
            })?;
            use cudarc::driver::PushKernelArg;
            let block_size = 256u32;
            let grid_size = (numel as u32).div_ceil(block_size);
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            let mut builder = stream.launch_builder(&function);
            builder.arg(&*grad_out.buffer.data);
            for input in inputs {
                builder.arg(&*input.buffer.data);
            }
            builder.arg(&mut *grad_in_u8);
            let numel_i32 =
                i32::try_from(numel).map_err(|_| Error::Msg("numel overflow".into()))?;
            builder.arg(&numel_i32);
            builder.launch(config).map_err(|e| {
                Error::Msg(alloc::format!("CUDA JIT backward launch failed: {e:?}"))
            })?;
        }

        Ok(CudaStorage::new(
            Arc::new(grad_in),
            inputs[input_idx].shape.to_vec(),
        ))
    }
}

/// Host CPU JIT Executor for arbitrary `KernelDefinition` expressions.
#[derive(Debug, Clone)]
pub struct CpuJitKernel {
    /// Kernel definition.
    pub definition: KernelDefinition,
}

impl CpuJitKernel {
    /// Creates a new CPU JIT kernel runner for a kernel definition.
    #[must_use]
    pub const fn new(definition: KernelDefinition) -> Self {
        Self { definition }
    }

    /// Evaluates forward pointwise operation on CPU slices of `f32`.
    pub fn eval_f32(&self, inputs: &[&[f32]], output: &mut [f32]) -> Result<()> {
        let len = output.len();
        for inp in inputs {
            if inp.len() != len {
                return Err(Error::Msg("Input slice length mismatch".into()));
            }
        }

        let mut args = alloc::vec![0.0f64; inputs.len()];
        for i in 0..len {
            for (arg_idx, inp) in inputs.iter().enumerate() {
                args[arg_idx] = inp[i] as f64;
            }
            output[i] = self.definition.forward.eval(&args) as f32;
        }

        Ok(())
    }

    /// Evaluates backward derivative pointwise operation for a given input index on CPU slices of `f32`.
    pub fn eval_backward_f32(
        &self,
        grad_out: &[f32],
        inputs: &[&[f32]],
        input_idx: usize,
        grad_in: &mut [f32],
    ) -> Result<()> {
        let len = grad_in.len();
        if grad_out.len() != len {
            return Err(Error::Msg("Gradient slice length mismatch".into()));
        }
        for inp in inputs {
            if inp.len() != len {
                return Err(Error::Msg("Input slice length mismatch".into()));
            }
        }
        let derivative = self
            .definition
            .backward_derivatives
            .get(input_idx)
            .ok_or_else(|| {
                Error::Msg(alloc::format!("Derivative for input {input_idx} not found"))
            })?;

        let mut args = alloc::vec![0.0f64; inputs.len()];
        for i in 0..len {
            for (arg_idx, inp) in inputs.iter().enumerate() {
                args[arg_idx] = inp[i] as f64;
            }
            let d = derivative.eval(&args) as f32;
            grad_in[i] = grad_out[i] * d;
        }

        Ok(())
    }
}
