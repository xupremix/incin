#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use alloc::sync::Arc;
    use cudarc::driver::CudaContext;
    use incin_core::error::Result;

    /// Cpu CUDA kernel compiler and execution dispatcher.
    pub(crate) struct CpuCudaDispatcher {
        pub(crate) device_id: usize,
        pub(crate) ctx: Arc<CudaContext>,
    }

    impl CpuCudaDispatcher {
        /// Creates a new instance with default (statically inferred) shape arguments.
        pub fn new(device_id: usize) -> Result<Self> {
            let ctx = cuda_cache::try_get_cuda_device(device_id).map_err(|error| {
                incin_core::error::Error::Backend(incin_core::error::BackendError::Execution {
                    operation: incin_core::shapes::error::OperationKind::Storage,
                    message: format!("CUDA context initialization failed: {error:?}").into(),
                })
            })?;
            Ok(Self { device_id, ctx })
        }

        /// Compile a CUDA C/C++ kernel source string and load it into the device context.
        pub fn compile_and_load_kernel(
            &self,
            _name: &str,
            src: &str,
            module_name: &str,
        ) -> Result<()> {
            let ptx = compile_ptx_with_cuda_includes(src).map_err(|e| {
                incin_core::error::Error::Msg(format!("PTX compile failed: {:?}", e))
            })?;
            let module = self
                .ctx
                .load_module(ptx)
                .map_err(|e| incin_core::error::Error::Msg(format!("Load PTX failed: {:?}", e)))?;
            cuda_cache::cache_module(self.device_id, module_name.to_string(), module);
            Ok(())
        }

        /// Retrieve a loaded function from the context.
        pub fn get_function(
            &self,
            module_name: &str,
            entry_point: &str,
        ) -> Result<cudarc::driver::CudaFunction> {
            let module = cuda_cache::get_module(self.device_id, module_name).ok_or_else(|| {
                incin_core::error::Error::Msg(format!("Module {} not found", module_name))
            })?;
            let f = module.load_function(entry_point).map_err(|e| {
                incin_core::error::Error::Msg(format!(
                    "Function {} not found: {:?}",
                    entry_point, e
                ))
            })?;
            Ok(f)
        }
    }

    /// Locates the CUDA toolkit's `include` directory so NVRTC can resolve
    /// `#include <cuda_fp16.h>` / `<cuda_bf16.h>` - NVRTC has no default
    /// header search path, unlike a host C++ compiler, so this must be
    /// passed explicitly via `--include-path`.
    pub(crate) fn cuda_include_paths() -> alloc::vec::Vec<String> {
        for var in ["CUDA_INCLUDE_PATH", "CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
            if let Ok(value) = std::env::var(var) {
                let candidate = if var == "CUDA_INCLUDE_PATH" {
                    value
                } else {
                    alloc::format!("{value}/include")
                };
                if std::path::Path::new(&candidate).is_dir() {
                    return alloc::vec![candidate];
                }
            }
        }
        for candidate in [
            "/usr/local/cuda/include",
            "/usr/local/cuda/targets/x86_64-linux/include",
        ] {
            if std::path::Path::new(candidate).is_dir() {
                return alloc::vec![candidate.to_string()];
            }
        }
        alloc::vec::Vec::new()
    }

    /// Compiles a CUDA C/C++ source string to PTX, wiring up `--include-path`
    /// so kernels that need `cuda_fp16.h`/`cuda_bf16.h` for half/bfloat16
    /// support can find them.
    pub(crate) fn compile_ptx_with_cuda_includes(
        src: &str,
    ) -> core::result::Result<cudarc::nvrtc::Ptx, cudarc::nvrtc::CompileError> {
        cudarc::nvrtc::compile_ptx_with_opts(
            src,
            cudarc::nvrtc::CompileOptions {
                include_paths: cuda_include_paths(),
                ..Default::default()
            },
        )
    }

    pub(crate) mod cuda_cache {
        use alloc::collections::BTreeMap;
        use cudarc::driver::{CudaContext, CudaModule, DriverError};
        use std::sync::{Arc, Mutex, OnceLock};

        type CudaModuleMap = BTreeMap<(usize, String), Arc<CudaModule>>;

        /// `CUDA_DEVICES`.
        static CUDA_DEVICES: OnceLock<Mutex<BTreeMap<usize, Arc<CudaContext>>>> = OnceLock::new();
        /// `CUDA_MODULES`.
        static CUDA_MODULES: OnceLock<Mutex<CudaModuleMap>> = OnceLock::new();

        /// The process-wide context for `id`, created once and never released.
        ///
        /// `CudaContext::new` retains the device's *primary* context, so every
        /// call returns the same underlying context and allocations made through
        /// any of them are mutually valid. What is not free is the retain/release
        /// cycle at the edges: when the last `Arc` for a device drops, the
        /// primary context is released, and the next retain pays full
        /// re-initialization. Measured on a GTX 1650 SUPER, that is 131 ms with
        /// no context held against 1 us with one held - five orders of
        /// magnitude, decided entirely by whether anything kept a handle.
        ///
        /// Holding one `Arc` per device here for the lifetime of the process is
        /// what keeps every later call on the 1 us path. The map is never
        /// evicted, and that is the point rather than an oversight: releasing the
        /// last handle is precisely the expensive event.
        pub fn try_get_cuda_device(id: usize) -> Result<Arc<CudaContext>, DriverError> {
            let map_mutex = CUDA_DEVICES.get_or_init(|| Mutex::new(BTreeMap::new()));
            let mut map = match map_mutex.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            if let Some(dev) = map.get(&id) {
                return Ok(dev.clone());
            }
            let res = std::panic::catch_unwind(|| CudaContext::new(id));
            match res {
                Ok(Ok(dev)) => {
                    map.insert(id, dev.clone());
                    Ok(dev)
                }
                Ok(Err(err)) => Err(err),
                Err(_) => Err(DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
                )),
            }
        }

        /// `cache_module`.
        pub fn cache_module(device_id: usize, module_name: String, module: Arc<CudaModule>) {
            let map_mutex = CUDA_MODULES.get_or_init(|| Mutex::new(BTreeMap::new()));
            let mut map = map_mutex
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert((device_id, module_name), module);
        }

        /// `get_module`.
        pub fn get_module(device_id: usize, module_name: &str) -> Option<Arc<CudaModule>> {
            let map_mutex = CUDA_MODULES.get_or_init(|| Mutex::new(BTreeMap::new()));
            let map = map_mutex
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            map.get(&(device_id, module_name.to_string())).cloned()
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) use cuda::*;
