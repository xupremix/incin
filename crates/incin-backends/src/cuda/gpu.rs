#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use alloc::sync::Arc;
    use cudarc::driver::CudaContext;
    use incin_core::prelude::Result;

    /// Cpu CUDA kernel compiler and execution dispatcher.
    pub(crate) struct CpuCudaDispatcher {
        pub(crate) device_id: usize,
        pub(crate) ctx: Arc<CudaContext>,
    }

    impl CpuCudaDispatcher {
        /// Creates a new instance with default (statically inferred) shape arguments.
        pub fn new(device_id: usize) -> Self {
            let ctx = cuda_cache::get_cuda_device(device_id);
            Self { device_id, ctx }
        }

        /// Compile a CUDA C/C++ kernel source string and load it into the device context.
        pub fn compile_and_load_kernel(
            &self,
            _name: &str,
            src: &str,
            module_name: &str,
        ) -> Result<()> {
            let ptx = compile_ptx_with_cuda_includes(src).map_err(|e| {
                incin_core::prelude::Error::Msg(format!("PTX compile failed: {:?}", e))
            })?;
            let module = self.ctx.load_module(ptx).map_err(|e| {
                incin_core::prelude::Error::Msg(format!("Load PTX failed: {:?}", e))
            })?;
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
                incin_core::prelude::Error::Msg(format!("Module {} not found", module_name))
            })?;
            let f = module.load_function(entry_point).map_err(|e| {
                incin_core::prelude::Error::Msg(format!(
                    "Function {} not found: {:?}",
                    entry_point, e
                ))
            })?;
            Ok(f)
        }
    }

    /// Locates the CUDA toolkit's `include` directory so NVRTC can resolve
    /// `#include <cuda_fp16.h>` / `<cuda_bf16.h>` — NVRTC has no default
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
        use cudarc::driver::{CudaContext, CudaModule};
        use std::sync::{Arc, Mutex, OnceLock};

        /// `CUDA_DEVICES`.
        static CUDA_DEVICES: OnceLock<Mutex<BTreeMap<usize, Arc<CudaContext>>>> = OnceLock::new();
        /// `CUDA_MODULES`.
        static CUDA_MODULES: OnceLock<Mutex<BTreeMap<(usize, String), Arc<CudaModule>>>> =
            OnceLock::new();

        /// `get_cuda_device`.
        pub fn get_cuda_device(id: usize) -> Arc<CudaContext> {
            let map_mutex = CUDA_DEVICES.get_or_init(|| Mutex::new(BTreeMap::new()));
            let mut map = map_mutex.lock().unwrap();
            if let Some(dev) = map.get(&id) {
                return dev.clone();
            }
            let dev = CudaContext::new(id).expect("Failed to initialize CUDA context");
            map.insert(id, dev.clone());
            dev
        }

        /// `cache_module`.
        pub fn cache_module(device_id: usize, module_name: String, module: Arc<CudaModule>) {
            let map_mutex = CUDA_MODULES.get_or_init(|| Mutex::new(BTreeMap::new()));
            let mut map = map_mutex.lock().unwrap();
            map.insert((device_id, module_name), module);
        }

        /// `get_module`.
        pub fn get_module(device_id: usize, module_name: &str) -> Option<Arc<CudaModule>> {
            let map_mutex = CUDA_MODULES.get_or_init(|| Mutex::new(BTreeMap::new()));
            let map = map_mutex.lock().unwrap();
            map.get(&(device_id, module_name.to_string())).cloned()
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) use cuda::*;
