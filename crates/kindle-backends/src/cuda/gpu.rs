#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use alloc::sync::Arc;
    use cudarc::driver::CudaContext;
    use kindle_core::prelude::Result;

    /// Cpu CUDA kernel compiler and execution dispatcher.
    pub(crate) struct CpuCudaDispatcher {
        pub(crate) device_id: usize,
        pub(crate) ctx: Arc<CudaContext>,
    }

    impl CpuCudaDispatcher {
        /// Core abstraction for `new` within the Kindle framework..
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
            let ptx = cudarc::nvrtc::compile_ptx(src).map_err(|e| {
                kindle_core::prelude::Error::Msg(format!("PTX compile failed: {:?}", e))
            })?;
            let module = self.ctx.load_module(ptx).map_err(|e| {
                kindle_core::prelude::Error::Msg(format!("Load PTX failed: {:?}", e))
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
                kindle_core::prelude::Error::Msg(format!("Module {} not found", module_name))
            })?;
            let f = module.load_function(entry_point).map_err(|e| {
                kindle_core::prelude::Error::Msg(format!(
                    "Function {} not found: {:?}",
                    entry_point, e
                ))
            })?;
            Ok(f)
        }
    }

    pub(crate) mod cuda_cache {
        use alloc::collections::BTreeMap;
        use cudarc::driver::{CudaContext, CudaModule};
        use std::sync::{Arc, Mutex, OnceLock};

        /// Core abstraction for `CUDA_DEVICES` within the Kindle framework..
        static CUDA_DEVICES: OnceLock<Mutex<BTreeMap<usize, Arc<CudaContext>>>> = OnceLock::new();
        /// Core abstraction for `CUDA_MODULES` within the Kindle framework..
        static CUDA_MODULES: OnceLock<Mutex<BTreeMap<(usize, String), Arc<CudaModule>>>> =
            OnceLock::new();

        /// Core abstraction for `get_cuda_device` within the Kindle framework..
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

        /// Core abstraction for `cache_module` within the Kindle framework..
        pub fn cache_module(device_id: usize, module_name: String, module: Arc<CudaModule>) {
            let map_mutex = CUDA_MODULES.get_or_init(|| Mutex::new(BTreeMap::new()));
            let mut map = map_mutex.lock().unwrap();
            map.insert((device_id, module_name), module);
        }

        /// Core abstraction for `get_module` within the Kindle framework..
        pub fn get_module(device_id: usize, module_name: &str) -> Option<Arc<CudaModule>> {
            let map_mutex = CUDA_MODULES.get_or_init(|| Mutex::new(BTreeMap::new()));
            let map = map_mutex.lock().unwrap();
            map.get(&(device_id, module_name.to_string())).cloned()
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) use cuda::*;
