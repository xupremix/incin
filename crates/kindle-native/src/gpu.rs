use crate::storage::NativeMetalBuffer;
use kindle_core::prelude::Result;

use std::sync::Arc;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;

/// Native CUDA kernel compiler and execution dispatcher.
pub struct NativeCudaDispatcher {
    pub device_id: usize,
    #[cfg(feature = "cuda")]
    pub ctx: Arc<CudaContext>,
}

impl NativeCudaDispatcher {
    pub fn new(device_id: usize) -> Self {
        #[cfg(feature = "cuda")]
        let ctx = cuda_cache::get_cuda_device(device_id);

        Self {
            device_id,
            #[cfg(feature = "cuda")]
            ctx,
        }
    }

    /// Compile a CUDA C/C++ kernel source string and load it into the device context.
    pub fn compile_and_load_kernel(&self, _name: &str, src: &str, module_name: &str) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let ptx = cudarc::nvrtc::compile_ptx(src).map_err(|e| kindle_core::err::Error::Msg(format!("PTX compile failed: {:?}", e)))?;
            let module = self.ctx.load_module(ptx).map_err(|e| kindle_core::err::Error::Msg(format!("Load PTX failed: {:?}", e)))?;
            cuda_cache::cache_module(self.device_id, module_name.to_string(), module);
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        Err(kindle_core::err::Error::UnsupportedBackendOperation { op: "compile_kernel", backend: "CUDA (not compiled)" })
    }

    /// Retrieve a loaded function from the context.
    #[cfg(feature = "cuda")]
    pub fn get_function(&self, module_name: &str, entry_point: &str) -> Result<cudarc::driver::CudaFunction> {
        let module = cuda_cache::get_module(self.device_id, module_name)
            .ok_or_else(|| kindle_core::err::Error::Msg(format!("Module {} not found", module_name)))?;
        let f = module.load_function(entry_point).map_err(|e| kindle_core::err::Error::Msg(format!("Function {} not found: {:?}", entry_point, e)))?;
        Ok(f)
    }
}

/// Native Apple Metal shading compiler and execution dispatcher.
pub struct NativeMetalDispatcher {
    pub device_id: usize,
}

impl NativeMetalDispatcher {
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    /// Compile a Metal Shading Language (MSL) source string into library bytecode.
    pub fn compile_kernel(&self, _src: &str) -> Result<Vec<u8>> {
        // Placeholder for Metal Library creation using compile-time or runtime metal-command-line compiler.
        Ok(Vec::new())
    }

    /// Launch a compiled Metal kernel pipeline.
    pub fn launch_kernel(
        &self,
        _library_data: &[u8],
        _entry_point: &str,
        _threadgroups_per_grid: (u32, u32, u32),
        _threads_per_threadgroup: (u32, u32, u32),
        _args: &[&mut NativeMetalBuffer],
    ) -> Result<()> {
        // Placeholder for MTLCommandQueue, MTLComputeCommandEncoder, and pipeline state dispatch.
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_cache {
    use cudarc::driver::{CudaContext, CudaModule};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    static CUDA_DEVICES: OnceLock<Mutex<HashMap<usize, Arc<CudaContext>>>> = OnceLock::new();
    static CUDA_MODULES: OnceLock<Mutex<HashMap<(usize, String), Arc<CudaModule>>>> = OnceLock::new();

    pub fn get_cuda_device(id: usize) -> Arc<CudaContext> {
        let map_mutex = CUDA_DEVICES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = map_mutex.lock().unwrap();
        if let Some(dev) = map.get(&id) {
            return dev.clone();
        }
        let dev = CudaContext::new(id).expect("Failed to initialize CUDA context");
        map.insert(id, dev.clone());
        dev
    }

    pub fn cache_module(device_id: usize, module_name: String, module: Arc<CudaModule>) {
        let map_mutex = CUDA_MODULES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = map_mutex.lock().unwrap();
        map.insert((device_id, module_name), module);
    }

    pub fn get_module(device_id: usize, module_name: &str) -> Option<Arc<CudaModule>> {
        let map_mutex = CUDA_MODULES.get_or_init(|| Mutex::new(HashMap::new()));
        let map = map_mutex.lock().unwrap();
        map.get(&(device_id, module_name.to_string())).cloned()
    }
}
