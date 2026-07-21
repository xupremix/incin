use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use spin::Mutex;
use std::borrow::Cow;
use wgpu::{ComputePipeline, ShaderModuleDescriptor};

use crate::device::get_device_state;

/// Auto-generated documentation for PIPELINE_CACHE.
static PIPELINE_CACHE: std::sync::OnceLock<Mutex<BTreeMap<String, Arc<ComputePipeline>>>> =
    std::sync::OnceLock::new();

pub(crate) fn get_or_create_pipeline(
    name: &str,
    shader_source: &str,
    entry_point: &str,
) -> Arc<ComputePipeline> {
    let cache = PIPELINE_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut map = cache.lock();

    let key = format!("{}:{}", name, entry_point);
    if let Some(pipeline) = map.get(&key) {
        return pipeline.clone();
    }

    let state = get_device_state();
    let shader = state.device.create_shader_module(ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
    });

    let pipeline = state
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&key),
            layout: None, // Auto-derive bind group layouts from the shader
            module: &shader,
            entry_point: entry_point,
            compilation_options: Default::default(),
            cache: None,
        });

    let arc_pipeline = Arc::new(pipeline);
    map.insert(key, arc_pipeline.clone());
    arc_pipeline
}
