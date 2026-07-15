use alloc::sync::Arc;
use alloc::collections::BTreeMap;
use spin::Mutex;
use wgpu::{ComputePipeline, ShaderModuleDescriptor, ComputePipelineDescriptor, PipelineLayoutDescriptor};
use std::borrow::Cow;

use crate::device::get_device_state;

static PIPELINE_CACHE: std::sync::OnceLock<Mutex<BTreeMap<String, Arc<ComputePipeline>>>> = std::sync::OnceLock::new();

pub(crate) fn get_or_create_pipeline(name: &str, shader_source: &str, entry_point: &str) -> Arc<ComputePipeline> {
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

    let pipeline_layout = state.device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(&format!("{}_layout", name)),
        bind_group_layouts: &[], // Will be auto-inferred or explicitly defined if needed. Wait, wgpu compute pipelines typically need bind group layouts if we use create_compute_pipeline.
        push_constant_ranges: &[],
    });

    // Actually, wgpu provides `create_compute_pipeline` but it requires bind group layouts.
    // If we want implicit layouts, we can use `create_compute_pipeline` without specifying layout if we don't pass it? Wait, wgpu requires explicit pipeline layout.
    // However, `device.create_compute_pipeline` requires a layout. Wait, we can pass `None` for layout to auto-derive it!
    // Let me check wgpu docs for auto layout. Yes, `layout: None` derives it from the shader.
    
    let pipeline = state.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
