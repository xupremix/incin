use crate::nn::StateDict;
use crate::prelude::*;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// Loads weights into a module from a safetensors file.
pub fn load_safetensors<B, M, P>(module: &mut M, path: P) -> Result<()>
where
    B: Backend,
    M: StateDict<B>,
    P: AsRef<Path>,
{
    let buffer = std::fs::read(path)
        .map_err(|e| Error::Msg(format!("Failed to read safetensors: {}", e)))?;
    let tensors = SafeTensors::deserialize(&buffer)
        .map_err(|e| Error::Msg(format!("Safetensors deserialization failed: {:?}", e)))?;

    let mapped_tensors = HashMap::new();

    for (_name, _view) in tensors.tensors() {
        // Convert to a Candle Tensor and then to RawVar?
        // Safetensors view gives us shape and byte data.
        // For simplicity, since we are Backend agnostic here, we need the Backend to implement `var_from_safetensors` or similar.
        // In a complete implementation, we'd iterate over view and construct a Tensor<Dyn, B>.
        // TODO: Implement actual data transfer to B::RawVar.
    }

    // For now, this is just a dummy load.
    module.load_state_dict("", &mapped_tensors)?;

    Ok(())
}

/// Saves the module's weights to a safetensors file.
pub fn save_safetensors<B, M, P>(module: &M, _path: P) -> Result<()>
where
    B: Backend,
    M: StateDict<B>,
    P: AsRef<Path>,
{
    let mut mapped_tensors = HashMap::new();
    module.state_dict("", &mut mapped_tensors);

    // TODO: Serialize mapped_tensors to safetensors.
    // We would need to extract the raw bytes from B::RawTensor (e.g. by converting to Vec<f32>)
    // and passing them to safetensors::serialize().

    Ok(())
}
