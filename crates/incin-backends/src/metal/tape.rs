//! Metal autograd tape and gradient container.

use incin_core::exec::GradientMap;

use crate::metal::storage::MetalStorage;

/// The Metal backend's gradient container.
///
/// Intentionally empty for now — Metal does not support autograd in this
/// implementation. The type exists to satisfy the `Backend::Grads` associated
/// type bound. A real Metal autograd pass would store gradient tensors here.
pub struct MetalGrads {
    pub(crate) grads: GradientMap<MetalStorage>,
}

impl MetalGrads {
    /// Create an empty gradient map.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            grads: GradientMap::default(),
        }
    }
}
