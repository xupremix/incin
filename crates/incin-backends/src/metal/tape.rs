//! Metal autograd tape and gradient container.

use core::cell::RefCell;

use incin_core::exec::tape;
use incin_core::exec::{GradientMap, Tape, TapeNode};
use incin_core::prelude::Result;

use crate::metal::storage::{MetalStorage, TensorId};

/// One recorded operation on Metal storage, as defined by core.
pub(crate) type TapeEntry = TapeNode<MetalStorage>;

thread_local! {
    static TAPE: RefCell<Tape<MetalStorage>> = const { RefCell::new(Tape::new()) };
}

/// Number of entries currently on the tape.
#[must_use]
pub fn depth() -> usize {
    TAPE.with(|t| t.borrow().depth())
}

/// Push a `TapeEntry` unless `GradMode` forbids recording.
pub fn push(entry: TapeEntry) {
    TAPE.with(|t| t.borrow_mut().push(entry));
}

/// The Metal backend's gradient container.
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

    /// Look up the accumulated gradient for a given tensor id, if any.
    #[must_use]
    pub fn get(&self, id: TensorId) -> Option<&MetalStorage> {
        self.grads.get(id)
    }

    /// How many tensors the backward pass reached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Whether it reached none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }
}

/// Perform a reverse-mode autograd walk starting at `loss`.
///
/// # Errors
///
/// Returns [`incin_core::prelude::Error`] if tape traversal or gradient accumulation fails.
pub fn backward(loss: &MetalStorage) -> Result<MetalGrads> {
    let nodes = TAPE.with(|t| t.borrow_mut().drain());
    let grads = tape::backward(nodes, loss)?;
    Ok(MetalGrads { grads })
}
