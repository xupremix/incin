//! `WgpuBackendImpl`, `WgpuVar`, and the tape-gradient-set alias every
//! other module in this split attaches its impls to.

use super::*;

/// WebGPU compute backend implementation for Incin.
#[derive(Clone)]
pub struct WgpuBackendImpl<D = Wgpu>(core::marker::PhantomData<D>);

impl<D> WgpuBackendImpl<D> {
    /// Construct the stateless WGPU executor.
    #[must_use]
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<D> Default for WgpuBackendImpl<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// A trainable-parameter slot: the one deliberate interior-mutability
/// boundary in the WGPU backend, mirroring `CpuVar`.
///
/// The shared handle is load bearing, not a style choice. An optimizer holds
/// its own map of these and commits an update through
/// [`VariableBackend::assign_var`]; the model holds the *same* parameters. With a
/// plain owned `WgpuStorage` here, `assign_var` replaced the optimizer's copy
/// and the model never saw it, so `optimizer.step()` was a no-op and training
/// loss sat at exactly its initial value forever - the failure looks like a
/// bad learning rate rather than a broken write, which is why it survived.
#[derive(Clone)]
pub struct WgpuVar {
    /// Boxed behind `Rc<RefCell<_>>` so every clone of this parameter slot
    /// observes an assignment. Private: handing out the cell would let a
    /// caller hold a live borrow across an `assign_var` and panic on the
    /// reentrant mutable borrow, which is the same hazard `cpu::var`
    /// documents.
    storage: alloc::rc::Rc<core::cell::RefCell<WgpuStorage>>,
}

impl WgpuVar {
    /// Wrap `storage` in a fresh parameter slot.
    #[must_use]
    pub(crate) fn new(storage: WgpuStorage) -> Self {
        Self {
            storage: alloc::rc::Rc::new(core::cell::RefCell::new(storage)),
        }
    }

    /// The current value, cloned out.
    ///
    /// Never returns the `Ref` guard: holding one across a later
    /// `assign_var` on the same slot would panic on a reentrant mutable
    /// borrow.
    #[must_use]
    pub(crate) fn value(&self) -> WgpuStorage {
        self.storage.borrow().clone()
    }

    /// Replace the wrapped value, visible through every clone of this slot.
    pub(crate) fn assign(&self, storage: WgpuStorage) {
        *self.storage.borrow_mut() = storage;
    }
}

/// Alias binding the WGPU tape's gradient map into the backend surface.
pub type WgpuGrads = crate::wgpu::tape::WgpuGrads;
