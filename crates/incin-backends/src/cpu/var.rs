//! `CpuVar`: the sole `Rc<RefCell<_>>` mutation boundary in the whole
//! crate (CPUBACK-09).
//!
//! Everywhere else, `CpuStorage` flows as an immutable, cheaply-cloned
//! value. Only `CpuVar` allows mutation, exclusively through
//! `assign_var`. `var_as_tensor` is a scoped `borrow().clone()` that never
//! returns a live `Ref` guard — per Pitfall 5's explicit warning, holding a
//! borrow across an `assign_var` call would panic on reentrant mutable
//! borrow. `SGD::step()` (existing, unmodified, generic over `Backend`)
//! drives this exact `var_as_tensor` (read) -> ... -> `assign_var` (write)
//! sequence once per parameter per optimizer step.

use alloc::rc::Rc;
use core::cell::RefCell;

#[cfg(feature = "test-utils")]
use core::cell::Cell;

use incin_core::error::Result;

#[cfg(feature = "test-utils")]
use incin_core::error::{BackendError, Error};
#[cfg(feature = "test-utils")]
use incin_core::shapes::error::OperationKind;

use crate::cpu::storage::CpuStorage;

/// A trainable-parameter slot: the one deliberate interior-mutability
/// boundary in `incin-cpu`. Wraps `CpuStorage` in `Rc<RefCell<_>>`
/// so `assign_var` can replace the value in place while other clones of
/// the same `CpuVar` observe the update.
#[derive(Debug, Clone)]
pub struct CpuVar(pub(crate) Rc<RefCell<CpuStorage>>);

impl incin_core::backend_authoring::ExecuteOutput for CpuVar {}

/// Read the current value of `var` as a plain `CpuStorage`.
///
/// Implemented as `var.0.borrow().clone()` — a scoped borrow that clones
/// the value out and releases the borrow immediately upon returning. This
/// function must NEVER return a live `Ref<'_, CpuStorage>` guard: doing
/// so would keep the `RefCell` borrowed past the call, and a subsequent
/// `assign_var` on the same `CpuVar` would panic on reentrant mutable
/// borrow (Pitfall 5).
pub(crate) fn var_as_tensor(var: &CpuVar) -> Result<CpuStorage> {
    Ok(var.0.borrow().clone())
}

/// Replace the value wrapped by `var` with a clone of `tensor`.
///
/// This is the ONLY function in the whole `incin-cpu` crate permitted
/// to call `borrow_mut()` on a `CpuVar`'s inner `RefCell` (CPUBACK-09,
/// structurally enforced — grep the crate for `borrow_mut()` to confirm
/// this is the sole call site).
pub(crate) fn assign_var(var: &mut CpuVar, tensor: &CpuStorage) -> Result<()> {
    #[cfg(feature = "test-utils")]
    if should_fail_assign() {
        return Err(Error::Backend(BackendError::Execution {
            operation: OperationKind::Pointwise,
            message: "injected CPU variable-assignment failure".into(),
        }));
    }
    *var.0.borrow_mut() = tensor.clone();
    Ok(())
}

#[cfg(feature = "test-utils")]
std::thread_local! {
    static FAIL_ASSIGN_AT: Cell<Option<usize>> = const { Cell::new(None) };
    static ASSIGN_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[cfg(feature = "test-utils")]
fn should_fail_assign() -> bool {
    let count = ASSIGN_COUNT.with(|count| {
        let next = count.get() + 1;
        count.set(next);
        next
    });
    FAIL_ASSIGN_AT.with(|fail_at| {
        if fail_at.get() == Some(count) {
            // One-shot failure: rollback assignments must remain possible.
            fail_at.set(None);
            true
        } else {
            false
        }
    })
}

/// Scope guard for deterministic CPU assignment fault injection.
///
/// Available only with `test-utils`; dropping it always disables the hook.
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub struct AssignFailureGuard;

#[cfg(feature = "test-utils")]
impl Drop for AssignFailureGuard {
    fn drop(&mut self) {
        FAIL_ASSIGN_AT.with(|fail_at| fail_at.set(None));
        ASSIGN_COUNT.with(|count| count.set(0));
    }
}

/// Injects a one-shot failure at the `nth` CPU variable assignment on this
/// thread. Test-only: product builds do not compile this hook.
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub fn fail_assign_on(nth: usize) -> AssignFailureGuard {
    assert!(nth > 0, "assignment failure index is one-based");
    ASSIGN_COUNT.with(|count| count.set(0));
    FAIL_ASSIGN_AT.with(|fail_at| fail_at.set(Some(nth)));
    AssignFailureGuard
}

/// Construct a fresh `CpuVar` wrapping a clone of `t`.
pub(crate) fn var_from_tensor(t: &CpuStorage) -> Result<CpuVar> {
    Ok(CpuVar(Rc::new(RefCell::new(t.clone()))))
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;
    use crate::cpu::storage::CpuBuffer;

    /// `storage`.
    fn storage(v: Vec<f32>) -> CpuStorage {
        let len = v.len();
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![len])
    }

    /// `f32_vec`.
    fn f32_vec(s: &CpuStorage) -> Vec<f32> {
        match &*s.buffer {
            CpuBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    #[test]
    /// `var_as_tensor_returns_clone_of_current_value`.
    fn var_as_tensor_returns_clone_of_current_value() {
        let t = storage(vec![1.0, 2.0, 3.0]);
        let var = var_from_tensor(&t).unwrap();
        let read_back = var_as_tensor(&var).unwrap();
        assert_eq!(f32_vec(&read_back), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    /// `assign_var_replaces_value_and_subsequent_read_reflects_it`.
    fn assign_var_replaces_value_and_subsequent_read_reflects_it() {
        let t = storage(vec![1.0, 2.0, 3.0]);
        let mut var = var_from_tensor(&t).unwrap();

        let new_value = storage(vec![9.0, 9.0, 9.0]);
        assign_var(&mut var, &new_value).unwrap();

        let read_back = var_as_tensor(&var).unwrap();
        assert_eq!(f32_vec(&read_back), vec![9.0, 9.0, 9.0]);
    }

    #[test]
    /// `var_as_tensor_does_not_hold_live_borrow_across_assign_var`.
    fn var_as_tensor_does_not_hold_live_borrow_across_assign_var() {
        // Calling var_as_tensor immediately followed by assign_var on the
        // same CpuVar within the same scope must not panic — proves
        // var_as_tensor's borrow() ends immediately after the clone,
        // rather than returning a live Ref guard (Pitfall 5).
        let t = storage(vec![1.0]);
        let mut var = var_from_tensor(&t).unwrap();

        let _read = var_as_tensor(&var).unwrap();
        // If var_as_tensor held a live Ref guard past its return, this
        // borrow_mut() would panic here.
        assign_var(&mut var, &storage(vec![2.0])).unwrap();

        assert_eq!(f32_vec(&var_as_tensor(&var).unwrap()), vec![2.0]);
    }

    #[test]
    /// `two_sequential_assign_var_calls_each_succeed_and_final_read_reflects_second`.
    fn two_sequential_assign_var_calls_each_succeed_and_final_read_reflects_second() {
        let t = storage(vec![0.0]);
        let mut var = var_from_tensor(&t).unwrap();

        assign_var(&mut var, &storage(vec![1.0])).unwrap();
        assign_var(&mut var, &storage(vec![2.0])).unwrap();

        assert_eq!(f32_vec(&var_as_tensor(&var).unwrap()), vec![2.0]);
    }
}
