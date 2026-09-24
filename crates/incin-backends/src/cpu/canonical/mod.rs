//! Canonical descriptor execution for the CPU backend.
//!
//! One `Execute<op::X>` implementation per exact catalog identity,
//! generated from the same `cpu_descriptor_operations!` declaration that
//! generates `CPU_CAPABILITIES`. Advertising an operation and implementing it
//! are therefore the same edit, and a row that claims support the executor does
//! not provide will not compile.

pub(crate) mod common;
pub(crate) mod creation;
pub(crate) mod elementwise;
pub(crate) mod linalg;
pub(crate) mod nn;
pub(crate) mod reduce;
pub(crate) mod shape_ops;

#[cfg(test)]
mod tests;

use crate::cpu::CpuBackendImpl;
use incin_core::backend_authoring::Execute;
use incin_core::exec::catalog::op;
use incin_core::tensor::device::Device;

/// Prove, at compile time, that every identity `CPU_CAPABILITIES` advertises
/// has an executor above.
///
/// Each group entry is either a bare `Op` or an `(Op, training)` pair - the
/// quantization groups spell their per-operation training flag that way (see
/// `capability::rules`'s `descriptor_capability_rules!`). The flag is a
/// capability claim, not an execution requirement, so both spellings carry the
/// same obligation here: `Execute<op::Op>` must exist.
macro_rules! assert_every_advertised_row_executes {
    (; $($group:ident = [$($entry:tt),* $(,)?]),* $(,)?) => {
        const _: () = {
            const fn executes<O, B>()
            where
                O: incin_core::exec::CanonicalOperation,
                B: Execute<O>,
            {
            }

            const fn assert_all<D: Device>() {
                macro_rules! assert_entry {
                    (($operation:ident, $training:expr)) => {
                        executes::<op::$operation, CpuBackendImpl<D>>()
                    };
                    ($operation:ident) => {
                        executes::<op::$operation, CpuBackendImpl<D>>()
                    };
                }
                $( $(assert_entry!($entry);)*)*
            }

            assert_all::<incin_core::tensor::device::Cpu>();
        };
    };
}

crate::capability::cpu_descriptor_operations!(assert_every_advertised_row_executes,);
