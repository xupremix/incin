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
macro_rules! assert_every_advertised_row_executes {
    (; $($group:ident = [$($operation:ident),* $(,)?]),* $(,)?) => {
        const _: () = {
            const fn executes<O, B>()
            where
                O: incin_core::exec::CanonicalOperation,
                B: Execute<O>,
            {
            }

            const fn assert_all<D: Device>() {
                $($(executes::<op::$operation, CpuBackendImpl<D>>();)*)*
            }

            assert_all::<incin_core::tensor::device::Cpu>();
        };
    };
}

crate::capability::cpu_descriptor_operations!(assert_every_advertised_row_executes,);
