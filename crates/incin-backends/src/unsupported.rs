//! Explicit declarations for operations a backend does not implement.
//!
//! Legacy operation-family traits used to give every method a default body returning
//! [`Error::UnsupportedBackendOperation`], so a backend covering sixteen of its
//! forty-two operations looked exactly like one covering all forty-two. The
//! refusal was real but invisible: it lived in the trait, not in the backend
//! that was actually refusing, and nothing failed when a backend's coverage
//! silently shrank.
//!
//! `EXE-009` removes those defaults, which makes the omission a compile error.
//! This macro is how a backend answers it: the gap is written down where the
//! backend is defined, greppable and reviewable, and the compiler adds any
//! newly-declared operation to the list a backend must consciously answer for.
//!
//! The error a caller sees is unchanged. What changes is that a reader can see
//! which operations a backend does not have without running it.
//!
//! [`Error::UnsupportedBackendOperation`]: incin_core::error::Error

#![allow(unused_macros, unused_imports, dead_code)]

/// Declares float operations a backend does not implement, grouped by
/// signature: `unary` takes one tensor, `exponent` a tensor and an exponent,
/// `bounds` a tensor and a min/max pair, and `binary` two tensors.
macro_rules! unsupported_float_ops {
    (
        unary: $($unary:ident),* $(,)?;
        exponent: $($exponent:ident),* $(,)?;
        bounds: $($bounds:ident),* $(,)?;
        binary: $($binary:ident),* $(,)?;
    ) => {
        $(
            /// Refuses a unary op with the registry's unsupported error.
            pub fn $unary<K: DType>(
                _t: &<Self as StorageBackend>::Storage<K>,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($unary)))
            }
        )*
        $(
            /// Refuses an exponent-style op with the registry's unsupported error.
            pub fn $exponent<K: DType>(
                _t: &<Self as StorageBackend>::Storage<K>,
                _exponent: f64,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($exponent)))
            }
        )*
        $(
            /// Refuses a bounds-checking op with the registry's unsupported error.
            pub fn $bounds<K: DType>(
                _t: &<Self as StorageBackend>::Storage<K>,
                _min: f64,
                _max: f64,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($bounds)))
            }
        )*
        $(
            /// Refuses a binary op with the registry's unsupported error.
            pub fn $binary<K: DType>(
                _lhs: &<Self as StorageBackend>::Storage<K>,
                _rhs: &<Self as StorageBackend>::Storage<K>,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err($crate::unsupported::unsupported::<Self>(stringify!($binary)))
            }
        )*
    };
}

pub(crate) use unsupported_float_ops;

/// The error a declared gap reports, identical to the removed default body's.
pub(crate) fn unsupported<B>(op: &'static str) -> incin_core::error::Error {
    incin_core::error::Error::UnsupportedBackendOperation {
        op,
        backend: core::any::type_name::<B>(),
    }
}
