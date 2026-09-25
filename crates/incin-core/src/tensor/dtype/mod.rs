//! Logical dtype descriptors, built-in dtype implementations, and storage
//! encodings.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `registry` is the mutually
//! referential identity/validation/encoding block (`DTypeKey`, `DTypeKind`,
//! `StorageEncoding`, `DTypeDescriptor`, `DTypeId`) that shares one stable
//! seam and is not split further; `traits` is the `DType` trait hierarchy
//! (`DType`, `ConstDType`, `BuiltinDType`, `TensorElement`, `PlainDType`,
//! `FloatDType`, `IntDType`, `BoolDType`, `QuantDType`); `builtin` is every
//! concrete implementation (`Q8_0`, the `impl_plain_builtin_dtype!` macro
//! and its generated impls, the manual `bool`/`Q8_0`/`Dyn` impls); `fp8` is
//! the OCP E4M3/E5M2 element types plus their conversion contract.

use crate::shapes::Dyn;
use crate::shapes::error::{OperationKind, ShapeError};

use core::{fmt::Debug, marker::PhantomData};
pub use half::{bf16, f16};

mod builtin;
mod fp8;
mod registry;
#[cfg(test)]
mod tests;
mod traits;

pub use builtin::Q8_0;
pub use fp8::{F8E4M3, F8E5M2};
pub use registry::{DTypeDescriptor, DTypeId, DTypeKey, DTypeKind, DTypeRegistry, StorageEncoding};
pub use traits::{
    BoolDType, BuiltinDType, ConstDType, DType, FloatDType, IntDType, PlainDType, QuantDType,
    TensorElement,
};
