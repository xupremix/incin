use super::*;
use crate::tensor::dtype::traits::sealed;

// ============================================================================
// Q8_0 logical dtype marker
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Q8_0 block quantization: groups of 32 elements share one `f16` scale,
/// each element stored as a scaled `i8`.
///
/// **`Q8_0` is a logical dtype marker only.**
///
/// It is NOT one physical scalar element per logical value. Physical data is
/// stored as `BlockQ8_0` blocks in the backend-specific storage. The
/// backend-specific storage. The block layout is:
///
/// - 32 logical `i8` quantized values per block
/// - 1 `f16` scale per block
/// - Total: 34 bytes per block, 2-byte aligned
///
/// Because `Q8_0` has no plain scalar representation, it does NOT implement:
/// - [`TensorElement`]
/// - [`bytemuck::Pod`]
/// - [`bytemuck::Zeroable`]
/// - [`PlainDType`]
///
/// To construct a Q8_0 tensor, use a quantization-specific constructor, not
/// `from_slice`.
pub struct Q8_0;

// ============================================================================
// Macro: impl_plain_builtin_dtype
// ============================================================================

macro_rules! impl_plain_builtin_dtype {
    ($repr:ident, $t:ty, $kind:expr, $encoding:expr, $name:expr) => {
        impl sealed::TensorElementSealed for $t {}

        impl DType for $t {
            /// No argument needed — the dtype is fixed by the Rust type itself.
            type Arg = ();
            /// Zero-sized: the value is fixed by the type.
            type Field = PhantomData<$t>;

            /// No-op: nothing to convert.
            fn init(_: Self::Arg) -> Self::Field {
                PhantomData
            }

            fn descriptor(_: &Self::Field) -> DTypeDescriptor {
                Self::DESCRIPTOR
            }
        }

        impl ConstDType for $t {
            /// The compile-time-known full descriptor.
            const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::builtin(
                DTypeId::$repr,
                DTypeKey::new("incin", $name, 1),
                $kind,
                $encoding,
            );
        }

        impl BuiltinDType for $t {
            /// The compile-time-known `DTypeId`.
            const DTYPE: DTypeId = DTypeId::$repr;
        }

        impl PlainDType for $t {
            /// This Rust type itself.
            type Elem = $t;
        }
    };
}

impl_plain_builtin_dtype!(
    F32,
    f32,
    DTypeKind::Float,
    StorageEncoding::scalar(4, 4),
    "f32"
);
impl_plain_builtin_dtype!(
    F64,
    f64,
    DTypeKind::Float,
    StorageEncoding::scalar(8, 8),
    "f64"
);
impl_plain_builtin_dtype!(
    U8,
    u8,
    DTypeKind::UnsignedInteger,
    StorageEncoding::scalar(1, 1),
    "u8"
);
impl_plain_builtin_dtype!(
    U32,
    u32,
    DTypeKind::UnsignedInteger,
    StorageEncoding::scalar(4, 4),
    "u32"
);
impl_plain_builtin_dtype!(
    I64,
    i64,
    DTypeKind::SignedInteger,
    StorageEncoding::scalar(8, 8),
    "i64"
);
impl_plain_builtin_dtype!(
    F16,
    f16,
    DTypeKind::Float,
    StorageEncoding::scalar(2, 2),
    "f16"
);
impl_plain_builtin_dtype!(
    BF16,
    bf16,
    DTypeKind::Float,
    StorageEncoding::scalar(2, 2),
    "bf16"
);

impl FloatDType for f32 {}
impl FloatDType for f64 {}
impl FloatDType for f16 {}
impl FloatDType for bf16 {}

impl IntDType for u8 {}
impl IntDType for u32 {}
impl IntDType for i64 {}

impl DType for bool {
    type Arg = ();
    type Field = PhantomData<bool>;

    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }

    fn descriptor(_: &Self::Field) -> DTypeDescriptor {
        Self::DESCRIPTOR
    }
}

impl ConstDType for bool {
    const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::builtin(
        DTypeId::Bool,
        DTypeKey::new("incin", "bool", 1),
        DTypeKind::Bool,
        StorageEncoding::scalar(1, 1),
    );
}

impl BuiltinDType for bool {
    const DTYPE: DTypeId = DTypeId::Bool;
}

impl sealed::TensorElementSealed for bool {}

impl PlainDType for bool {
    type Elem = bool;
}

impl BoolDType for bool {}

// ============================================================================
// Q8_0 trait implementations (no TensorElement, no PlainDType, no Pod)
// ============================================================================

impl DType for Q8_0 {
    /// No argument needed — the dtype is fixed by the Rust type itself.
    type Arg = ();
    /// Zero-sized: the value is fixed by the type.
    type Field = PhantomData<Q8_0>;

    /// No-op: nothing to convert.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }

    fn descriptor(_: &Self::Field) -> DTypeDescriptor {
        Self::DESCRIPTOR
    }
}

impl ConstDType for Q8_0 {
    /// Q8_0 block encoding: 32 logical i8 values + 1 f16 scale = 34 bytes,
    /// 2-byte aligned. This is the single authoritative definition.
    const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::builtin(
        DTypeId::Q8_0,
        DTypeKey::new("incin", "q8_0", 1),
        DTypeKind::Quantized,
        StorageEncoding::block(32, 34, 2),
    );
}

impl BuiltinDType for Q8_0 {
    const DTYPE: DTypeId = DTypeId::Q8_0;
}

impl QuantDType for Q8_0 {}

impl Default for DTypeDescriptor {
    fn default() -> Self {
        DTypeId::F32.descriptor()
    }
}

impl From<DTypeId> for DTypeDescriptor {
    fn from(id: DTypeId) -> Self {
        id.descriptor()
    }
}

// ============================================================================
// Dyn dtype
// ============================================================================

impl DType for Dyn {
    /// The runtime-chosen dtype descriptor.
    type Arg = DTypeDescriptor;
    /// Stored directly — `Dyn`'s whole point is deferring dtype choice
    /// to runtime, so `Field` is the `DTypeDescriptor` itself.
    type Field = DTypeDescriptor;

    /// Stores the `DTypeDescriptor` verbatim.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }

    /// Returns the stored `DTypeDescriptor`.
    fn descriptor(field: &Self::Field) -> DTypeDescriptor {
        *field
    }
}
