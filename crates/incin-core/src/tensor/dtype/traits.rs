use super::*;

// ============================================================================
// DType - the fundamental trait for any logical dtype
// ============================================================================

/// A type-level tensor dtype: a logical description of what kind of values a
/// tensor holds.
///
/// Implement this to define a new logical dtype - no `DTypeId` variant required.
/// The `descriptor` method returns the dtype's full description (key, kind,
/// physical encoding) at any given runtime moment.
///
/// For compile-time-known dtypes, also implement [`ConstDType`].
/// For dtypes that have an ordinary Rust scalar element, also implement
/// [`PlainDType`].
pub trait DType: 'static + Clone + Debug + Send + Sync + PartialEq {
    /// The user-facing constructor argument (`()` for compile-time-fixed
    /// dtypes, `DTypeId` for `Dyn`).
    type Arg;
    /// The runtime-stored representation (a `PhantomData` for compile-
    /// time-fixed dtypes, `DTypeId` for `Dyn`).
    type Field: Debug + Clone + Default;
    /// Converts a user-facing `Arg` into the stored `Field` representation.
    fn init(arg: Self::Arg) -> Self::Field;
    /// Returns the full logical + physical descriptor for this dtype instance.
    fn descriptor(field: &Self::Field) -> DTypeDescriptor;
}

// ============================================================================
// ConstDType - dtype known at compile time
// ============================================================================

/// A `DType` whose identity is fully known at compile time (as opposed to
/// `Dyn`, which is resolved at runtime).
///
/// A compile-time-known dtype does **not** imply that it has an ordinary Rust
/// scalar element - `Q8_0` implements `ConstDType` but does NOT implement
/// [`PlainDType`] because Q8_0 is a block-quantized format with no per-element
/// Rust scalar. See [`PlainDType`] for dtypes that genuinely have one Rust
/// scalar value per logical element.
///
/// Callers that require a built-in `DTypeId` should use the
/// [`BuiltinDType`] bound instead of `ConstDType`.
pub trait ConstDType: DType<Arg = ()> {
    /// The compile-time-known full descriptor (key, kind, encoding).
    const DESCRIPTOR: DTypeDescriptor;
}

// ============================================================================
// BuiltinDType - compile-time dtype with a current built-in DTypeId
// ============================================================================

/// A [`ConstDType`] that additionally has a current built-in [`DTypeId`].
///
/// All current Incin built-in dtypes (`f32`, `f64`, `f16`, `bf16`, `u8`,
/// `u32`, `i64`, `Q8_0`) implement this.
///
/// Subsystems that still use the closed built-in `DTypeId` vocabulary
/// (distributed plans, capability registry, operation catalog, serialization,
/// backend kernel tables) should require `K: BuiltinDType` rather than
/// requiring every `ConstDType` to have a built-in ID. This makes the
/// temporary limitation explicit in the type system.
///
/// # Design boundary
///
/// A future phase will migrate those subsystems to accept arbitrary descriptors.
/// Until then, `BuiltinDType` is the honest narrow bound.
pub trait BuiltinDType: ConstDType {
    /// The built-in `DTypeId` for this dtype.
    const DTYPE: DTypeId;
}

// ============================================================================
// TensorElement - ordinary Rust POD scalar element
// ============================================================================

pub(super) mod sealed {
    pub trait TensorElementSealed {}
}

/// Marker trait enforcing that a tensor element type is POD, Zeroable, safe,
/// and sealed (`SEC-005`). The set of implementors is limited to Incin's
/// built-in scalar element types; custom logical [`DType`] implementations
/// remain supported, but cannot provide a custom [`TensorElement`].
///
/// Only scalar (non-block) dtypes have a `TensorElement`. Block-quantized
/// dtypes such as `Q8_0` do NOT implement this - their physical representation
/// is backend-specific and carries no per-element Rust type.
pub trait TensorElement:
    sealed::TensorElementSealed
    + bytemuck::NoUninit
    + bytemuck::Zeroable
    + Copy
    + Debug
    + Send
    + Sync
    + 'static
{
}

impl<T> TensorElement for T where
    T: sealed::TensorElementSealed
        + bytemuck::NoUninit
        + bytemuck::Zeroable
        + Copy
        + Debug
        + Send
        + Sync
        + 'static
{
}

// ============================================================================
// PlainDType - dtype with a real ordinary Rust POD scalar element
// ============================================================================

/// A [`ConstDType`] that additionally has a plain Rust POD scalar element.
///
/// The `Elem` associated type is the Rust type whose bytes are the storage
/// representation - `f32` for `f32`, `half::f16` for `f16`, etc.
///
/// **Q8_0 does NOT implement `PlainDType`** because Q8_0 is a block-quantized
/// format. Physical data is a sequence of `BlockQ8_0` structs, not `Q8_0`
/// scalars. APIs that accept `&[K::Elem]` slices (e.g. `from_slice`) require
/// `PlainDType` so that quantized formats are rejected at compile time.
pub trait PlainDType: ConstDType {
    /// The Rust type stored for each logical element.
    type Elem: TensorElement;
}

// ============================================================================
// Semantic marker traits
// ============================================================================

/// Marker for floating-point dtypes (`f32`/`f64`/`f16`/`bf16`).
pub trait FloatDType: PlainDType {}
/// Marker for integer dtypes (`u8`/`u32`/`i64`).
pub trait IntDType: PlainDType {}
/// Marker for the boolean dtype.
pub trait BoolDType: ConstDType {}
/// Marker for block-quantized dtypes (e.g. `Q8_0`) - storage formats with
/// their own internal scale/block structure, not plain scalar elements.
pub trait QuantDType: ConstDType {}
