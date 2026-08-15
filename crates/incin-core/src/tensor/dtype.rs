use crate::shapes::Dyn;
use crate::shapes::error::{OperationKind, ShapeError};

use core::{fmt::Debug, marker::PhantomData};
pub use half::{bf16, f16};

// ============================================================================
// DTypeKey — stable logical dtype identity
// ============================================================================

/// A stable, extensible logical dtype key.
///
/// Logical identity is independent of any Rust type, any `DTypeId` variant, or
/// any backend capability. Two descriptors are the same logical dtype when they
/// have the same `DTypeKey`.
///
/// Built-in Incin dtypes use the `"incin"` namespace. Custom dtypes should use
/// a project-specific namespace to avoid collisions.
///
/// Do NOT use `std::any::TypeId` as logical dtype identity. `TypeId` is a
/// process/compiler identity, not a stable storage or interchange identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DTypeKey {
    namespace: &'static str,
    name: &'static str,
    version: u32,
}

impl serde::Serialize for DTypeKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (self.namespace, self.name, self.version).serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for DTypeKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (ns, name, version) =
            <(alloc::string::String, alloc::string::String, u32)>::deserialize(deserializer)?;

        if ns == "incin" {
            match (name.as_str(), version) {
                ("u8", 1) => return Ok(<u8 as ConstDType>::DESCRIPTOR.key()),
                ("u32", 1) => return Ok(<u32 as ConstDType>::DESCRIPTOR.key()),
                ("i64", 1) => return Ok(<i64 as ConstDType>::DESCRIPTOR.key()),
                ("f16", 1) => return Ok(<half::f16 as ConstDType>::DESCRIPTOR.key()),
                ("bf16", 1) => return Ok(<half::bf16 as ConstDType>::DESCRIPTOR.key()),
                ("f32", 1) => return Ok(<f32 as ConstDType>::DESCRIPTOR.key()),
                ("f64", 1) => return Ok(<f64 as ConstDType>::DESCRIPTOR.key()),
                ("bool", 1) => return Ok(<bool as ConstDType>::DESCRIPTOR.key()),
                ("q8_0", 1) => return Ok(<Q8_0 as ConstDType>::DESCRIPTOR.key()),
                _ => {}
            }
        }

        Err(serde::de::Error::custom(alloc::format!(
            "Deserializing custom DTypeKey ({}, {}, {}) is not supported without a persistent DType registry",
            ns,
            name,
            version
        )))
    }
}

impl DTypeKey {
    /// Constructs a new `DTypeKey` from its three components.
    ///
    /// The combination of `(namespace, name, version)` must be unique across
    /// all dtypes you use together. The `"incin"` namespace is reserved for
    /// built-in Incin dtypes.
    pub const fn new(namespace: &'static str, name: &'static str, version: u32) -> Self {
        Self {
            namespace,
            name,
            version,
        }
    }

    /// The namespace component (e.g. `"incin"` for built-ins).
    pub const fn namespace(self) -> &'static str {
        self.namespace
    }

    /// The name component (e.g. `"f32"`, `"q8_0"`).
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// The version component. Start at `1`; increment when the physical layout
    /// changes in an incompatible way.
    pub const fn version(self) -> u32 {
        self.version
    }
}

// ============================================================================
// DTypeKind — broad semantic classification
// ============================================================================

/// Broad semantic category of a dtype.
///
/// Used for classification and display, not for operation dispatch.
/// The variants exist so that future logical types fit without another
/// category redesign.
///
/// `Bool` is supported by logical and comparison operations. `Complex` remains
/// reserved for future arithmetic support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DTypeKind {
    /// Boolean dtype for logical and comparison operations.
    Bool,
    /// Unsigned integer dtype (e.g. `u8`, `u32`).
    UnsignedInteger,
    /// Signed integer dtype (e.g. `i64`).
    SignedInteger,
    /// Floating-point dtype (e.g. `f16`, `bf16`, `f32`, `f64`).
    Float,
    /// Complex dtype (reserved for future use).
    Complex,
    /// Block-quantized dtype (e.g. `Q8_0`).
    Quantized,
    /// Opaque dtype with custom storage encoding, not interpretable by Incin
    /// built-in backends.
    Opaque,
}

// ============================================================================
// StorageEncoding — physical byte/block encoding contract
// ============================================================================

/// Physical byte/block encoding of a dtype's storage.
///
/// This one representation handles both scalar and block storage:
///
/// - **Scalar dtypes** (e.g. `f32`): one logical value per block, `bytes_per_block`
///   equals the scalar size.
/// - **Block dtypes** (e.g. `Q8_0`): multiple logical values share one physical
///   block. The block size may not evenly divide into per-element bytes.
///
/// `StorageEncoding` is the single authoritative source of storage arithmetic.
/// Do not compute `numel * element_size` anywhere outside of this type; use
/// [`size_bytes`](Self::size_bytes) instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct StorageEncoding {
    logical_elements_per_block: usize,
    bytes_per_block: usize,
    alignment: usize,
}

impl<'de> serde::Deserialize<'de> for StorageEncoding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Wire {
            logical_elements_per_block: usize,
            bytes_per_block: usize,
            alignment: usize,
        }

        let wire = <Wire as serde::Deserialize>::deserialize(deserializer)?;
        if wire.logical_elements_per_block == 0
            || wire.bytes_per_block == 0
            || wire.alignment == 0
            || !wire.alignment.is_power_of_two()
        {
            return Err(serde::de::Error::custom(
                "invalid storage encoding: block and alignment fields must be non-zero, and alignment must be a power of two",
            ));
        }
        Ok(Self {
            logical_elements_per_block: wire.logical_elements_per_block,
            bytes_per_block: wire.bytes_per_block,
            alignment: wire.alignment,
        })
    }
}

impl StorageEncoding {
    /// Creates a scalar encoding: one logical value per block.
    ///
    /// # Panics
    ///
    /// Panics if `bytes == 0`, `alignment == 0`, or `alignment` is not a
    /// power of two.
    pub const fn scalar(bytes: usize, alignment: usize) -> Self {
        assert!(bytes > 0, "scalar encoding requires bytes > 0");
        assert!(alignment > 0, "encoding alignment must be > 0");
        assert!(
            alignment.is_power_of_two(),
            "encoding alignment must be a power of two"
        );
        Self {
            logical_elements_per_block: 1,
            bytes_per_block: bytes,
            alignment,
        }
    }

    /// Creates a block encoding: multiple logical values per physical block.
    ///
    /// # Panics
    ///
    /// Panics if any of `logical_elements`, `bytes`, or `alignment` is zero,
    /// or if `alignment` is not a power of two.
    pub const fn block(logical_elements: usize, bytes: usize, alignment: usize) -> Self {
        assert!(
            logical_elements > 0,
            "block encoding requires logical_elements > 0"
        );
        assert!(bytes > 0, "block encoding requires bytes > 0");
        assert!(alignment > 0, "encoding alignment must be > 0");
        assert!(
            alignment.is_power_of_two(),
            "encoding alignment must be a power of two"
        );
        Self {
            logical_elements_per_block: logical_elements,
            bytes_per_block: bytes,
            alignment,
        }
    }

    /// Number of logical values packed into one physical block.
    ///
    /// Always `1` for scalar dtypes.
    pub const fn logical_elements_per_block(self) -> usize {
        self.logical_elements_per_block
    }

    /// Number of physical bytes in one storage block.
    pub const fn bytes_per_block(self) -> usize {
        self.bytes_per_block
    }

    /// Required byte alignment of the storage buffer.
    pub const fn alignment(self) -> usize {
        self.alignment
    }

    /// Returns `true` if this is a scalar (one element per block) encoding.
    pub const fn is_scalar(self) -> bool {
        self.logical_elements_per_block == 1
    }

    /// Returns `true` if this is a block (multi-element) encoding.
    pub const fn is_block(self) -> bool {
        self.logical_elements_per_block > 1
    }

    /// Returns the byte width of a single scalar value, or `None` for block
    /// encodings.
    ///
    /// Callers that require a scalar element width (e.g. scalar kernel ABI
    /// validation) must check for `None` and reject block dtypes explicitly.
    pub const fn scalar_bytes(self) -> Option<usize> {
        if self.is_scalar() {
            Some(self.bytes_per_block)
        } else {
            None
        }
    }

    /// Returns the byte length of storage for `logical_elements` logical values.
    ///
    /// For **scalar** dtypes: `elements * bytes_per_block` (checked).
    ///
    /// For **block** dtypes: `elements` must be divisible by
    /// `logical_elements_per_block`; partial blocks are rejected. Then:
    /// `(elements / logical_elements_per_block) * bytes_per_block` (checked).
    ///
    /// Zero elements always returns `Ok(0)`.
    pub fn size_bytes(
        self,
        logical_elements: usize,
        operation: OperationKind,
    ) -> Result<usize, ShapeError> {
        if logical_elements == 0 {
            return Ok(0);
        }
        let per_block = self.logical_elements_per_block;
        if !logical_elements.is_multiple_of(per_block) {
            return Err(ShapeError::InvalidParameter {
                operation,
                parameter: "elements",
                value: logical_elements,
            });
        }
        let block_count = logical_elements / per_block;
        block_count
            .checked_mul(self.bytes_per_block)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation,
                expression: "block count * block size",
            })
    }
}

// ============================================================================
// DTypeDescriptor — logical + physical description
// ============================================================================

/// Combined logical and physical description of a dtype.
///
/// Every dtype — whether built-in or third-party — is described by a
/// `DTypeDescriptor`. Backends that only support built-in dtypes can extract the
/// [`DTypeId`] via [`builtin_id`](Self::builtin_id) and error gracefully when
/// it is `None`.
///
/// # Design boundary
///
/// `TensorMeta`, capability registries, and the operation catalog still use
/// [`DTypeId`] as their built-in runtime vocabulary. Migrating them to
/// arbitrary descriptors will happen in a future phase. This boundary is
/// intentional and documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct DTypeDescriptor {
    key: DTypeKey,
    kind: DTypeKind,
    encoding: StorageEncoding,
    builtin: Option<DTypeId>,
}

impl<'de> serde::Deserialize<'de> for DTypeDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Wire {
            key: DTypeKey,
            kind: DTypeKind,
            encoding: StorageEncoding,
            builtin: Option<DTypeId>,
        }

        let wire = <Wire as serde::Deserialize>::deserialize(deserializer)?;
        let descriptor = match wire.builtin {
            Some(id) => {
                let expected = id.descriptor();
                if expected.key() != wire.key
                    || expected.kind() != wire.kind
                    || expected.encoding() != wire.encoding
                {
                    return Err(serde::de::Error::custom(
                        "dtype descriptor builtin identity does not match its fields",
                    ));
                }
                Self::builtin(id, wire.key, wire.kind, wire.encoding)
            }
            None => Self::new(wire.key, wire.kind, wire.encoding),
        };
        Ok(descriptor)
    }
}

impl DTypeDescriptor {
    /// Constructs a descriptor for a custom (non-built-in) dtype.
    pub const fn new(key: DTypeKey, kind: DTypeKind, encoding: StorageEncoding) -> Self {
        Self {
            key,
            kind,
            encoding,
            builtin: None,
        }
    }

    /// Constructs a descriptor for a built-in Incin dtype, attaching its
    /// [`DTypeId`] for compatibility with the existing runtime vocabulary.
    pub const fn builtin(
        id: DTypeId,
        key: DTypeKey,
        kind: DTypeKind,
        encoding: StorageEncoding,
    ) -> Self {
        Self {
            key,
            kind,
            encoding,
            builtin: Some(id),
        }
    }

    /// The logical dtype key (stable identity, namespace + name + version).
    pub const fn key(self) -> DTypeKey {
        self.key
    }

    /// Broad semantic category.
    pub const fn kind(self) -> DTypeKind {
        self.kind
    }

    /// Physical byte/block encoding.
    pub const fn encoding(self) -> StorageEncoding {
        self.encoding
    }

    /// The `DTypeId` for built-in dtypes, or `None` for custom dtypes.
    ///
    /// Backends that only support built-in dtypes should call this and return
    /// an error when it is `None`, rather than panicking or silently
    /// misidentifying the dtype.
    pub const fn builtin_id(self) -> Option<DTypeId> {
        self.builtin
    }

    /// Returns the name component of the dtype key (e.g. `"f32"`).
    pub const fn name(self) -> &'static str {
        self.key.name()
    }

    /// Whether this descriptor is an integer dtype (signed or unsigned).
    pub const fn is_integer(self) -> bool {
        matches!(
            self.kind,
            DTypeKind::SignedInteger | DTypeKind::UnsignedInteger
        )
    }

    /// Whether this descriptor is a floating-point dtype.
    pub const fn is_float(self) -> bool {
        matches!(self.kind, DTypeKind::Float)
    }

    /// Whether this descriptor is a block-quantized dtype.
    pub const fn is_quantized(self) -> bool {
        matches!(self.kind, DTypeKind::Quantized)
    }

    /// Whether this descriptor is a boolean dtype.
    pub const fn is_bool(self) -> bool {
        matches!(self.kind, DTypeKind::Bool)
    }

    /// Bytes occupied by `logical_elements` values of this dtype.
    ///
    /// Delegates to [`StorageEncoding::size_bytes`]; see that method for the
    /// full contract.
    pub fn size_bytes(
        self,
        logical_elements: usize,
        operation: OperationKind,
    ) -> Result<usize, ShapeError> {
        self.encoding.size_bytes(logical_elements, operation)
    }
}

// ============================================================================
// DTypeId — built-in runtime compatibility identifier
// ============================================================================

#[non_exhaustive]
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
/// A runtime-identifiable tag for the current built-in Incin dtypes.
///
/// **Role**: compatibility identifier for existing subsystems (`TensorMeta`,
/// capability registry, operation catalog, serialization, distributed planning,
/// backend kernel tables). These subsystems continue to use `DTypeId` as their
/// built-in runtime vocabulary and will be migrated to arbitrary descriptors
/// in a future phase.
///
/// **Not a storage-layout table**: Physical byte layout is authoritative in
/// [`DTypeDescriptor`] / [`StorageEncoding`], not here. All size arithmetic
/// methods on `DTypeId` delegate to [`descriptor`](Self::descriptor) /
/// [`encoding`](Self::encoding).
///
/// To define a **new logical dtype** without adding a `DTypeId` variant, use
/// [`DTypeDescriptor::new`] with a custom [`DTypeKey`]. Built-in backends
/// will reject it via [`builtin_id`](DTypeDescriptor::builtin_id) returning
/// `None`.
pub enum DTypeId {
    /// 8-bit unsigned integer.
    U8,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit signed integer.
    I64,
    /// 16-bit brain floating point.
    BF16,
    /// 16-bit (IEEE 754 half-precision) floating point.
    F16,
    /// 32-bit floating point.
    #[default]
    F32,
    /// 64-bit floating point.
    F64,
    /// Q8_0 block-quantized 8-bit integer.
    ///
    /// Logical: 32 values per block. Physical: one `f16` scale + 32 `i8`
    /// quants = 34 bytes per block.
    Q8_0,
    /// 8-bit logical boolean.
    Bool,
}

impl DTypeId {
    /// The descriptor for this built-in dtype, including its logical key,
    /// semantic kind, and physical storage encoding.
    ///
    /// All classification and sizing methods delegate here.
    #[must_use]
    pub const fn descriptor(self) -> DTypeDescriptor {
        match self {
            DTypeId::U8 => DTypeDescriptor::builtin(
                DTypeId::U8,
                DTypeKey::new("incin", "u8", 1),
                DTypeKind::UnsignedInteger,
                StorageEncoding::scalar(1, 1),
            ),
            DTypeId::U32 => DTypeDescriptor::builtin(
                DTypeId::U32,
                DTypeKey::new("incin", "u32", 1),
                DTypeKind::UnsignedInteger,
                StorageEncoding::scalar(4, 4),
            ),
            DTypeId::I64 => DTypeDescriptor::builtin(
                DTypeId::I64,
                DTypeKey::new("incin", "i64", 1),
                DTypeKind::SignedInteger,
                StorageEncoding::scalar(8, 8),
            ),
            DTypeId::BF16 => DTypeDescriptor::builtin(
                DTypeId::BF16,
                DTypeKey::new("incin", "bf16", 1),
                DTypeKind::Float,
                StorageEncoding::scalar(2, 2),
            ),
            DTypeId::F16 => DTypeDescriptor::builtin(
                DTypeId::F16,
                DTypeKey::new("incin", "f16", 1),
                DTypeKind::Float,
                StorageEncoding::scalar(2, 2),
            ),
            DTypeId::F32 => DTypeDescriptor::builtin(
                DTypeId::F32,
                DTypeKey::new("incin", "f32", 1),
                DTypeKind::Float,
                StorageEncoding::scalar(4, 4),
            ),
            DTypeId::F64 => DTypeDescriptor::builtin(
                DTypeId::F64,
                DTypeKey::new("incin", "f64", 1),
                DTypeKind::Float,
                StorageEncoding::scalar(8, 8),
            ),
            DTypeId::Q8_0 => DTypeDescriptor::builtin(
                DTypeId::Q8_0,
                DTypeKey::new("incin", "q8_0", 1),
                DTypeKind::Quantized,
                // 32 logical i8 values + 1 f16 scale = 34 bytes, 2-byte aligned.
                StorageEncoding::block(32, 34, 2),
            ),
            DTypeId::Bool => DTypeDescriptor::builtin(
                DTypeId::Bool,
                DTypeKey::new("incin", "bool", 1),
                DTypeKind::Bool,
                StorageEncoding::scalar(1, 1),
            ),
        }
    }

    /// The storage encoding for this dtype. Shortcut for
    /// `self.descriptor().encoding()`.
    #[must_use]
    pub const fn encoding(self) -> StorageEncoding {
        self.descriptor().encoding()
    }

    /// The lowercase name used in diagnostics, generated documentation, and
    /// `cargo incin doctor`'s report.
    ///
    /// The counterpart of [`OperationKind::name`](crate::shapes::error::OperationKind::name)
    /// and [`LayoutClass::as_str`](crate::exec::LayoutClass::as_str): one
    /// spelling per dtype, so the capability tables, the doctor's probe lines
    /// and a shape error cannot disagree about what to call `F32`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.descriptor().key().name()
    }

    /// True for `U8`, `U32`, `I64` — dtypes with no fractional part, which
    /// [`Tensor`](crate::tensor::base::Tensor)'s `Display`/`Debug` render as plain
    /// integers rather than with decimal formatting.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self.descriptor().kind(),
            DTypeKind::UnsignedInteger | DTypeKind::SignedInteger
        )
    }

    /// True for `BF16`, `F16`, `F32`, `F64`.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self.descriptor().kind(), DTypeKind::Float)
    }

    /// True for `Q8_0` — a packed block format with no per-element scalar
    /// representation, so printing it requires dequantizing first rather
    /// than reading one value per logical element.
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        matches!(self.descriptor().kind(), DTypeKind::Quantized)
    }

    /// True for `Bool` dtypes.
    #[must_use]
    pub const fn is_bool(self) -> bool {
        matches!(self.descriptor().kind(), DTypeKind::Bool)
    }

    /// True for `Complex` dtypes (none built-in yet).
    #[must_use]
    pub const fn is_complex(self) -> bool {
        matches!(self.descriptor().kind(), DTypeKind::Complex)
    }

    /// Returns the size in bytes of a single element of this dtype.
    ///
    /// # Deprecation
    ///
    /// This method is **deprecated** and retained only for source
    /// compatibility. Use
    /// [`encoding().scalar_bytes()`](StorageEncoding::scalar_bytes) or
    /// [`size_bytes()`](Self::size_bytes) instead.
    ///
    /// `element_size` is **wrong for block dtypes**: `Q8_0` reports `1`
    /// (the raw quant width) but `Q8_0` storage is 34-byte blocks, not
    /// one-byte scalars. Production allocation code must not call this method.
    #[must_use]
    #[deprecated(
        since = "0.1.0",
        note = "Use `encoding().scalar_bytes()` or `size_bytes()` instead. \
                This method is incorrect for block dtypes such as Q8_0."
    )]
    pub fn element_size(&self) -> usize {
        match self {
            DTypeId::U8 | DTypeId::Q8_0 | DTypeId::Bool => 1,
            DTypeId::F16 | DTypeId::BF16 => 2,
            DTypeId::F32 | DTypeId::U32 => 4,
            DTypeId::F64 | DTypeId::I64 => 8,
        }
    }

    /// Logical values packed into one physical storage block.
    ///
    /// Delegates to [`StorageEncoding::logical_elements_per_block`].
    ///
    /// # Deprecation
    ///
    /// Prefer `self.encoding().logical_elements_per_block()`.
    #[must_use]
    #[deprecated(
        since = "0.1.0",
        note = "Use `encoding().logical_elements_per_block()` instead."
    )]
    pub const fn block_elements(self) -> usize {
        self.encoding().logical_elements_per_block()
    }

    /// Bytes occupied by one physical storage block.
    ///
    /// Delegates to [`StorageEncoding::bytes_per_block`].
    ///
    /// # Deprecation
    ///
    /// Prefer `self.encoding().bytes_per_block()`.
    #[must_use]
    #[deprecated(since = "0.1.0", note = "Use `encoding().bytes_per_block()` instead.")]
    pub const fn block_bytes(self) -> usize {
        self.encoding().bytes_per_block()
    }

    /// Bytes occupied by `elements` logical values of this dtype.
    ///
    /// This is the single byte-arithmetic entry point for storage sizing. It
    /// is checked: an element count that fits `usize` but whose byte length
    /// does not is reported rather than silently truncated. A block-quantized
    /// count that does not fill whole blocks is rejected.
    ///
    /// Delegates to [`StorageEncoding::size_bytes`]; the encoding is the
    /// authoritative source of the arithmetic.
    pub fn size_bytes(
        self,
        elements: usize,
        operation: OperationKind,
    ) -> Result<usize, ShapeError> {
        self.encoding().size_bytes(elements, operation)
    }
}

// ============================================================================
// DType — the fundamental trait for any logical dtype
// ============================================================================

/// A type-level tensor dtype: a logical description of what kind of values a
/// tensor holds.
///
/// Implement this to define a new logical dtype — no `DTypeId` variant required.
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
// ConstDType — dtype known at compile time
// ============================================================================

/// A `DType` whose identity is fully known at compile time (as opposed to
/// `Dyn`, which is resolved at runtime).
///
/// A compile-time-known dtype does **not** imply that it has an ordinary Rust
/// scalar element — `Q8_0` implements `ConstDType` but does NOT implement
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
// BuiltinDType — compile-time dtype with a current built-in DTypeId
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
// TensorElement — ordinary Rust POD scalar element
// ============================================================================

pub mod sealed {
    pub trait TensorElementSealed {}
}

/// Marker trait enforcing that a tensor element type is POD, Zeroable, safe, and sealed (`SEC-005`).
///
/// Only scalar (non-block) dtypes have a `TensorElement`. Block-quantized
/// dtypes such as `Q8_0` do NOT implement this — their physical representation
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
// PlainDType — dtype with a real ordinary Rust POD scalar element
// ============================================================================

/// A [`ConstDType`] that additionally has a plain Rust POD scalar element.
///
/// The `Elem` associated type is the Rust type whose bytes are the storage
/// representation — `f32` for `f32`, `half::f16` for `f16`, etc.
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
/// Marker for block-quantized dtypes (e.g. `Q8_0`) — storage formats with
/// their own internal scale/block structure, not plain scalar elements.
pub trait QuantDType: ConstDType {}

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
/// stored as [`BlockQ8_0`](incin_backends::quant::BlockQ8_0) blocks in the
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::error::OperationKind;

    // --- StorageEncoding tests ---

    #[test]
    fn f32_encoding_is_scalar() {
        let enc = DTypeId::F32.encoding();
        assert_eq!(enc.logical_elements_per_block(), 1);
        assert_eq!(enc.bytes_per_block(), 4);
        assert_eq!(enc.alignment(), 4);
        assert!(enc.is_scalar());
        assert!(!enc.is_block());
        assert_eq!(enc.scalar_bytes(), Some(4));
        assert_eq!(enc.size_bytes(64, OperationKind::Storage).unwrap(), 256);
    }

    #[test]
    fn f64_encoding() {
        let enc = DTypeId::F64.encoding();
        assert_eq!(enc.size_bytes(64, OperationKind::Storage).unwrap(), 512);
    }

    #[test]
    fn q8_0_encoding_is_block() {
        let enc = DTypeId::Q8_0.encoding();
        assert_eq!(enc.logical_elements_per_block(), 32);
        assert_eq!(enc.bytes_per_block(), 34);
        assert_eq!(enc.alignment(), 2);
        assert!(!enc.is_scalar());
        assert!(enc.is_block());
        assert_eq!(enc.scalar_bytes(), None);
        assert_eq!(enc.size_bytes(32, OperationKind::Storage).unwrap(), 34);
        assert_eq!(enc.size_bytes(64, OperationKind::Storage).unwrap(), 68);
        assert_eq!(enc.size_bytes(0, OperationKind::Storage).unwrap(), 0);
        assert!(enc.size_bytes(33, OperationKind::Storage).is_err());
    }

    #[test]
    fn scalar_overflow_returns_error() {
        let enc = StorageEncoding::scalar(8, 8);
        assert!(enc.size_bytes(usize::MAX, OperationKind::Storage).is_err());
    }

    #[test]
    fn block_overflow_returns_error() {
        let enc = StorageEncoding::block(32, 34, 2);
        // usize::MAX is not a multiple of 32 in general, so this might
        // fail at the divisibility check or the overflow check.
        assert!(
            enc.size_bytes(usize::MAX - 1, OperationKind::Storage)
                .is_err()
        );
    }

    // --- DTypeDescriptor tests ---

    #[test]
    fn f32_descriptor() {
        let d = DTypeId::F32.descriptor();
        assert_eq!(d.key().namespace(), "incin");
        assert_eq!(d.key().name(), "f32");
        assert_eq!(d.key().version(), 1);
        assert_eq!(d.kind(), DTypeKind::Float);
        assert_eq!(d.builtin_id(), Some(DTypeId::F32));
    }

    #[test]
    fn q8_0_descriptor() {
        let d = DTypeId::Q8_0.descriptor();
        assert_eq!(d.kind(), DTypeKind::Quantized);
        assert!(!d.encoding().is_scalar());
        assert_eq!(d.encoding().logical_elements_per_block(), 32);
        assert_eq!(d.encoding().bytes_per_block(), 34);
        assert_eq!(d.builtin_id(), Some(DTypeId::Q8_0));
    }

    #[test]
    fn builtin_round_trip() {
        for id in [
            DTypeId::U8,
            DTypeId::U32,
            DTypeId::I64,
            DTypeId::BF16,
            DTypeId::F16,
            DTypeId::F32,
            DTypeId::F64,
            DTypeId::Q8_0,
        ] {
            assert_eq!(id.descriptor().builtin_id(), Some(id));
        }
    }

    #[test]
    fn custom_dtype_has_no_builtin_id() {
        let key = DTypeKey::new("test", "packed", 1);
        let enc = StorageEncoding::block(3, 5, 1);
        let desc = DTypeDescriptor::new(key, DTypeKind::Opaque, enc);
        assert_eq!(desc.builtin_id(), None);
    }

    // --- DTypeId classification ---

    #[test]
    fn dtype_id_classification() {
        assert!(DTypeId::F32.is_float());
        assert!(DTypeId::F16.is_float());
        assert!(DTypeId::BF16.is_float());
        assert!(DTypeId::F64.is_float());
        assert!(DTypeId::U8.is_integer());
        assert!(DTypeId::U32.is_integer());
        assert!(DTypeId::I64.is_integer());
        assert!(DTypeId::Q8_0.is_quantized());
        assert!(!DTypeId::F32.is_integer());
        assert!(!DTypeId::U8.is_float());
        assert!(!DTypeId::Q8_0.is_float());
    }

    // --- Extensibility: custom dtype without DTypeId variant ---

    #[test]
    fn custom_dtype_compile_test() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct TestPacked;

        impl DType for TestPacked {
            type Arg = ();
            type Field = core::marker::PhantomData<Self>;
            fn init(_: ()) -> Self::Field {
                core::marker::PhantomData
            }
            fn descriptor(_: &Self::Field) -> DTypeDescriptor {
                Self::DESCRIPTOR
            }
        }

        impl ConstDType for TestPacked {
            const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::new(
                DTypeKey::new("test", "packed", 1),
                DTypeKind::Opaque,
                StorageEncoding::block(3, 5, 1),
            );
        }

        // Verify it compiles and has no builtin ID.
        let desc = TestPacked::DESCRIPTOR;
        assert_eq!(desc.builtin_id(), None);
        assert_eq!(desc.key().namespace(), "test");
        assert_eq!(desc.key().name(), "packed");
        assert_eq!(desc.encoding().logical_elements_per_block(), 3);
        assert_eq!(desc.encoding().bytes_per_block(), 5);
    }

    // --- PlainDType trait bounds compile tests ---

    fn _assert_plain<K: PlainDType>() {}
    fn _assert_not_plain_compile_check() {
        _assert_plain::<f32>();
        _assert_plain::<f64>();
        _assert_plain::<i64>();
        // Q8_0 deliberately NOT listed — it must NOT implement PlainDType.
        // Uncomment the next line to verify it fails to compile:
        // _assert_plain::<Q8_0>();
    }
}
