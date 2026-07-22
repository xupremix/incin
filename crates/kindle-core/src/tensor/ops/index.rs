//! Tensor indexing, slicing, and dynamic stacking operations.
//!
//! This module provides methods to interact with sub-regions of tensors (e.g. slicing, narrowing)
//! as well as operations to concatenate or stack multiple tensors together. These methods ensure
//! that the resulting dimensions are verified and computed either at compile-time (using `Axis`)
//! or dynamically (using `try_stack` / `dyn_slice`) depending on the operation chosen.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// One dimension's worth of an indexing expression like `t.get((0, 2..5, ..))`,
/// built via the `From` impls below so callers can write plain Rust ranges
/// and integers instead of constructing this directly.
pub enum IndexSpec {
    /// The whole dimension, unchanged (from `..`).
    All,
    /// A `start..end` window, negative indices counting from the end
    /// (Python-style).
    Range(isize, isize),
    /// A `start..` window to the end of the dimension.
    RangeFrom(isize),
    /// A `..end` window from the start of the dimension.
    RangeTo(isize),
    /// A single index, removing this dimension from the result.
    Index(isize),
}

impl From<isize> for IndexSpec {
    /// A bare integer indexes a single position (negative = from the end).
    fn from(idx: isize) -> Self {
        IndexSpec::Index(idx)
    }
}
impl From<core::ops::Range<isize>> for IndexSpec {
    /// `a..b` becomes a `Range` window.
    fn from(r: core::ops::Range<isize>) -> Self {
        IndexSpec::Range(r.start, r.end)
    }
}
impl From<core::ops::RangeFrom<isize>> for IndexSpec {
    /// `a..` becomes a `RangeFrom` window.
    fn from(r: core::ops::RangeFrom<isize>) -> Self {
        IndexSpec::RangeFrom(r.start)
    }
}
impl From<core::ops::RangeTo<isize>> for IndexSpec {
    /// `..b` becomes a `RangeTo` window.
    fn from(r: core::ops::RangeTo<isize>) -> Self {
        IndexSpec::RangeTo(r.end)
    }
}

impl From<usize> for IndexSpec {
    /// Widens to `isize` and indexes a single position.
    fn from(idx: usize) -> Self {
        IndexSpec::Index(idx as isize)
    }
}
impl From<core::ops::Range<usize>> for IndexSpec {
    /// Widens bounds to `isize` and becomes a `Range` window.
    fn from(r: core::ops::Range<usize>) -> Self {
        IndexSpec::Range(r.start as isize, r.end as isize)
    }
}
impl From<core::ops::RangeFrom<usize>> for IndexSpec {
    /// Widens the bound to `isize` and becomes a `RangeFrom` window.
    fn from(r: core::ops::RangeFrom<usize>) -> Self {
        IndexSpec::RangeFrom(r.start as isize)
    }
}
impl From<core::ops::RangeTo<usize>> for IndexSpec {
    /// Widens the bound to `isize` and becomes a `RangeTo` window.
    fn from(r: core::ops::RangeTo<usize>) -> Self {
        IndexSpec::RangeTo(r.end as isize)
    }
}
impl From<i32> for IndexSpec {
    /// Widens to `isize` and indexes a single position (negative = from
    /// the end).
    fn from(idx: i32) -> Self {
        IndexSpec::Index(idx as isize)
    }
}

impl From<core::ops::Range<i32>> for IndexSpec {
    /// Widens bounds to `isize` and becomes a `Range` window.
    fn from(r: core::ops::Range<i32>) -> Self {
        IndexSpec::Range(r.start as isize, r.end as isize)
    }
}
impl From<core::ops::RangeFrom<i32>> for IndexSpec {
    /// Widens the bound to `isize` and becomes a `RangeFrom` window.
    fn from(r: core::ops::RangeFrom<i32>) -> Self {
        IndexSpec::RangeFrom(r.start as isize)
    }
}
impl From<core::ops::RangeTo<i32>> for IndexSpec {
    /// Widens the bound to `isize` and becomes a `RangeTo` window.
    fn from(r: core::ops::RangeTo<i32>) -> Self {
        IndexSpec::RangeTo(r.end as isize)
    }
}
impl From<core::ops::RangeFull> for IndexSpec {
    /// `..` selects the entire dimension.
    fn from(_: core::ops::RangeFull) -> Self {
        IndexSpec::All
    }
}

/// Converts an indexing argument — a single `IndexSpec`-convertible value,
/// or a tuple of up to 7 of them (one per dimension) — into a flat
/// `Vec<IndexSpec>` for `get`/`slice`-style methods to consume.
pub trait IndexArgs {
    /// Produces one `IndexSpec` per indexed dimension, in order.
    fn into_specs(self) -> alloc::vec::Vec<IndexSpec>;
}

impl<T: Into<IndexSpec>> IndexArgs for T {
    /// A single value indexes a single (the first) dimension.
    fn into_specs(self) -> alloc::vec::Vec<IndexSpec> {
        alloc::vec![self.into()]
    }
}

macro_rules! impl_index_args_tuple {
    ($($t:ident),+) => {
        impl<$($t: Into<IndexSpec>),+> IndexArgs for ($($t,)+) {
            /// Converts each tuple element to an `IndexSpec`, in order.
            fn into_specs(self) -> alloc::vec::Vec<IndexSpec> {
                let mut specs = alloc::vec::Vec::new();
                #[allow(non_snake_case)]
                let ($($t,)+) = self;
                $(
                    specs.push($t.into());
                )+
                specs
            }
        }
    };
}

impl_index_args_tuple!(A);
impl_index_args_tuple!(A, B);
impl_index_args_tuple!(A, B, C);
impl_index_args_tuple!(A, B, C, D);
impl_index_args_tuple!(A, B, C, D, E);
impl_index_args_tuple!(A, B, C, D, E, F);
impl_index_args_tuple!(A, B, C, D, E, F, G);

/// Compile-time shape-equality check between two `Shape` types, used to
/// reject shape-mismatched tensor operations before runtime rather than
/// after (where possible — dynamic shapes still need a runtime check too).
pub trait ShapeEq<Other> {
    /// `true` iff `Self` and `Other` are the same shape type.
    const SHAPES_EQUAL: bool;
    /// Evaluating this associated const panics at compile time (via
    /// `assert!` in a `const` context) if `SHAPES_EQUAL` is `false`.
    const ASSERT_SHAPES_MATCH: ();
}

impl<S> ShapeEq<S> for S {
    /// A type is always shape-equal to itself.
    const SHAPES_EQUAL: bool = true;
    /// Always passes, since `S: ShapeEq<S>` only reaches this impl when
    /// the two types genuinely match.
    const ASSERT_SHAPES_MATCH: () = assert!(
        Self::SHAPES_EQUAL,
        "Shape Mismatch: Attempted to operate on tensors of incompatible shapes."
    );
}

/// Compile-time dtype-equality check between two `DType` types, the same
/// pattern as `ShapeEq` but for element type instead of shape.
pub trait DTypeEq<Other> {
    /// `true` iff `Self` and `Other` are the same dtype type.
    const DTYPES_EQUAL: bool;
    /// Evaluating this associated const panics at compile time if
    /// `DTYPES_EQUAL` is `false`.
    const ASSERT_DTYPES_MATCH: ();
}

impl<T> DTypeEq<T> for T {
    /// A type is always dtype-equal to itself.
    const DTYPES_EQUAL: bool = true;
    /// Always passes, since `T: DTypeEq<T>` only reaches this impl when
    /// the two types genuinely match.
    const ASSERT_DTYPES_MATCH: () = assert!(
        Self::DTYPES_EQUAL,
        "DType Mismatch: Attempted to operate on tensors of incompatible datatypes."
    );
}
