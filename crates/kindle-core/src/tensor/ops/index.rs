//! Tensor indexing, slicing, and dynamic stacking operations.
//!
//! This module provides methods to interact with sub-regions of tensors (e.g. slicing, narrowing)
//! as well as operations to concatenate or stack multiple tensors together. These methods ensure
//! that the resulting dimensions are verified and computed either at compile-time (using `Axis`)
//! or dynamically (using `try_stack` / `dyn_slice`) depending on the operation chosen.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// `IndexSpec`.
pub enum IndexSpec {
    /// `All`.
    All,
    /// `Range`.
    Range(isize, isize),
    /// `RangeFrom`.
    RangeFrom(isize),
    /// `RangeTo`.
    RangeTo(isize),
    /// `Index`.
    Index(isize),
}

impl From<isize> for IndexSpec {
    /// `from`.
    fn from(idx: isize) -> Self {
        IndexSpec::Index(idx)
    }
}
impl From<core::ops::Range<isize>> for IndexSpec {
    /// `from`.
    fn from(r: core::ops::Range<isize>) -> Self {
        IndexSpec::Range(r.start, r.end)
    }
}
impl From<core::ops::RangeFrom<isize>> for IndexSpec {
    /// `from`.
    fn from(r: core::ops::RangeFrom<isize>) -> Self {
        IndexSpec::RangeFrom(r.start)
    }
}
impl From<core::ops::RangeTo<isize>> for IndexSpec {
    /// `from`.
    fn from(r: core::ops::RangeTo<isize>) -> Self {
        IndexSpec::RangeTo(r.end)
    }
}

impl From<usize> for IndexSpec {
    /// `from`.
    fn from(idx: usize) -> Self {
        IndexSpec::Index(idx as isize)
    }
}
impl From<core::ops::Range<usize>> for IndexSpec {
    /// `from`.
    fn from(r: core::ops::Range<usize>) -> Self {
        IndexSpec::Range(r.start as isize, r.end as isize)
    }
}
impl From<core::ops::RangeFrom<usize>> for IndexSpec {
    /// `from`.
    fn from(r: core::ops::RangeFrom<usize>) -> Self {
        IndexSpec::RangeFrom(r.start as isize)
    }
}
impl From<core::ops::RangeTo<usize>> for IndexSpec {
    /// `from`.
    fn from(r: core::ops::RangeTo<usize>) -> Self {
        IndexSpec::RangeTo(r.end as isize)
    }
}
impl From<i32> for IndexSpec {
    /// `from`.
    fn from(idx: i32) -> Self {
        IndexSpec::Index(idx as isize)
    }
}

impl From<core::ops::Range<i32>> for IndexSpec {
    /// `from`.
    fn from(r: core::ops::Range<i32>) -> Self {
        IndexSpec::Range(r.start as isize, r.end as isize)
    }
}
impl From<core::ops::RangeFrom<i32>> for IndexSpec {
    /// `from`.
    fn from(r: core::ops::RangeFrom<i32>) -> Self {
        IndexSpec::RangeFrom(r.start as isize)
    }
}
impl From<core::ops::RangeTo<i32>> for IndexSpec {
    /// `from`.
    fn from(r: core::ops::RangeTo<i32>) -> Self {
        IndexSpec::RangeTo(r.end as isize)
    }
}
impl From<core::ops::RangeFull> for IndexSpec {
    /// `from`.
    fn from(_: core::ops::RangeFull) -> Self {
        IndexSpec::All
    }
}

/// `IndexArgs`.
pub trait IndexArgs {
    /// `into_specs`.
    fn into_specs(self) -> alloc::vec::Vec<IndexSpec>;
}

impl<T: Into<IndexSpec>> IndexArgs for T {
    /// `into_specs`.
    fn into_specs(self) -> alloc::vec::Vec<IndexSpec> {
        alloc::vec![self.into()]
    }
}

macro_rules! impl_index_args_tuple {
    ($($t:ident),+) => {
        impl<$($t: Into<IndexSpec>),+> IndexArgs for ($($t,)+) {
            /// `into_specs`.
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

/// `ShapeEq`.
pub trait ShapeEq<Other> {
    /// `SHAPES_EQUAL`.
    const SHAPES_EQUAL: bool;
    /// `ASSERT_SHAPES_MATCH`.
    const ASSERT_SHAPES_MATCH: ();
}

impl<S> ShapeEq<S> for S {
    /// `SHAPES_EQUAL`.
    const SHAPES_EQUAL: bool = true;
    /// `ASSERT_SHAPES_MATCH`.
    const ASSERT_SHAPES_MATCH: () = assert!(
        Self::SHAPES_EQUAL,
        "Shape Mismatch: Attempted to operate on tensors of incompatible shapes."
    );
}

/// `DTypeEq`.
pub trait DTypeEq<Other> {
    /// `DTYPES_EQUAL`.
    const DTYPES_EQUAL: bool;
    /// `ASSERT_DTYPES_MATCH`.
    const ASSERT_DTYPES_MATCH: ();
}

impl<T> DTypeEq<T> for T {
    /// `DTYPES_EQUAL`.
    const DTYPES_EQUAL: bool = true;
    /// `ASSERT_DTYPES_MATCH`.
    const ASSERT_DTYPES_MATCH: () = assert!(
        Self::DTYPES_EQUAL,
        "DType Mismatch: Attempted to operate on tensors of incompatible datatypes."
    );
}
