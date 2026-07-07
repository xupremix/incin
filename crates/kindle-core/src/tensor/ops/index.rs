//! Element-wise tensor operations with compile-time shape checking.
//!
//! Operations require matching Shape, DType, Device, and RequiresGrad.
//! This ensures at compile time that you can't accidentally add tensors
//! of different shapes, dtypes, or on different devices.

use crate::prelude::{Backend, Dyn, DynShape, RequiresGrad, Result, Shape, Tensor};
use crate::nn::loss::{Mean, ReductionMode, CrossEntropyReductionShape, MseReductionShape, L1ReductionShape, BceReductionShape, Reduction};

use alloc::vec::Vec;
use alloc::format;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexSpec {
    All,
    Range(isize, isize),
    RangeFrom(isize),
    RangeTo(isize),
    Index(isize),
}

impl From<isize> for IndexSpec {
    fn from(idx: isize) -> Self {
        IndexSpec::Index(idx)
    }
}
impl From<core::ops::Range<isize>> for IndexSpec {
    fn from(r: core::ops::Range<isize>) -> Self {
        IndexSpec::Range(r.start, r.end)
    }
}
impl From<core::ops::RangeFrom<isize>> for IndexSpec {
    fn from(r: core::ops::RangeFrom<isize>) -> Self {
        IndexSpec::RangeFrom(r.start)
    }
}
impl From<core::ops::RangeTo<isize>> for IndexSpec {
    fn from(r: core::ops::RangeTo<isize>) -> Self {
        IndexSpec::RangeTo(r.end)
    }
}

impl From<usize> for IndexSpec {
    fn from(idx: usize) -> Self {
        IndexSpec::Index(idx as isize)
    }
}
impl From<core::ops::Range<usize>> for IndexSpec {
    fn from(r: core::ops::Range<usize>) -> Self {
        IndexSpec::Range(r.start as isize, r.end as isize)
    }
}
impl From<core::ops::RangeFrom<usize>> for IndexSpec {
    fn from(r: core::ops::RangeFrom<usize>) -> Self {
        IndexSpec::RangeFrom(r.start as isize)
    }
}
impl From<core::ops::RangeTo<usize>> for IndexSpec {
    fn from(r: core::ops::RangeTo<usize>) -> Self {
        IndexSpec::RangeTo(r.end as isize)
    }
}
impl From<i32> for IndexSpec {
    fn from(idx: i32) -> Self {
        IndexSpec::Index(idx as isize)
    }
}

impl From<core::ops::Range<i32>> for IndexSpec {
    fn from(r: core::ops::Range<i32>) -> Self {
        IndexSpec::Range(r.start as isize, r.end as isize)
    }
}
impl From<core::ops::RangeFrom<i32>> for IndexSpec {
    fn from(r: core::ops::RangeFrom<i32>) -> Self {
        IndexSpec::RangeFrom(r.start as isize)
    }
}
impl From<core::ops::RangeTo<i32>> for IndexSpec {
    fn from(r: core::ops::RangeTo<i32>) -> Self {
        IndexSpec::RangeTo(r.end as isize)
    }
}
impl From<core::ops::RangeFull> for IndexSpec {
    fn from(_: core::ops::RangeFull) -> Self {
        IndexSpec::All
    }
}

pub trait ShapeEq<Other> {
    const SHAPES_EQUAL: bool;
    const ASSERT_SHAPES_MATCH: ();
}

impl<S> ShapeEq<S> for S {
    const SHAPES_EQUAL: bool = true;
    const ASSERT_SHAPES_MATCH: () = assert!(
        Self::SHAPES_EQUAL,
        "Shape Mismatch: Attempted to operate on tensors of incompatible shapes."
    );
}

pub trait DTypeEq<Other> {
    const DTYPES_EQUAL: bool;
    const ASSERT_DTYPES_MATCH: ();
}

impl<T> DTypeEq<T> for T {
    const DTYPES_EQUAL: bool = true;
    const ASSERT_DTYPES_MATCH: () = assert!(
        Self::DTYPES_EQUAL,
        "DType Mismatch: Attempted to operate on tensors of incompatible datatypes."
    );
}

