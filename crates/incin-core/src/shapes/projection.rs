//! Internal lowering from sealed shape semantics to compiler expressions.
//!
//! Compiler IR intentionally lives on this side of the boundary. `Shape` and
//! `Dim` describe validated semantic facts; this module is the adapter used by
//! tracing and symbolic capture.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use super::dim::{
    AddDim, BroadcastExtent, CheckedSubDim, ConstDim, Dim, ExactDivDim, MulDim, NamedDim,
    StaticExtent,
};
use super::shape::{DimCons, Nil, Ranked, Shape};
use crate::exec::{DimExpr, RankExpr, ShapeExpr, SymbolId};
use crate::shapes::Dyn;
use typenum::{Bit, UInt, UTerm, Unsigned};

pub trait DimProjection {
    fn project(axis: usize, base: u32) -> DimExpr;
}

fn basic<D: Dim>(axis: usize, base: u32) -> DimExpr {
    match D::STATIC {
        StaticExtent::Value(value) => DimExpr::Const(value),
        StaticExtent::RuntimeUnknown => DimExpr::Fresh(base.saturating_add(axis as u32)),
        StaticExtent::Invalid => DimExpr::Unknown,
    }
}

impl DimProjection for usize {
    fn project(axis: usize, base: u32) -> DimExpr {
        basic::<Self>(axis, base)
    }
}
impl<const N: usize> DimProjection for ConstDim<N> {
    fn project(axis: usize, base: u32) -> DimExpr {
        basic::<Self>(axis, base)
    }
}
impl DimProjection for UTerm {
    fn project(axis: usize, base: u32) -> DimExpr {
        basic::<Self>(axis, base)
    }
}
impl<U, B> DimProjection for UInt<U, B>
where
    UInt<U, B>: Dim,
    U: Unsigned + Dim,
    B: Bit + Default + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq + 'static,
{
    fn project(axis: usize, base: u32) -> DimExpr {
        basic::<Self>(axis, base)
    }
}

macro_rules! binary_projection {
    ($ty:ident, $variant:ident) => {
        impl<A: DimProjection, B: DimProjection> DimProjection for $ty<A, B> {
            fn project(axis: usize, base: u32) -> DimExpr {
                DimExpr::$variant(
                    Box::new(A::project(axis, base)),
                    Box::new(B::project(axis, base.saturating_add(1))),
                )
                .simplify()
            }
        }
    };
}
binary_projection!(AddDim, Add);
binary_projection!(CheckedSubDim, Sub);
binary_projection!(ExactDivDim, ExactDiv);
binary_projection!(MulDim, Mul);

impl<L: DimProjection, R: DimProjection> DimProjection for BroadcastExtent<L, R> {
    fn project(axis: usize, base: u32) -> DimExpr {
        DimExpr::Broadcast(
            Box::new(L::project(axis, base)),
            Box::new(R::project(axis, base.saturating_add(1))),
        )
        .simplify()
    }
}

impl<Tag: super::dim::AxisTag, Extent: DimProjection> DimProjection for NamedDim<Tag, Extent> {
    fn project(axis: usize, base: u32) -> DimExpr {
        DimExpr::NamedExpr {
            expr: Box::new(Extent::project(axis, base)),
            id: SymbolId(base.saturating_add(axis as u32)),
            name: String::from(Tag::NAME),
            identity: Tag::key().qualified(),
        }
    }
}

pub trait ShapeProjection {
    fn project(base: u32) -> ShapeExpr;
}

impl ShapeProjection for Nil {
    fn project(_: u32) -> ShapeExpr {
        ShapeExpr {
            rank: RankExpr::Static(0),
            dims: Vec::new(),
            constraints: Vec::new(),
        }
    }
}

impl<H: DimProjection, T: ShapeProjection> ShapeProjection for DimCons<H, T> {
    fn project(base: u32) -> ShapeExpr {
        let tail = T::project(base.saturating_add(1));
        let mut dims = alloc::vec![H::project(0, base)];
        dims.extend(tail.dims);
        ShapeExpr {
            rank: RankExpr::Static(dims.len()),
            dims,
            constraints: tail.constraints,
        }
    }
}

impl<R: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static> ShapeProjection for Ranked<R> {
    fn project(base: u32) -> ShapeExpr {
        ShapeExpr::symbolic(&(0..R::USIZE).map(|_| 0).collect::<Vec<_>>(), base)
    }
}

impl ShapeProjection for Dyn {
    fn project(_: u32) -> ShapeExpr {
        ShapeExpr {
            rank: RankExpr::Dynamic,
            dims: Vec::new(),
            constraints: Vec::new(),
        }
    }
}

pub(crate) fn shape_expr<S: Shape + ShapeProjection>(base: u32) -> ShapeExpr {
    S::project(base)
}
