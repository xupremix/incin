// Argument conversion logic for the Tensor construction system.
//
// The ArgInto trait converts user-friendly arguments into the internal
// representation needed by each tensor parameter (Shape, DType, Device, Grad).
//
// The "lifting" impls allow partial specification: if a user only provides
// the dynamic parameters, the static ones are filled in with ().
// For example, `Tensor::<Dyn, f32, Cpu, Dyn>::zeros(([2, 3], true))`
// lifts `([2,3], true)` into `(Vec<usize>, (), (), bool)`.

use core::marker::PhantomData;

#[cfg(feature = "cuda")]
use crate::prelude::Cuda;

#[cfg(feature = "metal")]
use crate::prelude::Metal;

use crate::prelude::{Const, Cpu, Dim, Grad, KindleDType, KindleDevice, NoGrad};

use alloc::vec::Vec;

/// Trait for converting user-provided arguments into the internal
/// representation expected by tensor parameters.
pub trait ArgInto<Target> {
    fn into_arg(self) -> Target;
}

/// Marker trait for types that represent non-trivial (non-unit) arguments.
/// Used to disambiguate the lifting impls so that `()` positions are
/// correctly identified as "no argument provided".
pub trait NotUnit {}

// ============================================================================
// Self (identity) conversions for primitive types
// ============================================================================

macro_rules! impl_self_arginto {
    ($($t:ty)*) => {
        $(
            impl ArgInto<$t> for $t {
                #[inline(always)]
                fn into_arg(self) -> $t {
                    self
                }
            }
        )*
    };
}

impl_self_arginto! {
    ()
    usize
    bool
    KindleDType
    KindleDevice
    Cpu
    Grad
    NoGrad
}

impl<const N: usize> ArgInto<Const<N>> for Const<N> {
    #[inline(always)]
    fn into_arg(self) -> Const<N> {
        self
    }
}

#[cfg(feature = "cuda")]
impl<const N: usize> ArgInto<Cuda<N>> for Cuda<N> {
    #[inline(always)]
    fn into_arg(self) -> Cuda<N> {
        self
    }
}

#[cfg(feature = "metal")]
impl<const N: usize> ArgInto<Metal<N>> for Metal<N> {
    #[inline(always)]
    fn into_arg(self) -> Metal<N> {
        self
    }
}

// ============================================================================
// Vec / Array / Slice conversions for dynamic shapes
// ============================================================================

impl<D: Dim> ArgInto<Vec<D>> for Vec<D> {
    #[inline(always)]
    fn into_arg(self) -> Vec<D> {
        self
    }
}

impl<const N: usize> ArgInto<Vec<usize>> for [usize; N] {
    #[inline(always)]
    fn into_arg(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<'a> ArgInto<Vec<usize>> for &'a [usize] {
    #[inline(always)]
    fn into_arg(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> ArgInto<[usize; N]> for [usize; N] {
    #[inline(always)]
    fn into_arg(self) -> [usize; N] {
        self
    }
}

// ============================================================================
// NotUnit markers — types that represent actual user-provided values
// ============================================================================

macro_rules! impl_not_unit {
    ($($t:ty)*) => {
        $(
            impl NotUnit for $t {}
        )*
    };
}

impl_not_unit! {
    usize
    bool
    &usize
    Cpu
    Grad
    NoGrad
    KindleDType
    KindleDevice
}

impl<const N: usize> NotUnit for Const<N> {}

#[cfg(feature = "cuda")]
impl<const N: usize> NotUnit for Cuda<N> {}

#[cfg(feature = "metal")]
impl<const N: usize> NotUnit for Metal<N> {}

impl<D: Dim> NotUnit for Vec<D> {}
// Handled in shape.rs

// ============================================================================
// Dim tuple self-conversions (identity for shape args)
// ============================================================================

macro_rules! impl_dim_tuple_arg_into {
    ($($name:ident),+ $(,)?) => {
        // Self -> Self identity
        impl<$($name: Dim,)*> ArgInto<($($name,)*)> for ($($name,)*) {
            #[inline(always)]
            fn into_arg(self) -> ($($name,)*) {
                self
            }
        }
    };
}

impl_dim_tuple_arg_into!(D0);
impl_dim_tuple_arg_into!(D0, D1);
impl_dim_tuple_arg_into!(D0, D1, D2);
// 4-tuple is handled by the generic `impl<A, B, C, D> ArgInto<(TA, TB, TC, TD)>`
impl_dim_tuple_arg_into!(D0, D1, D2, D3, D4);
impl_dim_tuple_arg_into!(D0, D1, D2, D3, D4, D5);
impl_dim_tuple_arg_into!(D0, D1, D2, D3, D4, D5, D6);

// ============================================================================
// Fully-static shape construction from ()
// When all dims are ConstDim (e.g. Const<N>), no runtime arg is needed.
// ============================================================================

macro_rules! impl_const_dim_tuple_from_unit {
    ($($name:ident),+ $(,)?) => {
        // Handled by kindle_macros::impl_arg_into!(7)
    };
}

impl_const_dim_tuple_from_unit!(D0);
impl_const_dim_tuple_from_unit!(D0, D1);
impl_const_dim_tuple_from_unit!(D0, D1, D2);
impl_const_dim_tuple_from_unit!(D0, D1, D2, D3);
impl_const_dim_tuple_from_unit!(D0, D1, D2, D3, D4);
impl_const_dim_tuple_from_unit!(D0, D1, D2, D3, D4, D5);
impl_const_dim_tuple_from_unit!(D0, D1, D2, D3, D4, D5, D6);

// ============================================================================
// Partially-static shape conversions (via proc macro)
//
// Generates impls like:
//   impl<D0: Dim, const N1: usize> ArgInto<(usize, Const<N1>)> for (D0,)
//   impl<D0: Dim, const N1: usize> ArgInto<(usize, Const<N1>)> for D0
//   impl<const N1: usize, const N2: usize> ArgInto<(Const<N1>, Const<N2>)> for ()
//
// This allows users to only provide the dynamic dimensions when constructing
// partially-static shapes.
// ============================================================================

kindle_macros::impl_arg_into!(7);

// ============================================================================
// 4-tuple lifting: converts user args into (S::Arg, T::Arg, D::Arg, G::Arg)
//
// The Tensor constructor expects a 4-tuple of args, one per parameter.
// These impls "lift" smaller tuples into the full 4-tuple by filling
// unspecified positions with ().
// ============================================================================

// 0 values: fully static tensor, no args needed
impl ArgInto<((), (), (), ())> for () {
    #[inline(always)]
    fn into_arg(self) -> ((), (), (), ()) {
        ((), (), (), ())
    }
}

// 1 value: placed in whichever position has a non-() arg type
impl<A: ArgInto<B>, B: NotUnit> ArgInto<(B, (), (), ())> for A {
    #[inline(always)]
    fn into_arg(self) -> (B, (), (), ()) {
        (self.into_arg(), (), (), ())
    }
}
impl<A: ArgInto<B>, B: NotUnit> ArgInto<((), B, (), ())> for A {
    #[inline(always)]
    fn into_arg(self) -> ((), B, (), ()) {
        ((), self.into_arg(), (), ())
    }
}
impl<A: ArgInto<B>, B: NotUnit> ArgInto<((), (), B, ())> for A {
    #[inline(always)]
    fn into_arg(self) -> ((), (), B, ()) {
        ((), (), self.into_arg(), ())
    }
}
impl<A: ArgInto<B>, B: NotUnit> ArgInto<((), (), (), B)> for A {
    #[inline(always)]
    fn into_arg(self) -> ((), (), (), B) {
        ((), (), (), self.into_arg())
    }
}

// 2 values: C(4,2) = 6 combinations
impl<A, B, TA, TB> ArgInto<(TA, TB, (), ())> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    fn into_arg(self) -> (TA, TB, (), ()) {
        (self.0.into_arg(), self.1.into_arg(), (), ())
    }
}

impl<A, B, TA, TB> ArgInto<(TA, (), TB, ())> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    fn into_arg(self) -> (TA, (), TB, ()) {
        (self.0.into_arg(), (), self.1.into_arg(), ())
    }
}

impl<A, B, TA, TB> ArgInto<(TA, (), (), TB)> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    fn into_arg(self) -> (TA, (), (), TB) {
        (self.0.into_arg(), (), (), self.1.into_arg())
    }
}

impl<A, B, TA, TB> ArgInto<((), TA, TB, ())> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    fn into_arg(self) -> ((), TA, TB, ()) {
        ((), self.0.into_arg(), self.1.into_arg(), ())
    }
}

impl<A, B, TA, TB> ArgInto<((), TA, (), TB)> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    fn into_arg(self) -> ((), TA, (), TB) {
        ((), self.0.into_arg(), (), self.1.into_arg())
    }
}

impl<A, B, TA, TB> ArgInto<((), (), TA, TB)> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    fn into_arg(self) -> ((), (), TA, TB) {
        ((), (), self.0.into_arg(), self.1.into_arg())
    }
}

// 3 values: C(4,3) = 4 combinations
impl<A, B, C, TA, TB, TC> ArgInto<((), TA, TB, TC)> for (A, B, C)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    C: ArgInto<TC>,
    TA: NotUnit,
    TB: NotUnit,
    TC: NotUnit,
{
    fn into_arg(self) -> ((), TA, TB, TC) {
        ((), self.0.into_arg(), self.1.into_arg(), self.2.into_arg())
    }
}
impl<A, B, C, TA, TB, TC> ArgInto<(TA, (), TB, TC)> for (A, B, C)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    C: ArgInto<TC>,
    TA: NotUnit,
    TB: NotUnit,
    TC: NotUnit,
{
    fn into_arg(self) -> (TA, (), TB, TC) {
        (self.0.into_arg(), (), self.1.into_arg(), self.2.into_arg())
    }
}
impl<A, B, C, TA, TB, TC> ArgInto<(TA, TB, (), TC)> for (A, B, C)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    C: ArgInto<TC>,
    TA: NotUnit,
    TB: NotUnit,
    TC: NotUnit,
{
    fn into_arg(self) -> (TA, TB, (), TC) {
        (self.0.into_arg(), self.1.into_arg(), (), self.2.into_arg())
    }
}
impl<A, B, C, TA, TB, TC> ArgInto<(TA, TB, TC, ())> for (A, B, C)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    C: ArgInto<TC>,
    TA: NotUnit,
    TB: NotUnit,
    TC: NotUnit,
{
    fn into_arg(self) -> (TA, TB, TC, ()) {
        (self.0.into_arg(), self.1.into_arg(), self.2.into_arg(), ())
    }
}

// 4 values: all positions specified
impl<A, B, C, D, TA, TB, TC, TD> ArgInto<(TA, TB, TC, TD)> for (A, B, C, D)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    C: ArgInto<TC>,
    D: ArgInto<TD>,
    TA: NotUnit,
    TB: NotUnit,
    TC: NotUnit,
    TD: NotUnit,
{
    fn into_arg(self) -> (TA, TB, TC, TD) {
        (
            self.0.into_arg(),
            self.1.into_arg(),
            self.2.into_arg(),
            self.3.into_arg(),
        )
    }
}

// ============================================================================
// PhantomData pass-through
// ============================================================================

impl<T> ArgInto<PhantomData<T>> for PhantomData<T> {
    #[inline(always)]
    fn into_arg(self) -> PhantomData<T> {
        self
    }
}
