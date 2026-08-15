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
use crate::tensor::device::{Cuda, CudaN};

#[cfg(feature = "wgpu")]
use crate::tensor::device::{Wgpu, WgpuN};

use crate::shapes::{Dim, ShapeBuf};
use crate::tensor::device::{Cpu, DeviceId};
use crate::tensor::dtype::{DTypeDescriptor, DTypeId};
use crate::tensor::grad::{Grad, NoGrad};
use typenum::{Bit, UInt, UTerm, Unsigned};

use alloc::vec::Vec;

/// Trait for converting user-provided arguments into the internal
/// representation expected by tensor parameters.
pub trait ArgInto<Target> {
    /// Converts `self` into `Target`.
    fn into_arg(self) -> Target;
}

/// Converts an exact compressed allocating-layer argument list.
pub trait LayerArgInto<Target> {
    /// Restores omitted static positions in declaration order.
    fn into_layer_arg(self) -> Target;
}

incin_macros::impl_layer_args!(9);

#[derive(Debug, Clone)]
/// The four tensor-construction parameters (shape, dtype, device, grad
/// tracking) after `ArgInto`/`LayerArgInto` has resolved user-friendly
/// arguments into their internal `Field` representations. `S`/`T`/`D`/`G`
/// are each either `()` (not specified, static/default) or the concrete
/// resolved type.
pub struct TensorArgsData<S, T, D, G> {
    /// The resolved shape argument.
    pub(crate) shape: S,
    /// The resolved dtype argument.
    pub(crate) dtype: T,
    /// The resolved device argument.
    pub(crate) device: D,
    /// The resolved grad-tracking argument.
    pub(crate) grad: G,
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
                /// Converts `self` into the target representation.
                fn into_arg(self) -> $t {
                    self
                }
            }
        )*
    };
}

impl_self_arginto! {
    f32
    f64
    usize
    bool
    DTypeId
    DTypeDescriptor
    DeviceId
    Cpu
    Grad
    NoGrad
    ShapeBuf
}

pub trait UnitTree: Default + Copy + Clone + core::fmt::Debug {}
impl UnitTree for () {}
impl<T: UnitTree> UnitTree for ((), T) {}
impl UnitTree for ((),) {}
impl UnitTree for ((), (), ()) {}
impl UnitTree for ((), (), (), ()) {}
impl UnitTree for ((), (), (), (), ()) {}
impl UnitTree for ((), (), (), (), (), ()) {}
impl UnitTree for ((), (), (), (), (), (), ()) {}
impl UnitTree for ((), (), (), (), (), (), (), ()) {}

impl<T: UnitTree> ArgInto<T> for () {
    #[inline(always)]
    fn into_arg(self) -> T {
        T::default()
    }
}

impl ArgInto<DTypeDescriptor> for DTypeId {
    #[inline(always)]
    fn into_arg(self) -> DTypeDescriptor {
        self.descriptor()
    }
}

impl ArgInto<UTerm> for UTerm {
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> UTerm {
        self
    }
}

impl<U, B> ArgInto<UInt<U, B>> for UInt<U, B>
where
    U: Unsigned + Dim,
    B: Bit + Default + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq + 'static,
    UInt<U, B>: Unsigned
        + Default
        + Copy
        + Clone
        + core::fmt::Debug
        + Send
        + Sync
        + Eq
        + PartialEq
        + 'static,
{
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> UInt<U, B> {
        self
    }
}

// Tier 2: Cuda / Wgpu (partial compile-time, runtime usize ordinal)
#[cfg(feature = "cuda")]
impl ArgInto<Cuda> for Cuda {
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> Cuda {
        self
    }
}

#[cfg(feature = "wgpu")]
impl ArgInto<Wgpu> for Wgpu {
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> Wgpu {
        self
    }
}

// Tier 3: CudaN<N> / WgpuN<N> (fully compile-time, typenum ordinal)
#[cfg(feature = "cuda")]
impl<N: Unsigned> ArgInto<CudaN<N>> for CudaN<N> {
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> CudaN<N> {
        self
    }
}

#[cfg(feature = "wgpu")]
impl<N: Unsigned> ArgInto<WgpuN<N>> for WgpuN<N> {
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> WgpuN<N> {
        self
    }
}

// ============================================================================
// Vec / Array / Slice conversions for dynamic shapes
// ============================================================================

impl<D: Dim> ArgInto<Vec<D>> for Vec<D> {
    #[inline(always)]
    /// Converts `self` into the target representation.
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

impl ArgInto<Vec<usize>> for &[usize] {
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

impl ArgInto<(usize, ())> for [usize; 1] {
    #[inline(always)]
    fn into_arg(self) -> (usize, ()) {
        (self[0], ())
    }
}

impl ArgInto<(usize, (usize, ()))> for [usize; 2] {
    #[inline(always)]
    fn into_arg(self) -> (usize, (usize, ())) {
        (self[0], (self[1], ()))
    }
}

impl ArgInto<(usize, (usize, (usize, ())))> for [usize; 3] {
    #[inline(always)]
    fn into_arg(self) -> (usize, (usize, (usize, ()))) {
        (self[0], (self[1], (self[2], ())))
    }
}

type Dim4Tuple = (usize, (usize, (usize, (usize, ()))));
impl ArgInto<Dim4Tuple> for [usize; 4] {
    #[inline(always)]
    fn into_arg(self) -> (usize, (usize, (usize, (usize, ())))) {
        (self[0], (self[1], (self[2], (self[3], ()))))
    }
}

impl<A, TA> ArgInto<(TA, ())> for (A,)
where
    A: ArgInto<TA>,
{
    #[inline(always)]
    fn into_arg(self) -> (TA, ()) {
        (self.0.into_arg(), ())
    }
}

impl<A, B, TA, TB> ArgInto<(TA, (TB, ()))> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
{
    #[inline(always)]
    fn into_arg(self) -> (TA, (TB, ())) {
        (self.0.into_arg(), (self.1.into_arg(), ()))
    }
}

impl<A, B, C, TA, TB, TC> ArgInto<(TA, (TB, (TC, ())))> for (A, B, C)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    C: ArgInto<TC>,
{
    #[inline(always)]
    fn into_arg(self) -> (TA, (TB, (TC, ()))) {
        (
            self.0.into_arg(),
            (self.1.into_arg(), (self.2.into_arg(), ())),
        )
    }
}

impl<A, B, C, D, TA, TB, TC, TD> ArgInto<(TA, (TB, (TC, (TD, ()))))> for (A, B, C, D)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    C: ArgInto<TC>,
    D: ArgInto<TD>,
{
    #[inline(always)]
    fn into_arg(self) -> (TA, (TB, (TC, (TD, ())))) {
        (
            self.0.into_arg(),
            (
                self.1.into_arg(),
                (self.2.into_arg(), (self.3.into_arg(), ())),
            ),
        )
    }
}

// Structural five-axis shapes use the canonical right-nested argument tree.
// Keep the exact-tree conversion available so callers can pass a ShapeBuf-like
// argument without first flattening it into a legacy tuple representation.
type Dim5Arg = ((), ((), ((), (usize, ((), ())))));

impl ArgInto<Dim5Arg> for Dim5Arg {
    #[inline(always)]
    fn into_arg(self) -> Dim5Arg {
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
    f32
    f64
    usize
    bool
    &usize
    Cpu
    Grad
    NoGrad
    DTypeId
    DTypeDescriptor
    DeviceId
    ShapeBuf
}

// () combinations used for static shapes should also be treated as Non-Unit
// from the perspective of TensorArgsData lifting (as they represent the shape argument).
impl<T1> NotUnit for (T1,) {}
impl<T1, T2> NotUnit for (T1, T2) {}
impl<T1, T2, T3> NotUnit for (T1, T2, T3) {}
impl<T1, T2, T3, T4> NotUnit for (T1, T2, T3, T4) {}
impl<T1, T2, T3, T4, T5> NotUnit for (T1, T2, T3, T4, T5) {}
impl<T1, T2, T3, T4, T5, T6> NotUnit for (T1, T2, T3, T4, T5, T6) {}
impl<T1, T2, T3, T4, T5, T6, T7> NotUnit for (T1, T2, T3, T4, T5, T6, T7) {}
impl<T1, T2, T3, T4, T5, T6, T7, T8> NotUnit for (T1, T2, T3, T4, T5, T6, T7, T8) {}

impl<T, const N: usize> NotUnit for [T; N] {}
impl<T, const N: usize> NotUnit for &[T; N] {}

impl NotUnit for UTerm {}
impl<U, B> NotUnit for UInt<U, B>
where
    U: Unsigned + Dim,
    B: Bit + Default + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq + 'static,
    UInt<U, B>: Unsigned
        + Default
        + Copy
        + Clone
        + core::fmt::Debug
        + Send
        + Sync
        + Eq
        + PartialEq
        + 'static,
{
}

// Tier 2: runtime ordinal
#[cfg(feature = "cuda")]
impl NotUnit for Cuda {}

#[cfg(feature = "wgpu")]
impl NotUnit for Wgpu {}

// Tier 3: typenum ordinal
#[cfg(feature = "cuda")]
impl<N: Unsigned> NotUnit for CudaN<N> {}

#[cfg(feature = "wgpu")]
impl<N: Unsigned> NotUnit for WgpuN<N> {}

impl<D: Dim> NotUnit for Vec<D> {}
// Handled in shape.rs

// ============================================================================
// Dim tuple self-conversions (identity for shape args)
// ============================================================================

macro_rules! impl_dim_tuple_arg_into {
    ($($name:ident),+ $(,)?) => {
        // Self -> Self identity
        impl<$($name,)*> ArgInto<($($name,)*)> for ($($name,)*) {
            #[inline(always)]
            /// Converts `self` into the target representation.
            fn into_arg(self) -> ($($name,)*) {
                self
            }
        }
    };
}

// 1-tuple and 2-tuple ArgInto are handled by the generic DimCons nesting impls above.
// impl_dim_tuple_arg_into!(D0);
// impl_dim_tuple_arg_into!(D0, D1);
impl_dim_tuple_arg_into!(D0, D1, D2);
// 4-tuple is handled by the generic `impl<A, B, C, D> ArgInto<(TA, TB, TC, TD)>`
impl_dim_tuple_arg_into!(D0, D1, D2, D3, D4);
impl_dim_tuple_arg_into!(D0, D1, D2, D3, D4, D5);
impl_dim_tuple_arg_into!(D0, D1, D2, D3, D4, D5, D6);
impl_dim_tuple_arg_into!(D0, D1, D2, D3, D4, D5, D6, D7);

// ============================================================================
// 4-tuple lifting: converts user args into TensorArgsData
// ============================================================================

// 0 values: fully static tensor, no args needed
impl ArgInto<TensorArgsData<(), (), (), ()>> for () {
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<(), (), (), ()> {
        TensorArgsData {
            shape: (),
            dtype: (),
            device: (),
            grad: (),
        }
    }
}

// 1 value: placed in whichever position has a non-() arg type
impl<A: ArgInto<B>, B: NotUnit> ArgInto<TensorArgsData<B, (), (), ()>> for A {
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<B, (), (), ()> {
        TensorArgsData {
            shape: self.into_arg(),
            dtype: (),
            device: (),
            grad: (),
        }
    }
}
impl<A: ArgInto<B>, B: NotUnit> ArgInto<TensorArgsData<(), B, (), ()>> for A {
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<(), B, (), ()> {
        TensorArgsData {
            shape: (),
            dtype: self.into_arg(),
            device: (),
            grad: (),
        }
    }
}
impl<A: ArgInto<B>, B: NotUnit> ArgInto<TensorArgsData<(), (), B, ()>> for A {
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<(), (), B, ()> {
        TensorArgsData {
            shape: (),
            dtype: (),
            device: self.into_arg(),
            grad: (),
        }
    }
}
impl<A: ArgInto<B>, B: NotUnit> ArgInto<TensorArgsData<(), (), (), B>> for A {
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<(), (), (), B> {
        TensorArgsData {
            shape: (),
            dtype: (),
            device: (),
            grad: self.into_arg(),
        }
    }
}

// 2 values: C(4,2) = 6 combinations
impl<A, B, TA, TB> ArgInto<TensorArgsData<TA, TB, (), ()>> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<TA, TB, (), ()> {
        TensorArgsData {
            shape: self.0.into_arg(),
            dtype: self.1.into_arg(),
            device: (),
            grad: (),
        }
    }
}

impl<A, B, TA, TB> ArgInto<TensorArgsData<TA, (), TB, ()>> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<TA, (), TB, ()> {
        TensorArgsData {
            shape: self.0.into_arg(),
            dtype: (),
            device: self.1.into_arg(),
            grad: (),
        }
    }
}

impl<A, B, TA, TB> ArgInto<TensorArgsData<TA, (), (), TB>> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<TA, (), (), TB> {
        TensorArgsData {
            shape: self.0.into_arg(),
            dtype: (),
            device: (),
            grad: self.1.into_arg(),
        }
    }
}

impl<A, B, TA, TB> ArgInto<TensorArgsData<(), TA, TB, ()>> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<(), TA, TB, ()> {
        TensorArgsData {
            shape: (),
            dtype: self.0.into_arg(),
            device: self.1.into_arg(),
            grad: (),
        }
    }
}

impl<A, B, TA, TB> ArgInto<TensorArgsData<(), TA, (), TB>> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<(), TA, (), TB> {
        TensorArgsData {
            shape: (),
            dtype: self.0.into_arg(),
            device: (),
            grad: self.1.into_arg(),
        }
    }
}

impl<A, B, TA, TB> ArgInto<TensorArgsData<(), (), TA, TB>> for (A, B)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    TA: NotUnit,
    TB: NotUnit,
{
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<(), (), TA, TB> {
        TensorArgsData {
            shape: (),
            dtype: (),
            device: self.0.into_arg(),
            grad: self.1.into_arg(),
        }
    }
}

// 3 values: C(4,3) = 4 combinations
impl<A, B, C, TA, TB, TC> ArgInto<TensorArgsData<(), TA, TB, TC>> for (A, B, C)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    C: ArgInto<TC>,
    TA: NotUnit,
    TB: NotUnit,
    TC: NotUnit,
{
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<(), TA, TB, TC> {
        TensorArgsData {
            shape: (),
            dtype: self.0.into_arg(),
            device: self.1.into_arg(),
            grad: self.2.into_arg(),
        }
    }
}
impl<A, B, C, TA, TB, TC> ArgInto<TensorArgsData<TA, (), TB, TC>> for (A, B, C)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    C: ArgInto<TC>,
    TA: NotUnit,
    TB: NotUnit,
    TC: NotUnit,
{
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<TA, (), TB, TC> {
        TensorArgsData {
            shape: self.0.into_arg(),
            dtype: (),
            device: self.1.into_arg(),
            grad: self.2.into_arg(),
        }
    }
}
impl<A, B, C, TA, TB, TC> ArgInto<TensorArgsData<TA, TB, (), TC>> for (A, B, C)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    C: ArgInto<TC>,
    TA: NotUnit,
    TB: NotUnit,
    TC: NotUnit,
{
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<TA, TB, (), TC> {
        TensorArgsData {
            shape: self.0.into_arg(),
            dtype: self.1.into_arg(),
            device: (),
            grad: self.2.into_arg(),
        }
    }
}
impl<A, B, C, TA, TB, TC> ArgInto<TensorArgsData<TA, TB, TC, ()>> for (A, B, C)
where
    A: ArgInto<TA>,
    B: ArgInto<TB>,
    C: ArgInto<TC>,
    TA: NotUnit,
    TB: NotUnit,
    TC: NotUnit,
{
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<TA, TB, TC, ()> {
        TensorArgsData {
            shape: self.0.into_arg(),
            dtype: self.1.into_arg(),
            device: self.2.into_arg(),
            grad: (),
        }
    }
}

// 4 values: all positions specified
impl<A, B, C, D, TA, TB, TC, TD> ArgInto<TensorArgsData<TA, TB, TC, TD>> for (A, B, C, D)
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
    /// Converts `self` into the target representation.
    fn into_arg(self) -> TensorArgsData<TA, TB, TC, TD> {
        TensorArgsData {
            shape: self.0.into_arg(),
            dtype: self.1.into_arg(),
            device: self.2.into_arg(),
            grad: self.3.into_arg(),
        }
    }
}

// ============================================================================
// PhantomData pass-through
// ============================================================================

impl<T> ArgInto<PhantomData<T>> for PhantomData<T> {
    #[inline(always)]
    /// Converts `self` into the target representation.
    fn into_arg(self) -> PhantomData<T> {
        self
    }
}
