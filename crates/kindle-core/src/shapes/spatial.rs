use core::ops::{Add, Div, Mul, Sub};
use typenum::{U1, U2, UInt, UTerm};

pub trait SpatialOut<Kernel, Stride, Padding, Dilation> {
    type Output;
}

impl<Kernel, Stride, Padding, Dilation> SpatialOut<Kernel, Stride, Padding, Dilation> for UTerm
where
    Padding: Mul<U2>,
    Kernel: Sub<U1>,
    Dilation: Mul<<Kernel as Sub<U1>>::Output>,
    UTerm: Add<<Padding as Mul<U2>>::Output>,
    <UTerm as Add<<Padding as Mul<U2>>::Output>>::Output:
        Sub<<Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output>,
    <<UTerm as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output: Sub<U1>,
    <<<UTerm as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output: Div<Stride>,
    <<<<UTerm as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output as Div<Stride>>::Output: Add<U1>,
{
    type Output = <<<<<UTerm as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output as Div<Stride>>::Output as Add<U1>>::Output;
}

impl<U, B, Kernel, Stride, Padding, Dilation> SpatialOut<Kernel, Stride, Padding, Dilation>
    for UInt<U, B>
where
    Padding: Mul<U2>,
    Kernel: Sub<U1>,
    Dilation: Mul<<Kernel as Sub<U1>>::Output>,
    UInt<U, B>: Add<<Padding as Mul<U2>>::Output>,
    <UInt<U, B> as Add<<Padding as Mul<U2>>::Output>>::Output:
        Sub<<Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output>,
    <<UInt<U, B> as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output: Sub<U1>,
    <<<UInt<U, B> as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output: Div<Stride>,
    <<<<UInt<U, B> as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output as Div<Stride>>::Output: Add<U1>,
{
    type Output = <<<<<UInt<U, B> as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output as Div<Stride>>::Output as Add<U1>>::Output;
}

impl<Kernel, Stride, Padding, Dilation> SpatialOut<Kernel, Stride, Padding, Dilation> for usize {
    type Output = usize;
}

pub trait Pool2dShape<K, S>: crate::prelude::Shape {
    type Output: crate::prelude::Shape;
}

use crate::prelude::{Dim, Dyn};
use typenum::{U0, Unsigned};

// Implement for (B, C, H, W) -> (B, C, HOut, WOut)
impl<B: Dim, C: Dim, HIn: Dim, WIn: Dim, K: Unsigned, S: Unsigned> Pool2dShape<K, S>
    for (B, C, HIn, WIn)
where
    HIn: SpatialOut<K, S, U0, U1>,
    WIn: SpatialOut<K, S, U0, U1>,
    <HIn as SpatialOut<K, S, U0, U1>>::Output: Dim,
    <WIn as SpatialOut<K, S, U0, U1>>::Output: Dim,
{
    type Output = (
        B,
        C,
        <HIn as SpatialOut<K, S, U0, U1>>::Output,
        <WIn as SpatialOut<K, S, U0, U1>>::Output,
    );
}

impl<K: Unsigned, S: Unsigned> Pool2dShape<K, S> for Dyn {
    type Output = Dyn;
}

pub trait Conv1dShape<K, S, P, D>: crate::prelude::Shape {
    type Output: crate::prelude::Shape;
}

// Implement for (B, CIn, LIn) -> (B, COut, LOut)
impl<Batch: Dim, CIn: Dim, COut: Dim, LIn: Dim, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned>
    Conv1dShape<K, S, P, D> for (Batch, CIn, LIn, COut)
where
    LIn: SpatialOut<K, S, P, D>,
    <LIn as SpatialOut<K, S, P, D>>::Output: Dim,
{
    type Output = (Batch, COut, <LIn as SpatialOut<K, S, P, D>>::Output);
}

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> Conv1dShape<K, S, P, D> for Dyn {
    type Output = Dyn;
}

pub trait Conv2dShape<K, S, P, D>: crate::prelude::Shape {
    type Output: crate::prelude::Shape;
}

// Implement for (B, CIn, HIn, WIn) -> (B, COut, HOut, WOut)
impl<
    Batch: Dim,
    CIn: Dim,
    COut: Dim,
    HIn: Dim,
    WIn: Dim,
    K: Unsigned,
    S: Unsigned,
    P: Unsigned,
    D: Unsigned,
> Conv2dShape<K, S, P, D> for (Batch, CIn, HIn, WIn, COut)
where
    HIn: SpatialOut<K, S, P, D>,
    WIn: SpatialOut<K, S, P, D>,
    <HIn as SpatialOut<K, S, P, D>>::Output: Dim,
    <WIn as SpatialOut<K, S, P, D>>::Output: Dim,
{
    type Output = (
        Batch,
        COut,
        <HIn as SpatialOut<K, S, P, D>>::Output,
        <WIn as SpatialOut<K, S, P, D>>::Output,
    );
}

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> Conv2dShape<K, S, P, D> for Dyn {
    type Output = Dyn;
}
