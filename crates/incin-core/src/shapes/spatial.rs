use core::ops::{Add, Div, Mul, Sub};
use typenum::{U1, U2, UInt, UTerm};

use super::dim::Dim;
use super::error::{Axis, OperationKind, RankExpectation, ShapeError};
use crate::shapes::ShapeBuf;

/// Compile-time formula for a single spatial dimension's conv/pool
/// output size: `(in + 2*Padding - Dilation*(Kernel-1) - 1) / Stride + 1`.
pub trait SpatialOut<Kernel, Stride, Padding, Dilation> {
    /// The output size for this input dimension.
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
    /// The output size, computed via the conv/pool output-size formula.
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
    /// The output size, computed via the conv/pool output-size formula.
    type Output = <<<<<UInt<U, B> as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output as Div<Stride>>::Output as Add<U1>>::Output;
}

impl<const N: usize, Kernel, Stride, Padding, Dilation>
    SpatialOut<Kernel, Stride, Padding, Dilation> for crate::shapes::dim::ConstDim<N>
{
    type Output = usize;
}

impl<Kernel, Stride, Padding, Dilation> SpatialOut<Kernel, Stride, Padding, Dilation> for usize {
    type Output = usize;
}

/// The conv/pool output extent for one axis, as a named checked sequence.
///
/// The formula is
/// `(input + 2*padding - dilation*(kernel - 1) - 1) / stride + 1`.
/// It is evaluated one named term at a time rather than as a single
/// expression, because every way it can fail is a different diagnostic:
///
/// * a `stride`, `kernel`, or `dilation` of 0 is an invalid parameter, not an
///   arithmetic accident - an unchecked `/ stride` divides by zero and an
///   unchecked `kernel - 1` underflows;
/// * a kernel that does not fit its padded input makes the subtraction
///   underflow. In a release build that wraps to an enormous value, which then
///   divides down to a plausible-looking extent. It is reported as
///   [`ShapeError::EmptyOutput`] instead;
/// * each multiplication can overflow, and the caller needs to know which one.
///
/// `SHP-005`. Before this, the whole formula was one expression evaluated with
/// wrapping arithmetic, and the static path did not evaluate it at all.
pub fn spatial_out_size(
    operation: OperationKind,
    axis: Axis,
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize, ShapeError> {
    for (parameter, value) in [
        ("stride", stride),
        ("kernel", kernel),
        ("dilation", dilation),
    ] {
        if value == 0 {
            return Err(ShapeError::InvalidParameter {
                operation,
                parameter,
                value,
            });
        }
    }

    let both_pads = padding
        .checked_mul(2)
        .ok_or(ShapeError::ArithmeticOverflow {
            operation,
            expression: "2 * padding",
        })?;
    let padded = input
        .checked_add(both_pads)
        .ok_or(ShapeError::ArithmeticOverflow {
            operation,
            expression: "input + 2 * padding",
        })?;

    // The extent the kernel actually covers, dilation included. `kernel - 1`
    // cannot underflow: a zero kernel was rejected above.
    let dilated = dilation
        .checked_mul(kernel - 1)
        .ok_or(ShapeError::ArithmeticOverflow {
            operation,
            expression: "dilation * (kernel - 1)",
        })?;
    let extent = dilated
        .checked_add(1)
        .ok_or(ShapeError::ArithmeticOverflow {
            operation,
            expression: "dilation * (kernel - 1) + 1",
        })?;

    // Checked *before* subtracting, so the underflow never happens.
    if extent > padded {
        return Err(ShapeError::EmptyOutput { operation, axis });
    }

    Ok((padded - extent) / stride + 1)
}

/// Check that a dynamic operand has one of the ranks its operation accepts.
fn expect_rank(
    operation: OperationKind,
    accepted: [usize; 2],
    actual: usize,
) -> Result<(), ShapeError> {
    if accepted.contains(&actual) {
        return Ok(());
    }
    Err(ShapeError::RankMismatch {
        operation,
        expected: RankExpectation::Between {
            min: accepted[0],
            max: accepted[1],
        },
        actual,
    })
}

#[diagnostic::on_unimplemented(
    message = "Cannot apply 2D pooling to shape `{Self}`",
    label = "Invalid shape for 2D pooling",
    note = "Pool2D requires a 3D or 4D tensor (C, H, W) or (B, C, H, W)"
)]
/// Compile-time-checked 2D pooling (`max_pool2d`/`avg_pool2d`) output
/// shape rule, given kernel `K`, stride `S`, padding `P`, and dilation `D`.
pub trait Pool2dShape<K, S, P, D>: crate::shapes::Shape {
    /// The pooled output shape (same batch/channel dims, spatial dims
    /// reduced via `SpatialOut`).
    type Output: crate::shapes::Shape + crate::shapes::DynShape;
    /// Computes the runtime `ShapeBuf` of `Output` from the input buffer.
    fn compute_output_shape(input: &ShapeBuf) -> Result<ShapeBuf, ShapeError>;
}

use crate::shapes::{Dyn, DynShape, Shape};
use typenum::Unsigned;

// `$c`/`$h`/`$w` are the tuple indices of the channel and two spatial dims.
// Pooling preserves the channel axis (conv replaces it), so unlike
// `impl_conv2d_shape!` this needs the channel index as well.
// Shape is `(Batch.., C, HIn, WIn)`, so rank = batch count + 3.

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> Pool2dShape<K, S, P, D> for Dyn {
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `ShapeBuf` of `Output` from the input buffer.
    ///
    /// Rank 3 is `(C, H, W)` and rank 4 is `(B, C, H, W)`, as the trait's
    /// diagnostic already promised. The rank-3 case used to fall through the
    /// `len() == 4` test and return the *input* shape unpooled; any other rank
    /// did the same instead of reporting it.
    fn compute_output_shape(input: &ShapeBuf) -> Result<ShapeBuf, ShapeError> {
        let op = OperationKind::Pool2d;
        let mut dims = input.clone();
        expect_rank(op, [3, 4], dims.len())?;
        let first_spatial = dims.len() - 2;
        for (offset, axis) in [Axis::Named("height"), Axis::Named("width")]
            .into_iter()
            .enumerate()
        {
            let index = first_spatial + offset;
            dims[index] = spatial_out_size(
                op,
                axis,
                dims[index],
                K::USIZE,
                S::USIZE,
                P::USIZE,
                D::USIZE,
            )?;
        }
        Ok(dims)
    }
}

#[diagnostic::on_unimplemented(
    message = "Cannot apply 1D convolution to shape `{Self}`",
    label = "Invalid shape for 1D convolution",
    note = "Conv1D requires a 2D or 3D tensor (C, L) or (B, C, L)"
)]
/// Compile-time-checked `conv1d` output shape rule, given output
/// channels `COut`, kernel `K`, stride `S`, padding `P`, and dilation `D`.
pub trait SpatialConv1d<COut, K, S, P, D>: crate::shapes::Shape {
    /// The convolved output shape (batch dims unchanged, channel dim
    /// replaced by `COut`, length dim reduced via `SpatialOut`).
    type Output: crate::shapes::Shape + crate::shapes::DynShape;
    /// Computes the runtime `ShapeBuf` of `Output` from the input buffer.
    fn compute_output_shape(input: &ShapeBuf, out_channels: usize) -> Result<ShapeBuf, ShapeError>;
}

// `$len` is the tuple index of the length dim, i.e. one past the channel dim.
// It is passed explicitly rather than derived, because counting the batch
// parameters inside the macro needs an unstable metavariable expression.
// Shape is `(Batch.., CIn, LIn)`, so rank = batch count + 2; `max = 6`
// The structural rule below is rank-polymorphic; backend limits are checked
// separately from frontend shape representability.

impl<COut, K: Dim<Arg = ()>, S: Dim<Arg = ()>, P: Dim<Arg = ()>, D: Dim<Arg = ()>>
    SpatialConv1d<COut, K, S, P, D> for Dyn
{
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `ShapeBuf` of `Output` from the input buffer.
    ///
    /// Rank 2 is `(C, L)` and rank 3 is `(B, C, L)`. Rank 2 used to fall
    /// through the `len() == 3` test and return the input unchanged.
    fn compute_output_shape(input: &ShapeBuf, out_channels: usize) -> Result<ShapeBuf, ShapeError> {
        const OP: OperationKind = OperationKind::Conv1d;
        let mut dims = input.clone();
        expect_rank(OP, [2, 3], dims.len())?;
        let length = dims.len() - 1;
        dims[length - 1] = out_channels;
        dims[length] = spatial_out_size(
            OP,
            Axis::Named("length"),
            dims[length],
            K::static_size()?,
            S::static_size()?,
            P::static_size()?,
            D::static_size()?,
        )?;
        Ok(dims)
    }
}

#[diagnostic::on_unimplemented(
    message = "Cannot apply 2D convolution to shape `{Self}`",
    label = "Invalid shape for 2D convolution",
    note = "Conv2D requires a 3D or 4D tensor (C, H, W) or (B, C, H, W)"
)]
/// Compile-time-checked `conv2d` output shape rule, given output
/// channels `COut`, kernel `K`, stride `S`, padding `P`, and dilation `D`.
pub trait SpatialConv2d<COut, K, S, P, D>: crate::shapes::Shape {
    /// The convolved output shape (batch dims unchanged, channel dim
    /// replaced by `COut`, spatial dims reduced via `SpatialOut`).
    type Output: crate::shapes::Shape + crate::shapes::DynShape;
    /// Computes the runtime `ShapeBuf` of `Output` from the input buffer.
    fn compute_output_shape(input: &ShapeBuf, out_channels: usize) -> Result<ShapeBuf, ShapeError>;
}

// `$h`/`$w` are the tuple indices of the two spatial dims. See
// `impl_conv1d_shape!` for why they are passed rather than counted.

// Shape is `(Batch.., CIn, HIn, WIn)`, so rank = batch count + 3.

impl<COut, K: Dim<Arg = ()>, S: Dim<Arg = ()>, P: Dim<Arg = ()>, D: Dim<Arg = ()>>
    SpatialConv2d<COut, K, S, P, D> for Dyn
{
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `ShapeBuf` of `Output` from the input buffer.
    ///
    /// Rank 3 is `(C, H, W)` and rank 4 is `(B, C, H, W)`. Rank 3 used to fall
    /// through the `len() == 4` test and return the input unchanged.
    fn compute_output_shape(input: &ShapeBuf, out_channels: usize) -> Result<ShapeBuf, ShapeError> {
        const OP: OperationKind = OperationKind::Conv2d;
        let mut dims = input.clone();
        expect_rank(OP, [3, 4], dims.len())?;
        let first_spatial = dims.len() - 2;
        dims[first_spatial - 1] = out_channels;
        for (offset, axis) in [Axis::Named("height"), Axis::Named("width")]
            .into_iter()
            .enumerate()
        {
            let index = first_spatial + offset;
            dims[index] = spatial_out_size(
                OP,
                axis,
                dims[index],
                K::static_size()?,
                S::static_size()?,
                P::static_size()?,
                D::static_size()?,
            )?;
        }
        Ok(dims)
    }
}

#[diagnostic::on_unimplemented(
    message = "Cannot apply adaptive 2D pooling to shape `{Self}`",
    label = "Invalid shape for adaptive 2D pooling",
    note = "AdaptiveAvgPool2D requires a 3D or 4D tensor (C, H, W) or (B, C, H, W)"
)]
/// Compile-time-checked `adaptive_avg_pool2d` output shape rule: the
/// output spatial size is exactly `(HOut, WOut)` regardless of input size.
pub trait AdaptiveAvgPool2dShape<HOut, WOut>: crate::shapes::Shape {
    /// The pooled output shape: batch/channel dims unchanged, spatial dims
    /// fixed to `(HOut, WOut)`.
    type Output: crate::shapes::Shape + crate::shapes::DynShape;
    /// Computes the runtime `ShapeBuf` of `Output` from the input buffer.
    fn compute_output_shape(input: &ShapeBuf) -> Result<ShapeBuf, ShapeError>;
}

/// Reject an adaptive output extent of 0, which addresses no input elements.
fn adaptive_extent(axis: Axis, requested: usize) -> Result<usize, ShapeError> {
    if requested == 0 {
        return Err(ShapeError::EmptyOutput {
            operation: OperationKind::AdaptiveAvgPool2d,
            axis,
        });
    }
    Ok(requested)
}

// Same batch-variadic form as `impl_pool2d_shape!`; the spatial extents are
// caller-chosen rather than derived, so `$h`/`$w` are only read to be replaced.

// Shape is `(Batch.., C, HIn, WIn)`, so rank = batch count + 3.

impl<HOut: Unsigned, WOut: Unsigned> AdaptiveAvgPool2dShape<HOut, WOut> for Dyn {
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `ShapeBuf` of `Output` from the input buffer.
    ///
    /// Rank 3 is `(C, H, W)` and rank 4 is `(B, C, H, W)`. Rank 3 used to fall
    /// through the `len() == 4` test and return the input unchanged.
    fn compute_output_shape(input: &ShapeBuf) -> Result<ShapeBuf, ShapeError> {
        let mut dims = input.clone();
        expect_rank(OperationKind::AdaptiveAvgPool2d, [3, 4], dims.len())?;
        let first_spatial = dims.len() - 2;
        dims[first_spatial] = adaptive_extent(Axis::Named("height"), HOut::USIZE)?;
        dims[first_spatial + 1] = adaptive_extent(Axis::Named("width"), WOut::USIZE)?;
        Ok(dims)
    }
}

use crate::shapes::shape::{DimCons, Nil, SplitLast2, SplitLast3, StructuralConcatShape};

type Conv1dTail<C, L> = DimCons<C, DimCons<L, Nil>>;
type Conv2dTail<C, H, W> = DimCons<C, DimCons<H, DimCons<W, Nil>>>;
type Pool2dTail<C, H, W> = DimCons<C, DimCons<H, DimCons<W, Nil>>>;
type Conv1dOutput<S, C, K, Stride, Padding, Dilation> =
    <<S as SplitLast2>::Prefix as StructuralConcatShape<
        Conv1dTail<
            C,
            <<S as SplitLast2>::Last as SpatialOut<K, Stride, Padding, Dilation>>::Output,
        >,
    >>::Output;
type Pool2dOutput<S, K, Stride, Padding, Dilation> =
    <<S as SplitLast3>::Prefix as StructuralConcatShape<
        Pool2dTail<
            <S as SplitLast3>::ThirdLast,
            <<S as SplitLast3>::SecondLast as SpatialOut<K, Stride, Padding, Dilation>>::Output,
            <<S as SplitLast3>::Last as SpatialOut<K, Stride, Padding, Dilation>>::Output,
        >,
    >>::Output;
type Conv2dOutput<S, C, K, Stride, Padding, Dilation> =
    <<S as SplitLast3>::Prefix as StructuralConcatShape<
        Conv2dTail<
            C,
            <<S as SplitLast3>::SecondLast as SpatialOut<K, Stride, Padding, Dilation>>::Output,
            <<S as SplitLast3>::Last as SpatialOut<K, Stride, Padding, Dilation>>::Output,
        >,
    >>::Output;
type AdaptivePool2dOutput<S, H, W> = <<S as SplitLast3>::Prefix as StructuralConcatShape<
    Pool2dTail<<S as SplitLast3>::ThirdLast, H, W>,
>>::Output;

fn append_spatial_suffix(
    operation: OperationKind,
    input: &crate::shapes::ShapeBuf,
    suffix: &[usize],
) -> Result<crate::shapes::ShapeBuf, ShapeError> {
    if input.len() < suffix.len() {
        return Err(ShapeError::RankMismatch {
            operation,
            expected: RankExpectation::AtLeast(suffix.len()),
            actual: input.len(),
        });
    }
    let mut output = crate::shapes::ShapeBuf::from_slice(&input[..input.len() - suffix.len()]);
    for &dim in suffix {
        output.push(dim);
    }
    Ok(output)
}

impl<S: Shape + SplitLast3, HOut, WOut> AdaptiveAvgPool2dShape<HOut, WOut> for S
where
    HOut: Unsigned + Dim + Default,
    WOut: Unsigned + Dim + Default,
    S::Prefix: StructuralConcatShape<Pool2dTail<S::ThirdLast, HOut, WOut>>,
    AdaptivePool2dOutput<S, HOut, WOut>: Shape + DynShape,
{
    type Output = AdaptivePool2dOutput<S, HOut, WOut>;

    fn compute_output_shape(input: &ShapeBuf) -> Result<ShapeBuf, ShapeError> {
        if input.len() < 3 {
            return Err(ShapeError::RankMismatch {
                operation: OperationKind::AdaptiveAvgPool2d,
                expected: RankExpectation::AtLeast(3),
                actual: input.len(),
            });
        }
        let height = adaptive_extent(Axis::Named("height"), HOut::USIZE)?;
        let width = adaptive_extent(Axis::Named("width"), WOut::USIZE)?;
        append_spatial_suffix(
            OperationKind::AdaptiveAvgPool2d,
            input,
            &[input[input.len() - 3], height, width],
        )
    }
}

impl<S: Shape + SplitLast2, COut: Dim + Default, K, Stride, Padding, Dilation>
    SpatialConv1d<COut, K, Stride, Padding, Dilation> for S
where
    K: Dim<Arg = ()>,
    Stride: Dim<Arg = ()>,
    Padding: Dim<Arg = ()>,
    Dilation: Dim<Arg = ()>,
    S::Last: SpatialOut<K, Stride, Padding, Dilation>,
    <S::Last as SpatialOut<K, Stride, Padding, Dilation>>::Output: Dim + Default,
    S::Prefix: StructuralConcatShape<
        Conv1dTail<COut, <S::Last as SpatialOut<K, Stride, Padding, Dilation>>::Output>,
    >,
    Conv1dOutput<S, COut, K, Stride, Padding, Dilation>: Shape + DynShape,
{
    type Output = Conv1dOutput<S, COut, K, Stride, Padding, Dilation>;

    fn compute_output_shape(input: &ShapeBuf, out_channels: usize) -> Result<ShapeBuf, ShapeError> {
        const OP: OperationKind = OperationKind::Conv1d;
        let length = spatial_out_size(
            OP,
            Axis::Named("length"),
            *input.as_ref().last().ok_or(ShapeError::RankMismatch {
                operation: OP,
                expected: RankExpectation::AtLeast(2),
                actual: input.len(),
            })?,
            K::static_size()?,
            Stride::static_size()?,
            Padding::static_size()?,
            Dilation::static_size()?,
        )?;
        append_spatial_suffix(OP, input, &[out_channels, length])
    }
}

impl<S: Shape + SplitLast3, K, Stride, Padding, Dilation> Pool2dShape<K, Stride, Padding, Dilation>
    for S
where
    K: Unsigned + Dim,
    Stride: Unsigned + Dim,
    Padding: Unsigned + Dim,
    Dilation: Unsigned + Dim,
    S::SecondLast: SpatialOut<K, Stride, Padding, Dilation>,
    S::Last: SpatialOut<K, Stride, Padding, Dilation>,
    <S::SecondLast as SpatialOut<K, Stride, Padding, Dilation>>::Output: Dim + Default,
    <S::Last as SpatialOut<K, Stride, Padding, Dilation>>::Output: Dim + Default,
    S::Prefix: StructuralConcatShape<
        Pool2dTail<
            S::ThirdLast,
            <S::SecondLast as SpatialOut<K, Stride, Padding, Dilation>>::Output,
            <S::Last as SpatialOut<K, Stride, Padding, Dilation>>::Output,
        >,
    >,
    Pool2dOutput<S, K, Stride, Padding, Dilation>: Shape + DynShape,
{
    type Output = Pool2dOutput<S, K, Stride, Padding, Dilation>;

    fn compute_output_shape(input: &ShapeBuf) -> Result<ShapeBuf, ShapeError> {
        const OP: OperationKind = OperationKind::Pool2d;
        let dims = input.as_ref();
        if dims.len() < 3 {
            return Err(ShapeError::RankMismatch {
                operation: OP,
                expected: RankExpectation::AtLeast(3),
                actual: dims.len(),
            });
        }
        let height = spatial_out_size(
            OP,
            Axis::Named("height"),
            dims[dims.len() - 2],
            K::USIZE,
            Stride::USIZE,
            Padding::USIZE,
            Dilation::USIZE,
        )?;
        let width = spatial_out_size(
            OP,
            Axis::Named("width"),
            dims[dims.len() - 1],
            K::USIZE,
            Stride::USIZE,
            Padding::USIZE,
            Dilation::USIZE,
        )?;
        append_spatial_suffix(OP, input, &[dims[dims.len() - 3], height, width])
    }
}

impl<S: Shape + SplitLast3, COut: Dim + Default, K, Stride, Padding, Dilation>
    SpatialConv2d<COut, K, Stride, Padding, Dilation> for S
where
    K: Dim<Arg = ()>,
    Stride: Dim<Arg = ()>,
    Padding: Dim<Arg = ()>,
    Dilation: Dim<Arg = ()>,
    S::SecondLast: SpatialOut<K, Stride, Padding, Dilation>,
    S::Last: SpatialOut<K, Stride, Padding, Dilation>,
    <S::SecondLast as SpatialOut<K, Stride, Padding, Dilation>>::Output: Dim + Default,
    <S::Last as SpatialOut<K, Stride, Padding, Dilation>>::Output: Dim + Default,
    S::Prefix: StructuralConcatShape<
        Conv2dTail<
            COut,
            <S::SecondLast as SpatialOut<K, Stride, Padding, Dilation>>::Output,
            <S::Last as SpatialOut<K, Stride, Padding, Dilation>>::Output,
        >,
    >,
    Conv2dOutput<S, COut, K, Stride, Padding, Dilation>: Shape + DynShape,
{
    type Output = Conv2dOutput<S, COut, K, Stride, Padding, Dilation>;

    fn compute_output_shape(input: &ShapeBuf, out_channels: usize) -> Result<ShapeBuf, ShapeError> {
        const OP: OperationKind = OperationKind::Conv2d;
        let dims = input.as_ref();
        if dims.len() < 3 {
            return Err(ShapeError::RankMismatch {
                operation: OP,
                expected: RankExpectation::AtLeast(3),
                actual: dims.len(),
            });
        }
        let height = spatial_out_size(
            OP,
            Axis::Named("height"),
            dims[dims.len() - 2],
            K::static_size()?,
            Stride::static_size()?,
            Padding::static_size()?,
            Dilation::static_size()?,
        )?;
        let width = spatial_out_size(
            OP,
            Axis::Named("width"),
            dims[dims.len() - 1],
            K::static_size()?,
            Stride::static_size()?,
            Padding::static_size()?,
            Dilation::static_size()?,
        )?;
        append_spatial_suffix(OP, input, &[out_channels, height, width])
    }
}
