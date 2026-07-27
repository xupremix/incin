use core::ops::{Add, Div, Mul, Sub};
use typenum::{U1, U2, UInt, UTerm};

use super::error::{
    Axis, DimensionConstraint, OperationKind, RankExpectation, ShapeError,
};

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

impl<Kernel, Stride, Padding, Dilation> SpatialOut<Kernel, Stride, Padding, Dilation> for usize {
    /// Runtime dimension in, runtime dimension out.
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
///   arithmetic accident — an unchecked `/ stride` divides by zero and an
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

/// Rebuild a typed dimension from a computed extent, reporting a mismatch
/// rather than unwrapping.
///
/// [`Dim::from_size`] only rejects a size that disagrees with a
/// compile-time-fixed dimension, so on the failing path `D::default()` *is*
/// that fixed value and naming it makes the diagnostic actionable. Runtime
/// (`usize`), symbolic, and product dimensions accept every size and never
/// reach the error arm.
pub fn dim_from_size<D: Dim + Default>(
    operation: OperationKind,
    axis: Axis,
    size: usize,
) -> Result<D, ShapeError> {
    D::from_size(size).ok_or(ShapeError::DimensionMismatch {
        operation,
        axis,
        lhs: size,
        rhs: D::default().size(),
        constraint: DimensionConstraint::Equal,
    })
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
pub trait Pool2dShape<K, S, P, D>: crate::prelude::Shape {
    /// The pooled output shape (same batch/channel dims, spatial dims
    /// reduced via `SpatialOut`).
    type Output: crate::prelude::Shape + crate::prelude::DynShape;
    /// Computes the runtime `Field` of `Output` from the input's own field.
    fn compute_output_shape(
        input: &Self::Field,
    ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError>;
}

use crate::prelude::{Dim, Dyn};
use typenum::Unsigned;

// `$c`/`$h`/`$w` are the tuple indices of the channel and two spatial dims.
// Pooling preserves the channel axis (conv replaces it), so unlike
// `impl_conv2d_shape!` this needs the channel index as well.
macro_rules! impl_pool2d_shape {
    ($c:tt, $h:tt, $w:tt; $($B:ident : $idx:tt),*) => {
        impl<$($B: Dim,)* C: Dim, HIn: Dim, WIn: Dim, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned>
            Pool2dShape<K, S, P, D> for ($($B,)* C, HIn, WIn)
        where
            HIn: SpatialOut<K, S, P, D>,
            WIn: SpatialOut<K, S, P, D>,
            <HIn as SpatialOut<K, S, P, D>>::Output: Dim + Default,
            <WIn as SpatialOut<K, S, P, D>>::Output: Dim + Default,
        {
            /// The resulting shape per this trait\'s rule.
            type Output = (
                $($B,)*
                C,
                <HIn as SpatialOut<K, S, P, D>>::Output,
                <WIn as SpatialOut<K, S, P, D>>::Output,
            );
            /// Computes the runtime `Field` of `Output` from the input\'s own field.
            ///
            /// The spatial extents are computed from the input\'s own runtime
            /// sizes. They used to be `Default::default()`, which is the right
            /// value for a `typenum` extent (the type carries it) and **0** for
            /// a `usize` or symbolic one — so a pooled tensor with runtime
            /// spatial dims silently claimed a zero-sized output and propagated
            /// it. `SHP-005`.
            fn compute_output_shape(
                input: &Self::Field,
            ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError> {
                const OP: OperationKind = OperationKind::Pool2d;
                let height = spatial_out_size(
                    OP, Axis::Named("height"), input.$h.size(),
                    K::USIZE, S::USIZE, P::USIZE, D::USIZE,
                )?;
                let width = spatial_out_size(
                    OP, Axis::Named("width"), input.$w.size(),
                    K::USIZE, S::USIZE, P::USIZE, D::USIZE,
                )?;
                Ok((
                    $(input.$idx,)*
                    input.$c,
                    dim_from_size(OP, Axis::Named("height"), height)?,
                    dim_from_size(OP, Axis::Named("width"), width)?,
                ))
            }
        }
    };
}

// Shape is `(Batch.., C, HIn, WIn)`, so rank = batch count + 3.
incin_macros::rank_sweep!(pool2d => impl_pool2d_shape, min = 0, max = 5);

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> Pool2dShape<K, S, P, D> for Dyn {
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `Field` of `Output` from the input\'s own field.
    ///
    /// Rank 3 is `(C, H, W)` and rank 4 is `(B, C, H, W)`, as the trait's
    /// diagnostic already promised. The rank-3 case used to fall through the
    /// `len() == 4` test and return the *input* shape unpooled; any other rank
    /// did the same instead of reporting it.
    fn compute_output_shape(
        input: &Self::Field,
    ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError> {
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
pub trait SpatialConv1d<COut, K, S, P, D>: crate::prelude::Shape {
    /// The convolved output shape (batch dims unchanged, channel dim
    /// replaced by `COut`, length dim reduced via `SpatialOut`).
    type Output: crate::prelude::Shape + crate::prelude::DynShape;
    /// Computes the runtime `Field` of `Output` from the input's own field.
    fn compute_output_shape(
        input: &Self::Field,
        out_channels: usize,
    ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError>;
}

// `$len` is the tuple index of the length dim, i.e. one past the channel dim.
// It is passed explicitly rather than derived, because counting the batch
// parameters inside the macro needs an unstable metavariable expression.
macro_rules! impl_conv1d_shape {
    ($len:tt; $($B:ident : $idx:tt),*) => {
        impl<$($B: Dim,)* CIn: Dim, COut: Dim + Default, LIn: Dim, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned>
            SpatialConv1d<COut, K, S, P, D> for ($($B,)* CIn, LIn)
        where
            LIn: SpatialOut<K, S, P, D>,
            <LIn as SpatialOut<K, S, P, D>>::Output: Dim + Default,
        {
            /// The resulting shape per this trait\'s rule.
            type Output = ($($B,)* COut, <LIn as SpatialOut<K, S, P, D>>::Output);
            /// Computes the runtime `Field` of `Output` from the input\'s own field.
            ///
            /// Both trailing dimensions are computed rather than defaulted.
            /// `COut` is bounded by `Dim`, not `Unsigned`, so for a `usize` or
            /// symbolic channel count `Default::default()` is 0, not the
            /// `out_channels` this function is already handed; the same is
            /// true of the length dim. The channel conversion reports a
            /// mismatch instead of unwrapping. `SHP-005`.
            fn compute_output_shape(
                input: &Self::Field,
                out_channels: usize,
            ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError> {
                const OP: OperationKind = OperationKind::Conv1d;
                let length = spatial_out_size(
                    OP,
                    Axis::Named("length"),
                    input.$len.size(),
                    K::USIZE,
                    S::USIZE,
                    P::USIZE,
                    D::USIZE,
                )?;
                Ok((
                    $(input.$idx,)*
                    dim_from_size::<COut>(OP, Axis::Named("channels"), out_channels)?,
                    dim_from_size(OP, Axis::Named("length"), length)?,
                ))
            }
        }
    };
}

// Shape is `(Batch.., CIn, LIn)`, so rank = batch count + 2; `max = 6`
// reaches `MAX_RANK`. Conv is rank-preserving, so 8 is its correct ceiling.
incin_macros::rank_sweep!(conv1d => impl_conv1d_shape, max = 6);

impl<COut, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> SpatialConv1d<COut, K, S, P, D>
    for Dyn
{
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `Field` of `Output` from the input\'s own field.
    ///
    /// Rank 2 is `(C, L)` and rank 3 is `(B, C, L)`. Rank 2 used to fall
    /// through the `len() == 3` test and return the input unchanged.
    fn compute_output_shape(
        input: &Self::Field,
        out_channels: usize,
    ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError> {
        const OP: OperationKind = OperationKind::Conv1d;
        let mut dims = input.clone();
        expect_rank(OP, [2, 3], dims.len())?;
        let length = dims.len() - 1;
        dims[length - 1] = out_channels;
        dims[length] = spatial_out_size(
            OP,
            Axis::Named("length"),
            dims[length],
            K::USIZE,
            S::USIZE,
            P::USIZE,
            D::USIZE,
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
pub trait SpatialConv2d<COut, K, S, P, D>: crate::prelude::Shape {
    /// The convolved output shape (batch dims unchanged, channel dim
    /// replaced by `COut`, spatial dims reduced via `SpatialOut`).
    type Output: crate::prelude::Shape + crate::prelude::DynShape;
    /// Computes the runtime `Field` of `Output` from the input's own field.
    fn compute_output_shape(
        input: &Self::Field,
        out_channels: usize,
    ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError>;
}

// `$h`/`$w` are the tuple indices of the two spatial dims. See
// `impl_conv1d_shape!` for why they are passed rather than counted.
macro_rules! impl_conv2d_shape {
    ($h:tt, $w:tt; $($B:ident : $idx:tt),*) => {
        impl<$($B: Dim,)* CIn: Dim, COut: Dim + Default, HIn: Dim, WIn: Dim, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned>
            SpatialConv2d<COut, K, S, P, D> for ($($B,)* CIn, HIn, WIn)
        where
            HIn: SpatialOut<K, S, P, D>,
            WIn: SpatialOut<K, S, P, D>,
            <HIn as SpatialOut<K, S, P, D>>::Output: Dim + Default,
            <WIn as SpatialOut<K, S, P, D>>::Output: Dim + Default,
        {
            /// The resulting shape per this trait\'s rule.
            type Output = ($($B,)* COut, <HIn as SpatialOut<K, S, P, D>>::Output, <WIn as SpatialOut<K, S, P, D>>::Output);
            /// Computes the runtime `Field` of `Output` from the input\'s own field.
            ///
            /// See `impl_conv1d_shape!`: every trailing dimension is computed
            /// from the input rather than defaulted, and the channel
            /// conversion reports a mismatch instead of unwrapping.
            fn compute_output_shape(
                input: &Self::Field,
                out_channels: usize,
            ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError> {
                const OP: OperationKind = OperationKind::Conv2d;
                let height = spatial_out_size(
                    OP, Axis::Named("height"), input.$h.size(),
                    K::USIZE, S::USIZE, P::USIZE, D::USIZE,
                )?;
                let width = spatial_out_size(
                    OP, Axis::Named("width"), input.$w.size(),
                    K::USIZE, S::USIZE, P::USIZE, D::USIZE,
                )?;
                Ok((
                    $(input.$idx,)*
                    dim_from_size::<COut>(OP, Axis::Named("channels"), out_channels)?,
                    dim_from_size(OP, Axis::Named("height"), height)?,
                    dim_from_size(OP, Axis::Named("width"), width)?,
                ))
            }
        }
    };
}

// Shape is `(Batch.., CIn, HIn, WIn)`, so rank = batch count + 3.
incin_macros::rank_sweep!(conv2d => impl_conv2d_shape, max = 5);

impl<COut, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> SpatialConv2d<COut, K, S, P, D>
    for Dyn
{
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `Field` of `Output` from the input\'s own field.
    ///
    /// Rank 3 is `(C, H, W)` and rank 4 is `(B, C, H, W)`. Rank 3 used to fall
    /// through the `len() == 4` test and return the input unchanged.
    fn compute_output_shape(
        input: &Self::Field,
        out_channels: usize,
    ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError> {
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
    message = "Cannot apply adaptive 2D pooling to shape `{Self}`",
    label = "Invalid shape for adaptive 2D pooling",
    note = "AdaptiveAvgPool2D requires a 3D or 4D tensor (C, H, W) or (B, C, H, W)"
)]
/// Compile-time-checked `adaptive_avg_pool2d` output shape rule: the
/// output spatial size is exactly `(HOut, WOut)` regardless of input size.
pub trait AdaptiveAvgPool2dShape<HOut, WOut>: crate::prelude::Shape {
    /// The pooled output shape: batch/channel dims unchanged, spatial dims
    /// fixed to `(HOut, WOut)`.
    type Output: crate::prelude::Shape + crate::prelude::DynShape;
    /// Computes the runtime `Field` of `Output` from the input's own field.
    fn compute_output_shape(
        input: &Self::Field,
    ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError>;
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
macro_rules! impl_adaptive_pool2d_shape {
    ($c:tt, $h:tt, $w:tt; $($B:ident : $idx:tt),*) => {
        impl<$($B: Dim,)* C: Dim, HIn: Dim, WIn: Dim, HOut: Unsigned, WOut: Unsigned>
            AdaptiveAvgPool2dShape<HOut, WOut> for ($($B,)* C, HIn, WIn)
        where
            HOut: Dim + Default,
            WOut: Dim + Default,
        {
            /// The resulting shape per this trait\'s rule.
            type Output = ($($B,)* C, HOut, WOut);
            /// Computes the runtime `Field` of `Output` from the input\'s own field.
            ///
            /// `HOut`/`WOut` are `Unsigned`, so unlike the other rules here
            /// their `Default` really is the requested extent. The check that
            /// remains is that neither is 0.
            fn compute_output_shape(
                input: &Self::Field,
            ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError> {
                adaptive_extent(Axis::Named("height"), HOut::USIZE)?;
                adaptive_extent(Axis::Named("width"), WOut::USIZE)?;
                Ok(($(input.$idx,)* input.$c, HOut::default(), WOut::default()))
            }
        }
    };
}

// Shape is `(Batch.., C, HIn, WIn)`, so rank = batch count + 3.
incin_macros::rank_sweep!(pool2d => impl_adaptive_pool2d_shape, min = 0, max = 5);

impl<HOut: Unsigned, WOut: Unsigned> AdaptiveAvgPool2dShape<HOut, WOut> for Dyn {
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `Field` of `Output` from the input\'s own field.
    ///
    /// Rank 3 is `(C, H, W)` and rank 4 is `(B, C, H, W)`. Rank 3 used to fall
    /// through the `len() == 4` test and return the input unchanged.
    fn compute_output_shape(
        input: &Self::Field,
    ) -> Result<<Self::Output as crate::prelude::Shape>::Field, ShapeError> {
        let mut dims = input.clone();
        expect_rank(OperationKind::AdaptiveAvgPool2d, [3, 4], dims.len())?;
        let first_spatial = dims.len() - 2;
        dims[first_spatial] = adaptive_extent(Axis::Named("height"), HOut::USIZE)?;
        dims[first_spatial + 1] = adaptive_extent(Axis::Named("width"), WOut::USIZE)?;
        Ok(dims)
    }
}
