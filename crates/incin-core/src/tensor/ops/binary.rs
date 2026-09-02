//! Binary tensor operations with static and dynamic shape checking.
//!
//! This module provides explicit exact-shape element-wise operations
//! (`add_exact`, `sub_exact`, `mul_exact`, `div_exact`) and broadcasting
//! operations (`try_add`, `try_sub`, `try_mul`, `try_div`).
//!
//! It also provides broadcasting variants (`broadcast_add`, etc.) and implements standard
//! `core::ops` traits (like `core::ops::Add`) which automatically leverage compile-time
//! broadcast shape resolution (`BroadcastShape`).

use crate::dist::placement::Local;
use crate::err::Result;
use crate::exec::capability::Capabilities;
use crate::exec::catalog::{AttributeContract, CanonicalOperation, NoAttributes, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::ShapeValue;
use crate::shapes::{DynShape, Shape};
use crate::tensor::backend::Backend;
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::dtype::DType;
use crate::tensor::grad::NoGrad;
use crate::tensor::grad::{GradJoin, JoinedGrad, RequiresGrad};
use crate::tensor::ops::*;

pub(crate) fn execute_binary_descriptor<
    O,
    S: Shape,
    S2: Shape,
    B: Backend,
    KIn: DType,
    KOut: DType,
    G1: RequiresGrad,
    G2: RequiresGrad,
    GOut: RequiresGrad,
>(
    lhs: &Tensor<S, B, KIn, G1>,
    rhs: &Tensor<S2, B, KIn, G2>,
    grad_out: GOut::Field,
) -> Result<Tensor<S, B, KOut, GOut>>
where
    O: CanonicalOperation + crate::exec::catalog::Operation<Attributes = NoAttributes>,
    S: ShapeEq<S2>,
    B: Execute<O>,
    <B as Execute<O>>::Output: Into<B::Storage<KOut>>,
{
    <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
    let h_lhs = TensorHandle::from_storage::<B, KIn, Local>(&lhs.inner);
    let h_rhs = TensorHandle::from_storage::<B, KIn, Local>(&rhs.inner);
    let shape_val = lhs._shape.clone();
    let context = crate::tensor::grad::execution_context::<B, GOut>(&grad_out);
    let storage =
        dispatch::execute_shaped::<O, B, S>(&context, NoAttributes, &[h_lhs, h_rhs], &shape_val)
            .map_err(crate::err::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        lhs._shape.clone(),
        Default::default(),
        lhs._device.clone(),
        grad_out,
    )
}

pub(crate) fn execute_binary_descriptor_with_attributes<
    O,
    S: Shape,
    S2: Shape,
    B: Backend,
    KIn: DType,
    KOut: DType,
    GOut: RequiresGrad,
>(
    lhs: &Tensor<S, B, KIn, impl RequiresGrad>,
    rhs: &Tensor<S2, B, KIn, impl RequiresGrad>,
    attributes: O::Attributes,
    grad_out: GOut::Field,
) -> Result<Tensor<S, B, KOut, GOut>>
where
    O: CanonicalOperation,
    O::Attributes: AttributeContract,
    S: ShapeEq<S2>,
    B: Execute<O>,
    <B as Execute<O>>::Output: Into<B::Storage<KOut>>,
{
    <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
    let h_lhs = TensorHandle::from_storage::<B, KIn, Local>(&lhs.inner);
    let h_rhs = TensorHandle::from_storage::<B, KIn, Local>(&rhs.inner);
    let shape_val = lhs._shape.clone();
    let context = crate::tensor::grad::execution_context::<B, GOut>(&grad_out);
    let storage =
        dispatch::execute_shaped::<O, B, S>(&context, attributes, &[h_lhs, h_rhs], &shape_val)
            .map_err(crate::err::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        lhs._shape.clone(),
        Default::default(),
        lhs._device.clone(),
        grad_out,
    )
}

pub(crate) fn execute_broadcast_binary_descriptor<
    O,
    S1: Shape + DynShape,
    S2: Shape + DynShape,
    SOut: Shape + DynShape,
    B: Backend,
    K: DType,
    G1: RequiresGrad,
    G2: RequiresGrad,
    GOut: RequiresGrad,
>(
    lhs: &Tensor<S1, B, K, G1>,
    rhs: &Tensor<S2, B, K, G2>,
    grad_out: GOut::Field,
) -> Result<Tensor<SOut, B, K, GOut>>
where
    O: CanonicalOperation + crate::exec::catalog::Operation<Attributes = NoAttributes>,
    S1: crate::shapes::broadcast::BroadcastShape<S2, Output = SOut>,
    B: Execute<O>,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    <SOut as Shape>::STATIC_VALID;
    let shape_val = if lhs.shape_buf().as_ref() == rhs.shape_buf().as_ref() {
        ShapeValue::<SOut>::try_new(lhs.shape_buf().clone()).map_err(crate::err::Error::Shape)?
    } else {
        let b_shape = <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::output_shape(
            lhs.shape_buf(),
            rhs.shape_buf(),
        )
        .map_err(crate::err::Error::Shape)?;
        ShapeValue::<SOut>::try_new(b_shape).map_err(crate::err::Error::Shape)?
    };
    let h_lhs = TensorHandle::from_storage::<B, K, Local>(&lhs.inner);
    let h_rhs = TensorHandle::from_storage::<B, K, Local>(&rhs.inner);
    let context = crate::tensor::grad::execution_context::<B, GOut>(&grad_out);
    let storage =
        dispatch::execute_shaped::<O, B, SOut>(&context, NoAttributes, &[h_lhs, h_rhs], &shape_val)
            .map_err(crate::err::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        shape_val,
        lhs._dtype.clone(),
        lhs._device.clone(),
        grad_out,
    )
}

pub(crate) fn execute_cmp_descriptor<
    O,
    S: Shape,
    S2: Shape,
    B: Backend,
    K: DType,
    G1: RequiresGrad,
    G2: RequiresGrad,
>(
    lhs: &Tensor<S, B, K, G1>,
    rhs: &Tensor<S2, B, K, G2>,
) -> Result<Tensor<S, B, bool, NoGrad>>
where
    O: CanonicalOperation + crate::exec::catalog::Operation<Attributes = NoAttributes>,
    S: ShapeEq<S2>,
    B: Execute<O>,
    <B as Execute<O>>::Output: Into<B::Storage<bool>>,
{
    <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
    let h_lhs = TensorHandle::from_storage::<B, K, Local>(&lhs.inner);
    let h_rhs = TensorHandle::from_storage::<B, K, Local>(&rhs.inner);
    let shape_val = lhs._shape.clone();
    let context =
        ExecutionContext::from_scope(B::default()).with_grad_mode(crate::exec::GradMode::Disabled);
    let storage =
        dispatch::execute_shaped::<O, B, S>(&context, NoAttributes, &[h_lhs, h_rhs], &shape_val)
            .map_err(crate::err::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        lhs._shape.clone(),
        Default::default(),
        lhs._device.clone(),
        Default::default(),
    )
}

pub(crate) fn execute_logical_binary_descriptor<
    O,
    S: Shape,
    S2: Shape,
    B: Backend,
    G1: RequiresGrad,
    G2: RequiresGrad,
>(
    lhs: &Tensor<S, B, bool, G1>,
    rhs: &Tensor<S2, B, bool, G2>,
) -> Result<Tensor<S, B, bool, NoGrad>>
where
    O: CanonicalOperation + crate::exec::catalog::Operation<Attributes = NoAttributes>,
    S: ShapeEq<S2>,
    B: Execute<O>,
    <B as Execute<O>>::Output: Into<B::Storage<bool>>,
{
    <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
    let h_lhs = TensorHandle::from_storage::<B, bool, Local>(&lhs.inner);
    let h_rhs = TensorHandle::from_storage::<B, bool, Local>(&rhs.inner);
    let shape_val = lhs._shape.clone();
    let context =
        ExecutionContext::from_scope(B::default()).with_grad_mode(crate::exec::GradMode::Disabled);
    let storage =
        dispatch::execute_shaped::<O, B, S>(&context, NoAttributes, &[h_lhs, h_rhs], &shape_val)
            .map_err(crate::err::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        lhs._shape.clone(),
        Default::default(),
        lhs._device.clone(),
        Default::default(),
    )
}

macro_rules! impl_exact_binary_op {
    (
        $(#[$meta:meta])*
        $op:ident, $method:ident, $op_marker:ident
    ) => {
        impl<S: Shape, B: Backend, K: DType, G1: RequiresGrad> Tensor<S, B, K, G1> {
            $(#[$meta])*
            pub fn $method<S2: Shape, G2: RequiresGrad>(
                &self,
                rhs: &Tensor<S2, B, K, G2>,
            ) -> Result<Tensor<S, B, K, JoinedGrad<G1, G2>>>
            where
                S: ShapeEq<S2>,
                G1: GradJoin<G2>,
                B: Execute<op::$op_marker>,
                <B as Execute<op::$op_marker>>::Output: Into<B::Storage<K>>,
            {
                let grad_out = <G1 as GradJoin<G2>>::join_field(&self._grad, &rhs._grad);
                JoinedGrad::<G1, G2>::grad_mode(&grad_out).restrict(|| {
                    execute_binary_descriptor::<op::$op_marker, S, S2, B, K, K, _, _, _>(self, rhs, grad_out)
                })
            }
        }
    };
}

impl_exact_binary_op!(
    /// Adds another tensor element-wise with exact shape equality.
    Add, add_exact, Add
);

impl_exact_binary_op!(
    /// Subtracts another tensor element-wise with exact shape equality.
    Sub, sub_exact, Sub
);

impl_exact_binary_op!(
    /// Multiplies by another tensor element-wise with exact shape equality.
    Mul, mul_exact, Mul
);

impl_exact_binary_op!(
    /// Divides by another tensor element-wise with exact shape equality.
    Div, div_exact, Div
);

macro_rules! impl_cmp_op {
    ($(#[$meta:meta])* $method:ident, $op:ident) => {
        impl<
            S: Shape,
            B: Backend + Capabilities + Default,
            K: DType,
            G: RequiresGrad,
        > Tensor<S, B, K, G> {
            $(#[$meta])*
            pub fn $method<S2: Shape, G2: RequiresGrad>(
                &self,
                rhs: &Tensor<S2, B, K, G2>,
            ) -> Result<Tensor<S, B, bool, NoGrad>>
            where
                S: ShapeEq<S2>,
                B: Execute<op::$op>,
                <B as Execute<op::$op>>::Output: Into<B::Storage<bool>>,
            {
                NoGrad::grad_mode(&Default::default()).restrict(|| {
                    execute_cmp_descriptor::<op::$op, S, S2, B, K, G, G2>(self, rhs)
                })
            }
        }
    };
}

impl_cmp_op!(
    /// Element-wise equality (`self == rhs`).
    eq, CmpEq
);
impl_cmp_op!(
    /// Element-wise inequality (`self != rhs`).
    ne, CmpNe
);
impl_cmp_op!(
    /// Element-wise less-than (`self < rhs`).
    lt, CmpLt
);
impl_cmp_op!(
    /// Element-wise less-than-or-equal (`self <= rhs`).
    le, CmpLe
);
impl_cmp_op!(
    /// Element-wise greater-than (`self > rhs`).
    gt, CmpGt
);
impl_cmp_op!(
    /// Element-wise greater-than-or-equal (`self >= rhs`).
    ge, CmpGe
);

impl<S: Shape, B: Backend + Capabilities + Default, G: RequiresGrad> Tensor<S, B, bool, G> {
    /// Element-wise logical AND.
    pub fn logical_and<S2: Shape, G2: RequiresGrad>(
        &self,
        rhs: &Tensor<S2, B, bool, G2>,
    ) -> Result<Tensor<S, B, bool, NoGrad>>
    where
        S: ShapeEq<S2>,
        B: Execute<op::LogicalAnd>,
        <B as Execute<op::LogicalAnd>>::Output: Into<B::Storage<bool>>,
    {
        NoGrad::grad_mode(&Default::default()).restrict(|| {
            execute_logical_binary_descriptor::<op::LogicalAnd, S, S2, B, G, G2>(self, rhs)
        })
    }

    /// Element-wise logical OR.
    pub fn logical_or<S2: Shape, G2: RequiresGrad>(
        &self,
        rhs: &Tensor<S2, B, bool, G2>,
    ) -> Result<Tensor<S, B, bool, NoGrad>>
    where
        S: ShapeEq<S2>,
        B: Execute<op::LogicalOr>,
        <B as Execute<op::LogicalOr>>::Output: Into<B::Storage<bool>>,
    {
        NoGrad::grad_mode(&Default::default()).restrict(|| {
            execute_logical_binary_descriptor::<op::LogicalOr, S, S2, B, G, G2>(self, rhs)
        })
    }
}

impl_exact_binary_op!(
    /// Element-wise maximum of two tensors.
    Maximum, maximum, Maximum
);
impl_exact_binary_op!(
    /// Element-wise minimum of two tensors.
    Minimum, minimum, Minimum
);
impl_exact_binary_op!(
    /// Element-wise absolute difference `|self - rhs|`.
    AbsDiff, abs_diff, AbsDiff
);

impl_exact_binary_op!(
    /// Element-wise 2-argument arctangent `atan2(self, rhs)`.
    Atan2, atan2, Atan2
);
impl_exact_binary_op!(
    /// Element-wise floating point remainder `self % rhs`.
    Fmod, fmod, Fmod
);
impl_exact_binary_op!(
    /// Element-wise IEEE remainder.
    Remainder, remainder, Remainder
);

impl<S: Shape, B: Backend + Capabilities + Default, K: DType, G: RequiresGrad> Tensor<S, B, K, G> {
    /// In-place addition: mutates `self` by adding `rhs` element-wise.
    pub fn add_<S2: Shape, G2: RequiresGrad>(&mut self, rhs: &Tensor<S2, B, K, G2>) -> Result<()>
    where
        S: ShapeEq<S2>,
        B: Execute<op::Add>,
        <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    {
        <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
        let res = execute_binary_descriptor::<op::Add, S, S2, B, K, K, G, G2, G>(
            self,
            rhs,
            self._grad.clone(),
        )?;
        self.inner = res.inner;
        Ok(())
    }

    /// In-place subtraction: mutates `self` by subtracting `rhs` element-wise.
    pub fn sub_<S2: Shape, G2: RequiresGrad>(&mut self, rhs: &Tensor<S2, B, K, G2>) -> Result<()>
    where
        S: ShapeEq<S2>,
        B: Execute<op::Sub>,
        <B as Execute<op::Sub>>::Output: Into<B::Storage<K>>,
    {
        <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
        let res = execute_binary_descriptor::<op::Sub, S, S2, B, K, K, G, G2, G>(
            self,
            rhs,
            self._grad.clone(),
        )?;
        self.inner = res.inner;
        Ok(())
    }

    /// In-place multiplication: mutates `self` by multiplying `rhs` element-wise.
    pub fn mul_<S2: Shape, G2: RequiresGrad>(&mut self, rhs: &Tensor<S2, B, K, G2>) -> Result<()>
    where
        S: ShapeEq<S2>,
        B: Execute<op::Mul>,
        <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    {
        <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
        let res = execute_binary_descriptor::<op::Mul, S, S2, B, K, K, G, G2, G>(
            self,
            rhs,
            self._grad.clone(),
        )?;
        self.inner = res.inner;
        Ok(())
    }

    /// In-place division: mutates `self` by dividing by `rhs` element-wise.
    pub fn div_<S2: Shape, G2: RequiresGrad>(&mut self, rhs: &Tensor<S2, B, K, G2>) -> Result<()>
    where
        S: ShapeEq<S2>,
        B: Execute<op::Div>,
        <B as Execute<op::Div>>::Output: Into<B::Storage<K>>,
    {
        <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
        let res = execute_binary_descriptor::<op::Div, S, S2, B, K, K, G, G2, G>(
            self,
            rhs,
            self._grad.clone(),
        )?;
        self.inner = res.inner;
        Ok(())
    }

    /// In-place zero: fills all elements with 0.0.
    pub fn zero_(&mut self) -> Result<()>
    where
        B: Execute<op::MulScalar>,
        <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
    {
        let res = self.mul_scalar(0.0)?;
        self.inner = res.inner;
        Ok(())
    }

    /// In-place fill: fills all elements with scalar `val`.
    pub fn fill_(&mut self, val: f64) -> Result<()>
    where
        B: Execute<op::MulScalar> + Execute<op::AddScalar>,
        <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::AddScalar>>::Output: Into<B::Storage<K>>,
    {
        let res = self.mul_scalar(0.0)?.add_scalar(val)?;
        self.inner = res.inner;
        Ok(())
    }

    /// Subtracts a scalar: `self - scalar`.
    pub fn sub_scalar(&self, val: f64) -> Result<Self>
    where
        B: Execute<op::SubScalar>,
        <B as Execute<op::SubScalar>>::Output: Into<B::Storage<K>>,
    {
        crate::tensor::ops::unary::execute_unary_descriptor_with_attributes::<
            op::SubScalar,
            S,
            B,
            K,
            G,
            crate::shapes::Unknown,
        >(self, crate::exec::catalog::ScalarAttributes { value: val })
    }

    /// Divides by a scalar: `self / scalar`.
    pub fn div_scalar(&self, val: f64) -> Result<Self>
    where
        B: Execute<op::DivScalar>,
        <B as Execute<op::DivScalar>>::Output: Into<B::Storage<K>>,
    {
        crate::tensor::ops::unary::execute_unary_descriptor_with_attributes::<
            op::DivScalar,
            S,
            B,
            K,
            G,
            crate::shapes::Unknown,
        >(self, crate::exec::catalog::ScalarAttributes { value: val })
    }

    /// Linear interpolation: `self + weight * (end - self)`.
    pub fn lerp<S2: Shape, G2: RequiresGrad>(
        &self,
        end: &Tensor<S2, B, K, G2>,
        weight: f64,
    ) -> Result<Self>
    where
        S: ShapeEq<S2>,
        B: Execute<op::Lerp>,
        <B as Execute<op::Lerp>>::Output: Into<B::Storage<K>>,
    {
        <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
        execute_binary_descriptor_with_attributes::<op::Lerp, S, S2, B, K, K, G>(
            self,
            end,
            crate::exec::catalog::LerpAttributes { weight },
            self._grad.clone(),
        )
    }
}

macro_rules! impl_broadcast_binary_op {
    (
        $(#[$meta:meta])*
        $try_method:ident, $method:ident, $op:ident
    ) => {
        impl<S1: Shape + DynShape, B: Backend, K: DType, G1: RequiresGrad> Tensor<S1, B, K, G1>
        {
            $(#[$meta])*
            #[inline]
            pub fn $try_method<S2, G2>(&self, rhs: &Tensor<S2, B, K, G2>) -> Result<Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, K, JoinedGrad<G1, G2>>>
            where
                S2: Shape + DynShape,
                G2: RequiresGrad,
                G1: GradJoin<G2>,
                S1: crate::shapes::broadcast::BroadcastShape<S2>,
                <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape + DynShape,
                B: Execute<op::$op>,
                <B as Execute<op::$op>>::Output: Into<B::Storage<K>>,
            {
                let grad_out = <G1 as GradJoin<G2>>::join_field(&self._grad, &rhs._grad);
                JoinedGrad::<G1, G2>::grad_mode(&grad_out).restrict(|| {
                    execute_broadcast_binary_descriptor::<op::$op, S1, S2, _, B, K, _, _, _>(self, rhs, grad_out)
                })
            }

            $(#[$meta])*
            #[inline]
            pub fn $method<S2, G2>(&self, rhs: &Tensor<S2, B, K, G2>) -> Result<Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, K, JoinedGrad<G1, G2>>>
            where
                S2: Shape + DynShape,
                G2: RequiresGrad,
                G1: GradJoin<G2>,
                S1: crate::shapes::broadcast::BroadcastShape<S2>,
                <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape + DynShape,
                B: Execute<op::$op>,
                <B as Execute<op::$op>>::Output: Into<B::Storage<K>>,
            {
                self.$try_method(rhs)
            }
        }
    };
}

impl_broadcast_binary_op!(
    /// Adds two tensors, broadcasting their shapes if necessary according to NumPy semantics.
    try_add, broadcast_add, Add
);

impl_broadcast_binary_op!(
    /// Subtracts the right tensor from the left tensor, broadcasting shapes if necessary.
    try_sub, broadcast_sub, Sub
);

impl_broadcast_binary_op!(
    /// Multiplies two tensors element-wise, broadcasting shapes if necessary.
    try_mul, broadcast_mul, Mul
);

impl_broadcast_binary_op!(
    /// Divides the left tensor by the right tensor, broadcasting shapes if necessary.
    try_div, broadcast_div, Div
);

macro_rules! impl_std_ops {
    ($trait:ident, $method:ident, $backend_method:ident, $op:ident, $operator:literal) => {
        impl<
            S1: Shape + DynShape,
            S2: Shape + DynShape,
            B: Backend + Execute<op::$op>,
            K: DType,
            G1: RequiresGrad,
            G2: RequiresGrad,
        > core::ops::$trait<Tensor<S2, B, K, G2>> for Tensor<S1, B, K, G1>
        where
            G1: GradJoin<G2>,
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape + DynShape,
            <B as Execute<op::$op>>::Output: Into<B::Storage<K>>,
        {
            type Output = Tensor<
                <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output,
                B,
                K,
                JoinedGrad<G1, G2>,
            >;
            fn $method(self, rhs: Tensor<S2, B, K, G2>) -> Self::Output {
                crate::tensor::ops::operator_or_panic($operator, self.$backend_method(&rhs))
            }
        }

        impl<
            'a,
            'b,
            S1: Shape + DynShape,
            S2: Shape + DynShape,
            B: Backend + Execute<op::$op>,
            K: DType,
            G1: RequiresGrad,
            G2: RequiresGrad,
        > core::ops::$trait<&'b Tensor<S2, B, K, G2>> for &'a Tensor<S1, B, K, G1>
        where
            G1: GradJoin<G2>,
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape + DynShape,
            <B as Execute<op::$op>>::Output: Into<B::Storage<K>>,
        {
            type Output = Tensor<
                <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output,
                B,
                K,
                JoinedGrad<G1, G2>,
            >;
            fn $method(self, rhs: &'b Tensor<S2, B, K, G2>) -> Self::Output {
                crate::tensor::ops::operator_or_panic($operator, self.$backend_method(rhs))
            }
        }

        impl<
            'a,
            S1: Shape + DynShape,
            S2: Shape + DynShape,
            B: Backend + Execute<op::$op>,
            K: DType,
            G1: RequiresGrad,
            G2: RequiresGrad,
        > core::ops::$trait<&'a Tensor<S2, B, K, G2>> for Tensor<S1, B, K, G1>
        where
            G1: GradJoin<G2>,
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape + DynShape,
            <B as Execute<op::$op>>::Output: Into<B::Storage<K>>,
        {
            type Output = Tensor<
                <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output,
                B,
                K,
                JoinedGrad<G1, G2>,
            >;
            fn $method(self, rhs: &'a Tensor<S2, B, K, G2>) -> Self::Output {
                crate::tensor::ops::operator_or_panic($operator, self.$backend_method(rhs))
            }
        }

        impl<
            'a,
            S1: Shape + DynShape,
            S2: Shape + DynShape,
            B: Backend + Execute<op::$op>,
            K: DType,
            G1: RequiresGrad,
            G2: RequiresGrad,
        > core::ops::$trait<Tensor<S2, B, K, G2>> for &'a Tensor<S1, B, K, G1>
        where
            G1: GradJoin<G2>,
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape + DynShape,
            <B as Execute<op::$op>>::Output: Into<B::Storage<K>>,
        {
            type Output = Tensor<
                <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output,
                B,
                K,
                JoinedGrad<G1, G2>,
            >;
            fn $method(self, rhs: Tensor<S2, B, K, G2>) -> Self::Output {
                crate::tensor::ops::operator_or_panic($operator, self.$backend_method(&rhs))
            }
        }
    };
}

impl_std_ops!(Add, add, broadcast_add, Add, "+");
impl_std_ops!(Sub, sub, broadcast_sub, Sub, "-");
impl_std_ops!(Mul, mul, broadcast_mul, Mul, "*");
impl_std_ops!(Div, div, broadcast_div, Div, "/");
