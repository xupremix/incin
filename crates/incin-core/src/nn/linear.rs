use crate::backend_authoring::TensorBackend;
use crate::err::{Error, Result};
use crate::exec::catalog::{Descriptor, op};
use crate::nn::module::Module;
use crate::nn::module::ShapeInfo;
use crate::nn::param::{Frozen, Param, TrainState, Trainable};
use crate::nn::stats::{ComputeStats, LayerStats};
use crate::shapes::error::OperationKind;
use crate::shapes::shape::shape_buf_from_dims;
use crate::shapes::{
    Dim, DimCons, Dyn, DynShape, Nil, ReplaceLastDim, Shape, ShapeError, ShapeValue,
};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::{GradJoin, JoinedGrad, RequiresGrad};
use alloc::string::String;

type LinearMatMulDescriptor = Descriptor<op::MatMulExact>;

/// A shape marker trait specifying the input and output features of a [`Linear`] layer.
///
/// The typical usage is to supply a 2-tuple `(InF, OutF)` where:
/// * `InF` — Number of input features (the last dimension of the input tensor).
/// * `OutF` — Number of output features.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// use incin_core::nn::linear::LinearShape;
/// use incin_macros::s;
/// // Static linear layer: 784 inputs → 256 outputs
/// type S = s![784, 256];
/// # fn assert_is_a_linear_shape<T: LinearShape>() {}
/// # assert_is_a_linear_shape::<S>();
/// ```
pub trait LinearShape: Shape + DynShape {
    /// The number of input features (last dimension of the input tensor).
    type InF: Dim;
    /// The number of output features (last dimension of the output tensor).
    type OutF: Dim;
    /// The shape argument type used to construct the weight tensor.
    type WeightArg: crate::tensor::arg_into::NotUnit;
    /// The shape argument type used to construct the bias tensor.
    type BiasArg: crate::tensor::arg_into::NotUnit;
    /// The static shape type of the weight parameter tensor.
    type WeightShape: Shape<Arg = Self::WeightArg> + DynShape;
    /// The static shape type of the bias parameter tensor.
    type BiasShape: Shape<Arg = Self::BiasArg> + DynShape;

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(
        target: (<Self::InF as Dim>::Arg, <Self::OutF as Dim>::Arg),
    ) -> core::result::Result<(usize, usize, Self::WeightArg, Self::BiasArg), ShapeError>;
}

impl<InF: Dim, OutF: Dim> LinearShape
    for crate::shapes::shape::DimCons<
        InF,
        crate::shapes::shape::DimCons<OutF, crate::shapes::shape::Nil>,
    >
{
    type InF = InF;
    type OutF = OutF;
    type WeightArg = (<OutF as Dim>::Arg, (<InF as Dim>::Arg, ()));
    type BiasArg = (<OutF as Dim>::Arg, ());
    type WeightShape = crate::shapes::shape::DimCons<
        OutF,
        crate::shapes::shape::DimCons<InF, crate::shapes::shape::Nil>,
    >;
    type BiasShape = crate::shapes::shape::DimCons<OutF, crate::shapes::shape::Nil>;

    #[inline]
    fn build_args(
        target: (<Self::InF as Dim>::Arg, <Self::OutF as Dim>::Arg),
    ) -> core::result::Result<(usize, usize, Self::WeightArg, Self::BiasArg), ShapeError> {
        let in_f = InF::resolve_arg(target.0.clone())?;
        let out_f = OutF::resolve_arg(target.1.clone())?;
        Ok((
            in_f,
            out_f,
            (target.1.clone(), (target.0, ())),
            (target.1, ()),
        ))
    }
}

impl LinearShape for Dyn {
    /// The number of input features (last dimension of the input tensor).
    type InF = usize;
    /// The number of output features (last dimension of the output tensor).
    type OutF = usize;
    /// The shape argument type used to construct the weight tensor.
    type WeightArg = alloc::vec::Vec<usize>;
    /// The shape argument type used to construct the bias tensor.
    type BiasArg = alloc::vec::Vec<usize>;
    /// The static shape type of the weight parameter tensor.
    type WeightShape = Dyn;
    /// The static shape type of the bias parameter tensor.
    type BiasShape = Dyn;

    #[inline]
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(
        target: (<Self::InF as Dim>::Arg, <Self::OutF as Dim>::Arg),
    ) -> core::result::Result<(usize, usize, Self::WeightArg, Self::BiasArg), ShapeError> {
        let in_f = target.0;
        let out_f = target.1;
        Ok((in_f, out_f, alloc::vec![out_f, in_f], alloc::vec![out_f]))
    }
}

/// A statically shaped linear layer whose output-feature axis can be split
/// evenly across the first two-rank tensor-parallel mesh.
///
/// Dynamic shapes use
/// [`TensorParallelPlanBuilder::push_column_dyn`](crate::dist::TensorParallelPlanBuilder::push_column_dyn)
/// and receive the equivalent check at runtime.
#[cfg(feature = "distributed")]
pub trait TwoWayColumnLinearShape: LinearShape {
    /// Output features held by each TP rank.
    const LOCAL_OUT_FEATURES: usize;
}

#[cfg(feature = "distributed")]
impl<InF, OutF> TwoWayColumnLinearShape for DimCons<InF, DimCons<OutF, Nil>>
where
    InF: Dim,
    OutF: Dim + crate::dist::ShardDivisible<typenum::U2>,
{
    const LOCAL_OUT_FEATURES: usize = <OutF as crate::dist::ShardDivisible<typenum::U2>>::LOCAL;
}

/// A statically shaped linear layer whose contraction/input-feature axis can
/// be split evenly across the first two-rank tensor-parallel mesh.
///
/// Dynamic shapes use
/// [`TensorParallelPlanBuilder::push_row_dyn`](crate::dist::TensorParallelPlanBuilder::push_row_dyn)
/// and receive the equivalent check at runtime.
#[cfg(feature = "distributed")]
pub trait TwoWayRowLinearShape: LinearShape {
    /// Input features held by each TP rank.
    const LOCAL_IN_FEATURES: usize;
}

#[cfg(feature = "distributed")]
impl<InF, OutF> TwoWayRowLinearShape for DimCons<InF, DimCons<OutF, Nil>>
where
    InF: Dim + crate::dist::ShardDivisible<typenum::U2>,
    OutF: Dim,
{
    const LOCAL_IN_FEATURES: usize = <InF as crate::dist::ShardDivisible<typenum::U2>>::LOCAL;
}

/// A fully connected (dense) linear layer: `y = x @ Wᵀ + b`.
///
/// `S` encodes both the input and output feature dimensions via [`LinearShape`]. The most common
/// form is `s![InF, OutF]`. For dynamic feature sizes use `Dyn` or mixed partial types.
///
/// ## Examples
///
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::tensor::device::Cpu>;
/// use incin::prelude::*;
///
/// // A fully static linear layer: 512 inputs → 256 outputs
/// let layer = Linear::<s![512, 256], DefaultBackend>::build(())?;
///
/// // A dynamic linear layer — shape known only at runtime
/// let layer = Linear::<Dyn, DefaultBackend>::build((512, 256))?;
/// # Ok(()) }
/// ```
/// A builder for constructing a [`Linear`] layer before target-based initialization.
///
/// Stores layer geometry ([`ShapeValue`]), weight initializer policy, and bias initializer policy.
/// Contains no backend, device, or target type parameters.
pub struct LinearBuilder<
    S: LinearShape,
    Bias: crate::nn::optional::OptionalField = crate::nn::optional::True,
    Train: TrainState = Trainable,
> {
    pub shape: ShapeValue<S>,
    pub weight_init: crate::nn::init::Init,
    pub bias_init: crate::nn::init::Init,
    pub _phantom: core::marker::PhantomData<(Bias, Train)>,
}

impl<S: LinearShape, Bias: crate::nn::optional::OptionalField, Train: TrainState>
    LinearBuilder<S, Bias, Train>
{
    /// Returns a reference to the shape specification of this builder.
    pub fn shape(&self) -> &ShapeValue<S> {
        &self.shape
    }

    /// Returns the weight initializer policy.
    pub fn weight_init_policy(&self) -> crate::nn::init::Init {
        self.weight_init
    }

    /// Returns the bias initializer policy.
    pub fn bias_init_policy(&self) -> crate::nn::init::Init {
        self.bias_init
    }

    /// Disables bias on the linear layer.
    pub fn no_bias(self) -> LinearBuilder<S, crate::nn::optional::False, Train> {
        LinearBuilder {
            shape: self.shape,
            weight_init: self.weight_init,
            bias_init: self.bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Marks the created linear layer parameters as frozen (non-trainable).
    pub fn frozen(self) -> LinearBuilder<S, Bias, Frozen> {
        LinearBuilder {
            shape: self.shape,
            weight_init: self.weight_init,
            bias_init: self.bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Sets the weight initializer policy.
    pub fn weight_init(mut self, init: crate::nn::init::Init) -> Self {
        self.weight_init = init;
        self
    }

    /// Sets the bias initializer policy.
    pub fn bias_init(mut self, init: crate::nn::init::Init) -> Self {
        self.bias_init = init;
        self
    }
}

/// Free constructor for a backend-independent [`LinearBuilder`].
pub fn linear<S: LinearShape>(shape: ShapeValue<S>) -> LinearBuilder<S> {
    LinearBuilder {
        shape,
        weight_init: crate::nn::init::kaiming_uniform(),
        bias_init: crate::nn::init::kaiming_uniform(),
        _phantom: core::marker::PhantomData,
    }
}

/// A fully connected (dense) linear layer: `y = x @ Wᵀ + b`.
#[derive(Debug, Clone)]
#[incin_macros::module(internal, no_stats)]
pub struct Linear<
    S: LinearShape,
    B: crate::tensor::backend::VariableBackend,
    Bias: crate::nn::optional::OptionalField = crate::nn::optional::True,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// The learnable weight matrix parameter.
    pub weight: Param<S::WeightShape, B, K, Train>,
    /// The optional learnable bias vector parameter.
    pub bias: Option<Param<S::BiasShape, B, K, Train>>,
    #[module(ignore)]
    _phantom: core::marker::PhantomData<(S, B, Bias, K, Train)>,
}

impl<
    S: LinearShape,
    B: crate::tensor::backend::VariableBackend,
    Bias: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> ShapeInfo for Linear<S, B, Bias, K, Train>
{
    fn shape_info(&self) -> Option<String> {
        None
    }
}

impl<
    S: LinearShape,
    B: crate::tensor::backend::VariableBackend,
    Bias: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> Linear<S, B, Bias, K, Train>
{
    /// Constructs a Linear layer from weight and bias parameter parts.
    pub fn from_raw_parts(
        weight: Param<S::WeightShape, B, K, Train>,
        bias: Option<Param<S::BiasShape, B, K, Train>>,
    ) -> Self {
        Self {
            weight,
            bias,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Converts this layer's parameters to frozen typestate.
    pub fn freeze(self) -> Linear<S, B, Bias, K, Frozen> {
        Linear {
            weight: self.weight.freeze(),
            bias: self.bias.map(|b| b.freeze()),
            _phantom: core::marker::PhantomData,
        }
    }

    /// Converts this layer's parameters to trainable typestate.
    pub fn unfreeze(self) -> Linear<S, B, Bias, K, Trainable> {
        Linear {
            weight: self.weight.unfreeze(),
            bias: self.bias.map(|b| b.unfreeze()),
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<
    S: LinearShape,
    B: crate::tensor::backend::VariableBackend,
    Bias: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> ComputeStats for Linear<S, B, Bias, K, Train>
{
    fn compute_stats(&self, batch: u64) -> LayerStats {
        let dims = self.weight.shape_dims();
        let (out_f, in_f) = (
            u64::try_from(dims[0]).expect("validated dimension must fit u64"),
            u64::try_from(dims[1]).expect("validated dimension must fit u64"),
        );
        let bias_params = self
            .bias
            .as_ref()
            .map(|b| crate::nn::stats::validated_parameter_count(&b.shape_dims()))
            .unwrap_or(0);
        LayerStats {
            params: out_f * in_f + bias_params,
            macs: in_f * out_f * batch,
        }
    }
}

impl<S, B, Bias, K: DType, Train: TrainState> Linear<S, B, Bias, K, Train>
where
    S: LinearShape,
    B: TensorBackend<K> + crate::nn::param::ParameterInit<K>,
    Bias: crate::nn::optional::OptionalField,
    <K as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
{
    /// Builds the layer from its exact compressed argument tuple.
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::InF as Dim>::Arg,
                <S::OutF as Dim>::Arg,
                <K as DType>::Arg,
                <B::Device as Device>::Arg,
                <Bias as crate::nn::optional::OptionalField>::Arg,
            )>,
    {
        let (in_arg, out_arg, dtype, device, bias) = args.into_layer_arg();
        Self::build_full(in_arg, out_arg, dtype, device, bias)
    }

    /// Builds the layer from every argument stated explicitly, in declaration order.
    pub fn build_full(
        in_arg: <S::InF as Dim>::Arg,
        out_arg: <S::OutF as Dim>::Arg,
        dtype: <K as DType>::Arg,
        device: <B::Device as Device>::Arg,
        bias_arg: <Bias as crate::nn::optional::OptionalField>::Arg,
    ) -> Result<Self> {
        let (in_f, _out_f, w_args, b_args) =
            S::build_args((in_arg, out_arg)).map_err(Error::Shape)?;
        let w_field = <S::WeightShape as Shape>::resolve(w_args).map_err(Error::Shape)?;
        let w_dims = w_field.clone();
        let init = crate::nn::init::kaiming_uniform();
        let context_w = crate::nn::init::InitContext::new(crate::nn::init::ParameterRole::Weight)
            .with_fan(in_f, _out_f);
        let plan_w = init.plan(context_w)?;
        let raw_w = crate::nn::param::execute_plan_raw::<B, K>(
            w_dims.as_ref(),
            &K::init(dtype.clone()),
            &<B::Device as Device>::init(device.clone()),
            plan_w,
        )?;
        let weight = Param::<S::WeightShape, B, K, Train>::from_parts_checked(
            raw_w,
            w_field,
            K::init(dtype.clone()),
            <B::Device as Device>::init(device.clone()),
        )?;

        let bias = if Bias::init(bias_arg) {
            let b_field = <S::BiasShape as Shape>::resolve(b_args).map_err(Error::Shape)?;
            let b_dims = b_field.clone();
            let context_b = crate::nn::init::InitContext::new(crate::nn::init::ParameterRole::Bias)
                .with_fan(in_f, _out_f);
            let plan_b = init.plan(context_b)?;
            let raw_b = crate::nn::param::execute_plan_raw::<B, K>(
                b_dims.as_ref(),
                &K::init(dtype.clone()),
                &<B::Device as Device>::init(device.clone()),
                plan_b,
            )?;
            Some(Param::<S::BiasShape, B, K, Train>::from_parts_checked(
                raw_b,
                b_field,
                K::init(dtype),
                <B::Device as Device>::init(device),
            )?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            _phantom: core::marker::PhantomData,
        })
    }
}

impl<
    B: crate::tensor::backend::VariableBackend
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + crate::exec::Capabilities
        + Execute<op::Add>,
    K: DType,
    Train: TrainState,
    G: RequiresGrad,
> Module<Tensor<Dyn, B, K, G>> for Linear<Dyn, B, crate::nn::optional::True, K, Train>
where
    G: GradJoin<Train::TensorGrad>,
    JoinedGrad<G, Train::TensorGrad>: GradJoin<Train::TensorGrad>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
{
    type Output = Tensor<Dyn, B, K, JoinedGrad<G, Train::TensorGrad>>;
    type Error = Error;

    fn forward(&self, x: Tensor<Dyn, B, K, G>) -> core::result::Result<Self::Output, Error> {
        let weight_t = self.weight.as_tensor()?.transpose_runtime(0, 1)?;
        let out = x.matmul(&weight_t)?;
        let bias_t = self.bias.as_ref().unwrap().as_tensor()?;
        let out_final = out.broadcast_add::<Dyn, Train::TensorGrad>(&bias_t)?;
        Tensor::from_shape_value(
            out_final.inner,
            out_final._shape,
            out_final._dtype,
            out_final._device,
            out._grad,
        )
    }
}

impl<
    B: crate::tensor::backend::VariableBackend
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + crate::exec::Capabilities,
    K: DType,
    Train: TrainState,
    G: RequiresGrad,
> Module<Tensor<Dyn, B, K, G>> for Linear<Dyn, B, crate::nn::optional::False, K, Train>
where
    G: GradJoin<Train::TensorGrad>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
{
    type Output = Tensor<Dyn, B, K, JoinedGrad<G, Train::TensorGrad>>;
    type Error = Error;

    fn forward(&self, x: Tensor<Dyn, B, K, G>) -> core::result::Result<Self::Output, Error> {
        let weight_t = self.weight.as_tensor()?.transpose_runtime(0, 1)?;
        x.matmul(&weight_t)
    }
}

impl<
    B: crate::tensor::backend::VariableBackend
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + crate::exec::Capabilities
        + Execute<op::Add>,
    K: DType,
    Train: TrainState,
> Module<Tensor<Dyn, B, K>> for Linear<Dyn, B, Dyn, K, Train>
where
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
{
    type Output = Tensor<Dyn, B, K>;
    type Error = Error;

    fn forward(&self, x: Tensor<Dyn, B, K>) -> core::result::Result<Tensor<Dyn, B, K>, Error> {
        let weight_t = self.weight.as_tensor()?.transpose_runtime(0, 1)?;
        let out = x.matmul(&weight_t)?;
        if let Some(b) = &self.bias {
            let bias_t = b.as_tensor()?;
            let out_final = out.broadcast_add(&bias_t)?;
            Tensor::from_shape_value(
                out_final.inner,
                out_final._shape,
                out_final._dtype,
                out_final._device,
                core::marker::PhantomData,
            )
        } else {
            Tensor::from_shape_value(
                out.inner,
                out._shape,
                out._dtype,
                out._device,
                core::marker::PhantomData,
            )
        }
    }
}

impl<
    InF: Dim,
    OutF: Dim,
    InShape: Shape + DynShape + ReplaceLastDim<OutF> + crate::shapes::EndsWith<InF>,
    B: crate::tensor::backend::VariableBackend
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + crate::exec::Capabilities
        + Execute<op::Add>,
    K: DType,
    Train: TrainState,
    G: RequiresGrad,
> Module<Tensor<InShape, B, K, G>>
    for Linear<DimCons<InF, DimCons<OutF, Nil>>, B, crate::nn::optional::True, K, Train>
where
    InShape::Output: DynShape,
    G: GradJoin<Train::TensorGrad>,
    JoinedGrad<G, Train::TensorGrad>: GradJoin<Train::TensorGrad>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
{
    type Output = Tensor<InShape::Output, B, K, JoinedGrad<G, Train::TensorGrad>>;
    type Error = Error;

    fn forward(&self, x: Tensor<InShape, B, K, G>) -> core::result::Result<Self::Output, Error> {
        let dtype = x._dtype.clone();
        let device = x._device.clone();

        let mut dims = x.shape_buf().as_ref().to_vec();
        let last_idx = dims.len().saturating_sub(1);
        if last_idx < dims.len() {
            dims[last_idx] = self.weight.shape_dims()[0];
        }
        let shape = shape_buf_from_dims::<InShape::Output>(OperationKind::MatMul, &dims)?;

        let weight_dyn = self.weight.as_tensor()?.into_shape::<Dyn>()?;
        let weight_t = weight_dyn.transpose_runtime(0, 1)?;
        let x_dyn = x.into_shape::<Dyn>()?;
        let out_dyn = x_dyn.matmul(&weight_t)?;

        let bias_dyn = self
            .bias
            .as_ref()
            .unwrap()
            .as_tensor()?
            .into_shape::<Dyn>()?;
        let out_final = out_dyn.broadcast_add::<Dyn, Train::TensorGrad>(&bias_dyn)?;

        let grad = out_dyn._grad.clone();
        Tensor::from_parts(out_final.into_inner(), shape, dtype, device, grad)
    }
}

impl<
    InF: Dim,
    OutF: Dim,
    InShape: Shape + DynShape + ReplaceLastDim<OutF> + crate::shapes::EndsWith<InF>,
    B: crate::tensor::backend::VariableBackend
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + crate::exec::Capabilities,
    K: DType,
    Train: TrainState,
    G: RequiresGrad,
> Module<Tensor<InShape, B, K, G>>
    for Linear<DimCons<InF, DimCons<OutF, Nil>>, B, crate::nn::optional::False, K, Train>
where
    InShape::Output: DynShape,
    G: GradJoin<Train::TensorGrad>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
{
    type Output = Tensor<InShape::Output, B, K, JoinedGrad<G, Train::TensorGrad>>;
    type Error = Error;

    fn forward(&self, x: Tensor<InShape, B, K, G>) -> core::result::Result<Self::Output, Error> {
        let dtype = x._dtype.clone();
        let device = x._device.clone();

        let mut dims = x.shape_buf().as_ref().to_vec();
        let last_idx = dims.len().saturating_sub(1);
        if last_idx < dims.len() {
            dims[last_idx] = self.weight.shape_dims()[0];
        }
        let shape = shape_buf_from_dims::<InShape::Output>(OperationKind::MatMul, &dims)?;

        let weight_dyn = self.weight.as_tensor()?.into_shape::<Dyn>()?;
        let weight_t = weight_dyn.transpose_runtime(0, 1)?;
        let x_dyn = x.into_shape::<Dyn>()?;
        let out_final = x_dyn.matmul(&weight_t)?;

        let grad = out_final._grad.clone();
        Tensor::from_parts(out_final.into_inner(), shape, dtype, device, grad)
    }
}

impl<
    InF: Dim,
    OutF: Dim,
    InShape: Shape + DynShape + ReplaceLastDim<OutF> + crate::shapes::EndsWith<InF>,
    B: crate::tensor::backend::VariableBackend
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + crate::exec::Capabilities
        + Execute<op::Add>,
    K: DType,
    Train: TrainState,
> Module<Tensor<InShape, B, K>> for Linear<DimCons<InF, DimCons<OutF, Nil>>, B, Dyn, K, Train>
where
    InShape::Output: DynShape,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
{
    type Output = Tensor<InShape::Output, B, K>;
    type Error = Error;

    fn forward(&self, x: Tensor<InShape, B, K>) -> core::result::Result<Self::Output, Error> {
        let dtype = x._dtype.clone();
        let device = x._device.clone();

        let mut dims = x.shape_buf().as_ref().to_vec();
        let last_idx = dims.len().saturating_sub(1);
        if last_idx < dims.len() {
            dims[last_idx] = self.weight.shape_dims()[0];
        }
        let shape = shape_buf_from_dims::<InShape::Output>(OperationKind::MatMul, &dims)?;

        let weight_dyn = self.weight.as_tensor()?.into_shape::<Dyn>()?;
        let weight_t = weight_dyn.transpose_runtime(0, 1)?;
        let x_dyn = x.into_shape::<Dyn>()?;
        let out_dyn = x_dyn.matmul(&weight_t)?;

        let out_final = if let Some(b) = &self.bias {
            let bias_dyn = b.as_tensor()?.into_shape::<Dyn>()?;
            out_dyn.broadcast_add(&bias_dyn)?
        } else {
            out_dyn
        };

        Tensor::from_parts(
            out_final.into_inner(),
            shape,
            dtype,
            device,
            core::marker::PhantomData,
        )
    }
}
