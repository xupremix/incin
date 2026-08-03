use crate::nn::module::Module;
use crate::prelude::*;
use crate::shapes::error::OperationKind;
use crate::shapes::shape::field_from_dims;

/// A shape marker trait specifying the input and output features of a [`Linear`] layer.
///
/// The typical usage is to supply a 2-tuple `(InF, OutF)` where:
/// * `InF` — Number of input features (the last dimension of the input tensor).
/// * `OutF` — Number of output features.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// use incin_core::prelude::{LinearShape, s};
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
    ) -> (usize, usize, Self::WeightArg, Self::BiasArg);
}

impl<InF: Dim, OutF: Dim> LinearShape for (InF, OutF) {
    /// The number of input features (last dimension of the input tensor).
    type InF = InF;
    /// The number of output features (last dimension of the output tensor).
    type OutF = OutF;
    /// The shape argument type used to construct the weight tensor.
    type WeightArg = (<OutF as Dim>::Arg, <InF as Dim>::Arg);
    /// The shape argument type used to construct the bias tensor.
    type BiasArg = (<OutF as Dim>::Arg,);
    /// The static shape type of the weight parameter tensor.
    type WeightShape = (OutF, InF);
    /// The static shape type of the bias parameter tensor.
    type BiasShape = (OutF,);

    #[inline]
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(
        target: (<Self::InF as Dim>::Arg, <Self::OutF as Dim>::Arg),
    ) -> (usize, usize, Self::WeightArg, Self::BiasArg) {
        let in_f = InF::from_arg(target.0.clone()).size();
        let out_f = OutF::from_arg(target.1.clone()).size();
        (in_f, out_f, (target.1.clone(), target.0), (target.1,))
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
    ) -> (usize, usize, Self::WeightArg, Self::BiasArg) {
        let in_f = target.0;
        let out_f = target.1;
        (in_f, out_f, alloc::vec![out_f, in_f], alloc::vec![out_f])
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
impl<InF, OutF> TwoWayColumnLinearShape for (InF, OutF)
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
impl<InF, OutF> TwoWayRowLinearShape for (InF, OutF)
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
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<f32, incin_core::prelude::Cpu>;
/// use incin::prelude::*;
///
/// // A fully static linear layer: 512 inputs → 256 outputs
/// let layer = Linear::<s![512, 256], DefaultBackend>::build(())?;
///
/// // A dynamic linear layer — shape known only at runtime
/// let layer = Linear::<Dyn, DefaultBackend>::build((512, 256))?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
#[incin_macros::module(internal, no_stats)]
pub struct Linear<
    S: LinearShape,
    B: Backend,
    Bias: crate::nn::optional::OptionalField = crate::nn::optional::True,
> {
    /// The learnable weight matrix parameter.
    pub weight: Param<S::WeightShape, B>,
    /// The optional learnable bias vector parameter.
    pub bias: Option<Param<S::BiasShape, B>>,
    #[module(ignore)]
    _phantom: core::marker::PhantomData<(S, B, Bias)>,
}

impl<S: LinearShape, B: Backend, Bias: crate::nn::optional::OptionalField> ComputeStats
    for Linear<S, B, Bias>
{
    /// `params` = weight elements + bias elements (if present). `macs` =
    /// `in_features * out_features * batch` — `y = xWᵀ + b` needs one MAC
    /// per (batch, in, out) triple. Unlike `Conv1d`/`Conv2d`, this needs no
    /// external input-shape info: a `Linear` layer's weight shape *is* its
    /// in/out feature count, so this is exact, not an approximation.
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

// Implement `Module` for `Linear` when input shape is `(Batch, In)`.

impl<S, B, Bias> Linear<S, B, Bias>
where
    S: LinearShape,
    B: Backend + SupportsDType<B::FloatElem>,
    Bias: crate::nn::optional::OptionalField,
    <B::FloatElem as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
{
    /// Builds the layer from its exact compressed argument tuple.
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::InF as Dim>::Arg,
                <S::OutF as Dim>::Arg,
                <B::FloatElem as DType>::Arg,
                <B::Device as Device>::Arg,
                <Bias as crate::nn::optional::OptionalField>::Arg,
            )>,
    {
        use crate::tensor::arg_into::LayerArgInto;
        let (in_arg, out_arg, dtype, device, bias) = args.into_layer_arg();
        Self::build_full(in_arg, out_arg, dtype, device, bias)
    }

    pub(crate) fn build_full(
        in_arg: <S::InF as Dim>::Arg,
        out_arg: <S::OutF as Dim>::Arg,
        dtype: <B::FloatElem as DType>::Arg,
        device: <B::Device as Device>::Arg,
        bias_arg: <Bias as crate::nn::optional::OptionalField>::Arg,
    ) -> Result<Self> {
        let (in_f, _out_f, w_args, b_args) = S::build_args((in_arg, out_arg));
        let init = crate::nn::init::Init::KaimingUniform {
            fan_in: in_f,
            a: f64::sqrt(5.0),
        };
        let weight = Param::<S::WeightShape, B>::new_init_raw(
            crate::tensor::arg_into::TensorArgsData {
                shape: w_args,
                dtype: dtype.clone(),
                device: device.clone(),
                grad: (),
            },
            init,
        )?;
        let bias = if Bias::init(bias_arg) {
            Some(Param::<S::BiasShape, B>::new_init_raw(
                crate::tensor::arg_into::TensorArgsData {
                    shape: b_args,
                    dtype,
                    device,
                    grad: (),
                },
                init,
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

// In PyTorch, input is typically (*, InF) and weight is (OutF, InF).
// In PyTorch, input is typically (*, InF) and weight is (OutF, InF).

// Dynamic input
impl<B: Backend> Module<Tensor<Dyn, B>> for Linear<Dyn, B, crate::nn::optional::True> {
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<Dyn, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<Dyn, B>) -> core::result::Result<Tensor<Dyn, B>, Error> {
        let weight_t = self.weight.as_tensor()?.transpose::<0, 1>()?;
        let out = x.matmul(&weight_t)?;
        out.add(&self.bias.as_ref().unwrap().as_tensor()?)
    }
}

impl<B: Backend> Module<Tensor<Dyn, B>> for Linear<Dyn, B, crate::nn::optional::False> {
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<Dyn, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<Dyn, B>) -> core::result::Result<Tensor<Dyn, B>, Error> {
        let weight_t = self.weight.as_tensor()?.transpose::<0, 1>()?;
        x.matmul(&weight_t)
    }
}

impl<B: Backend> Module<Tensor<Dyn, B>> for Linear<Dyn, B, Dyn> {
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<Dyn, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<Dyn, B>) -> core::result::Result<Tensor<Dyn, B>, Error> {
        let weight_t = self.weight.as_tensor()?.transpose::<0, 1>()?;
        let out = x.matmul(&weight_t)?;
        if let Some(b) = &self.bias {
            out.add(&b.as_tensor()?)
        } else {
            Ok(out)
        }
    }
}

// Statically typed input utilizing ReplaceLastDim
impl<
    InF: Dim,
    OutF: Dim,
    InShape: Shape + DynShape + ReplaceLastDim<OutF> + crate::shapes::EndsWith<InF>,
    B: Backend,
> Module<Tensor<InShape, B>> for Linear<(InF, OutF), B, crate::nn::optional::True>
where
    InShape::Output: DynShape,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<InShape::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<InShape, B>) -> core::result::Result<Self::Output, Error> {
        let dtype = x._dtype.clone();
        let device = x._device.clone();

        let mut dims = <InShape as Shape>::dims(x.shape_field()).into();
        let last_idx = dims.len().saturating_sub(1);
        if last_idx < dims.len() {
            let w_dims = <(OutF, InF) as Shape>::dims(self.weight.as_tensor()?.shape_field());
            dims[last_idx] = w_dims[0];
        }
        let shape = field_from_dims::<InShape::Output>(OperationKind::MatMul, &dims)?;

        let weight_dyn = self.weight.as_tensor()?.into_shape::<Dyn>()?;
        let weight_t = weight_dyn.transpose::<0, 1>()?;
        let x_dyn = x.into_shape::<Dyn>()?;
        let out_dyn = x_dyn.matmul(&weight_t)?;

        let bias_dyn = self
            .bias
            .as_ref()
            .unwrap()
            .as_tensor()?
            .into_shape::<Dyn>()?;
        let out_final = out_dyn.broadcast_add(&bias_dyn)?;

        Tensor::from_parts(
            out_final.into_inner(),
            shape,
            dtype,
            device,
            core::marker::PhantomData,
        )
    }
}

impl<
    InF: Dim,
    OutF: Dim,
    InShape: Shape + DynShape + ReplaceLastDim<OutF> + crate::shapes::EndsWith<InF>,
    B: Backend,
> Module<Tensor<InShape, B>> for Linear<(InF, OutF), B, crate::nn::optional::False>
where
    InShape::Output: DynShape,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<InShape::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<InShape, B>) -> core::result::Result<Self::Output, Error> {
        let dtype = x._dtype.clone();
        let device = x._device.clone();

        let mut dims = <InShape as Shape>::dims(x.shape_field()).into();
        let last_idx = dims.len().saturating_sub(1);
        if last_idx < dims.len() {
            let w_dims = <(OutF, InF) as Shape>::dims(self.weight.as_tensor()?.shape_field());
            dims[last_idx] = w_dims[0];
        }
        let shape = field_from_dims::<InShape::Output>(OperationKind::MatMul, &dims)?;

        let weight_dyn = self.weight.as_tensor()?.into_shape::<Dyn>()?;
        let weight_t = weight_dyn.transpose::<0, 1>()?;
        let x_dyn = x.into_shape::<Dyn>()?;
        let out_final = x_dyn.matmul(&weight_t)?;

        Tensor::from_parts(
            out_final.into_inner(),
            shape,
            dtype,
            device,
            core::marker::PhantomData,
        )
    }
}

impl<
    InF: Dim,
    OutF: Dim,
    InShape: Shape + DynShape + ReplaceLastDim<OutF> + crate::shapes::EndsWith<InF>,
    B: Backend,
> Module<Tensor<InShape, B>> for Linear<(InF, OutF), B, Dyn>
where
    InShape::Output: DynShape,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<InShape::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<InShape, B>) -> core::result::Result<Self::Output, Error> {
        let dtype = x._dtype.clone();
        let device = x._device.clone();

        let mut dims = <InShape as Shape>::dims(x.shape_field()).into();
        let last_idx = dims.len().saturating_sub(1);
        if last_idx < dims.len() {
            let w_dims = <(OutF, InF) as Shape>::dims(self.weight.as_tensor()?.shape_field());
            dims[last_idx] = w_dims[0];
        }
        let shape = field_from_dims::<InShape::Output>(OperationKind::MatMul, &dims)?;

        let weight_dyn = self.weight.as_tensor()?.into_shape::<Dyn>()?;
        let weight_t = weight_dyn.transpose::<0, 1>()?;
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
