use crate::nn::module::Module;
use crate::prelude::*;

/// A shape marker trait specifying the input and output features of a [`Linear`] layer.
///
/// The typical usage is to supply a 2-tuple `(InF, OutF)` where:
/// * `InF` — Number of input features (the last dimension of the input tensor).
/// * `OutF` — Number of output features.
///
/// ## Examples
/// ```rust,ignore
/// // Static linear layer: 784 inputs → 256 outputs
/// type S = s![784, 256];
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

/// A fully connected (dense) linear layer: `y = x @ Wᵀ + b`.
///
/// `S` encodes both the input and output feature dimensions via [`LinearShape`]. The most common
/// form is `s![InF, OutF]`. For dynamic feature sizes use `Dyn` or mixed partial types.
///
/// ## Examples
///
/// ```rust,ignore
/// use incin::prelude::*;
///
/// // A fully static linear layer: 512 inputs → 256 outputs
/// let layer = Linear::<s![512, 256], MyBackend>::build(())?;
///
/// // A dynamic linear layer — shape known only at runtime
/// let layer = Linear::<Dyn, MyBackend>::build((512, 256))?;
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
        let (out_f, in_f) = (dims[0] as u64, dims[1] as u64);
        let bias_params = self
            .bias
            .as_ref()
            .map(|b| b.shape_dims().iter().product::<usize>() as u64)
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

        let mut dims = <InShape as DynShape>::dims(x.shape_field()).into();
        let last_idx = dims.len().saturating_sub(1);
        if last_idx < dims.len() {
            let w_dims = <(OutF, InF) as DynShape>::dims(self.weight.as_tensor()?.shape_field());
            dims[last_idx] = w_dims[0];
        }
        let shape = <InShape::Output as Shape>::from_dyn(&dims).unwrap();

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

        Ok(Tensor::from_parts_unchecked(
            out_final.into_inner(),
            shape,
            dtype,
            device,
            core::marker::PhantomData,
        ))
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

        let mut dims = <InShape as DynShape>::dims(x.shape_field()).into();
        let last_idx = dims.len().saturating_sub(1);
        if last_idx < dims.len() {
            let w_dims = <(OutF, InF) as DynShape>::dims(self.weight.as_tensor()?.shape_field());
            dims[last_idx] = w_dims[0];
        }
        let shape = <InShape::Output as Shape>::from_dyn(&dims).unwrap();

        let weight_dyn = self.weight.as_tensor()?.into_shape::<Dyn>()?;
        let weight_t = weight_dyn.transpose::<0, 1>()?;
        let x_dyn = x.into_shape::<Dyn>()?;
        let out_final = x_dyn.matmul(&weight_t)?;

        Ok(Tensor::from_parts_unchecked(
            out_final.into_inner(),
            shape,
            dtype,
            device,
            core::marker::PhantomData,
        ))
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

        let mut dims = <InShape as DynShape>::dims(x.shape_field()).into();
        let last_idx = dims.len().saturating_sub(1);
        if last_idx < dims.len() {
            let w_dims = <(OutF, InF) as DynShape>::dims(self.weight.as_tensor()?.shape_field());
            dims[last_idx] = w_dims[0];
        }
        let shape = <InShape::Output as Shape>::from_dyn(&dims).unwrap();

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

        Ok(Tensor::from_parts_unchecked(
            out_final.into_inner(),
            shape,
            dtype,
            device,
            core::marker::PhantomData,
        ))
    }
}
