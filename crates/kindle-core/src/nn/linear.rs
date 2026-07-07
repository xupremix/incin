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
    type InF;
    type OutF;
    type WeightArg: crate::tensor::arg_into::NotUnit;
    type BiasArg: crate::tensor::arg_into::NotUnit;
    type WeightShape: Shape<Arg = Self::WeightArg> + DynShape;
    type BiasShape: Shape<Arg = Self::BiasArg> + DynShape;

    type Target;
    fn build_args(target: Self::Target) -> (usize, usize, Self::WeightArg, Self::BiasArg);
}

impl<InF: Dim, OutF: Dim> LinearShape for (InF, OutF) {
    type InF = InF;
    type OutF = OutF;
    type WeightArg = (<OutF as Dim>::Arg, <InF as Dim>::Arg);
    type BiasArg = (<OutF as Dim>::Arg,);
    type WeightShape = (OutF, InF);
    type BiasShape = (OutF,);

    type Target = (InF::Arg, OutF::Arg);

    #[inline]
    fn build_args(target: Self::Target) -> (usize, usize, Self::WeightArg, Self::BiasArg) {
        let in_f = InF::from_arg(target.0.clone()).size();
        let out_f = OutF::from_arg(target.1.clone()).size();
        (in_f, out_f, (target.1.clone(), target.0), (target.1,))
    }
}

impl LinearShape for Dyn {
    type InF = Dyn;
    type OutF = Dyn;
    type WeightArg = alloc::vec::Vec<usize>;
    type BiasArg = alloc::vec::Vec<usize>;
    type WeightShape = Dyn;
    type BiasShape = Dyn;

    type Target = (usize, usize);

    #[inline]
    fn build_args(target: Self::Target) -> (usize, usize, Self::WeightArg, Self::BiasArg) {
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
/// use kindle::prelude::*;
///
/// // A fully static linear layer: 512 inputs → 256 outputs
/// let layer = Linear::<s![512, 256], MyBackend>::new()?;
///
/// // A dynamic linear layer — shape known only at runtime
/// let layer = Linear::<Dyn, MyBackend>::new(512, 256)?;
/// ```
#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Linear<S: LinearShape, B: Backend, Bias: crate::nn::optional::OptionalField = crate::nn::optional::True> {
    pub weight: Param<S::WeightShape, B>,
    pub bias: Option<Param<S::BiasShape, B>>,
    #[module(ignore)]
    _phantom: core::marker::PhantomData<(S, B, Bias)>,
}

// Implement `Module` for `Linear` when input shape is `(Batch, In)`.

impl<S: LinearShape, B: Backend, Bias: crate::nn::optional::OptionalField> Linear<S, B, Bias>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new_with(args: S::Target, bias_args: Bias::BuildArgs) -> Result<Self> {
        let (in_f, _out_f, w_args, b_args) = S::build_args(args);

        let w_args_data = crate::tensor::arg_into::TensorArgsData {
            shape: w_args,
            dtype: (),
            device: (),
            grad: (),
        };
        let b_args_data = crate::tensor::arg_into::TensorArgsData {
            shape: b_args,
            dtype: (),
            device: (),
            grad: (),
        };

        let weight = Param::<S::WeightShape, B>::new_init_raw(
            w_args_data,
            crate::nn::init::Init::KaimingUniform {
                fan_in: in_f,
                a: f64::sqrt(5.0),
            },
        )?;
        
        let bias = Bias::build(
            b_args_data,
            crate::nn::init::Init::KaimingUniform {
                fan_in: in_f,
                a: f64::sqrt(5.0),
            },
            bias_args,
        )?;
        
        Ok(Self {
            weight,
            bias,
            _phantom: core::marker::PhantomData,
        })
    }
}

impl<S, B, Bias> Linear<S, B, Bias>
where
    S: LinearShape<Target = ((), ())>,
    B: Backend,
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    Bias: crate::nn::optional::OptionalField<BuildArgs = ()>,
{
    pub fn new() -> Result<Self> {
        Self::new_with(((), ()), ())
    }
}

// In PyTorch, input is typically (*, InF) and weight is (OutF, InF).
// In PyTorch, input is typically (*, InF) and weight is (OutF, InF).

// Dynamic input
impl<B: Backend> Module<Tensor<Dyn, B>> for Linear<Dyn, B, crate::nn::optional::True> {
    type Output = Tensor<Dyn, B>;
    type Error = Error;

    fn forward(&self, x: Tensor<Dyn, B>) -> core::result::Result<Tensor<Dyn, B>, Error> {
        let weight_t = self.weight.as_tensor()?.transpose::<0, 1>()?;
        let out = x.matmul(&weight_t)?;
        out.add(&self.bias.as_ref().unwrap().as_tensor()?)
    }
}

impl<B: Backend> Module<Tensor<Dyn, B>> for Linear<Dyn, B, crate::nn::optional::False> {
    type Output = Tensor<Dyn, B>;
    type Error = Error;

    fn forward(&self, x: Tensor<Dyn, B>) -> core::result::Result<Tensor<Dyn, B>, Error> {
        let weight_t = self.weight.as_tensor()?.transpose::<0, 1>()?;
        x.matmul(&weight_t)
    }
}

impl<B: Backend> Module<Tensor<Dyn, B>> for Linear<Dyn, B, crate::nn::optional::DynParam> {
    type Output = Tensor<Dyn, B>;
    type Error = Error;

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
    type Output = Tensor<InShape::Output, B>;
    type Error = Error;

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
        
        let bias_dyn = self.bias.as_ref().unwrap().as_tensor()?.into_shape::<Dyn>()?;
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
    type Output = Tensor<InShape::Output, B>;
    type Error = Error;

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
> Module<Tensor<InShape, B>> for Linear<(InF, OutF), B, crate::nn::optional::DynParam>
where
    InShape::Output: DynShape,
{
    type Output = Tensor<InShape::Output, B>;
    type Error = Error;

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
