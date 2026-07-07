use crate::nn::{Module, Param};
use crate::prelude::*;
use typenum::Unsigned;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Conv1d<S: Conv1dShape, B: Backend> {
    pub weight: Param<S::WeightShape, B>,
    pub bias: Option<Param<S::BiasShape, B>>,
    #[module(ignore)]
    _phantom: core::marker::PhantomData<(S, B)>,
}


pub trait Conv1dShape: Shape + DynShape {
    type OutC: Dim;
    type InC: Dim;
    type K: Unsigned + Dim<Arg = ()>;
    type S: Unsigned + Dim<Arg = ()>;
    type P: Unsigned + Dim<Arg = ()>;
    type D: Unsigned + Dim<Arg = ()>;
    type WeightArg: crate::tensor::arg_into::NotUnit;
    type BiasArg: crate::tensor::arg_into::NotUnit;
    type WeightShape: Shape<Arg = Self::WeightArg> + DynShape;
    type BiasShape: Shape<Arg = Self::BiasArg> + DynShape;
    type Target;

    fn build_args(target: Self::Target) -> (usize, usize, Self::WeightArg, Self::BiasArg);
}

impl<OutC: Dim, InC: Dim, K: Unsigned + Dim<Arg=()>, S: Unsigned + Dim<Arg=()>, P: Unsigned + Dim<Arg=()>, D: Unsigned + Dim<Arg=()>> Conv1dShape for (OutC, InC, K, S, P, D) {
    type OutC = OutC;
    type InC = InC;
    type K = K;
    type S = S;
    type P = P;
    type D = D;
    type WeightArg = (<OutC as Dim>::Arg, <InC as Dim>::Arg, <K as Dim>::Arg);
    type BiasArg = (<OutC as Dim>::Arg,);
    type WeightShape = (OutC, InC, K);
    type BiasShape = (OutC,);
    type Target = (OutC::Arg, InC::Arg);

    #[inline]
    fn build_args(target: Self::Target) -> (usize, usize, Self::WeightArg, Self::BiasArg) {
        let out_channels = OutC::from_arg(target.0.clone()).size();
        let in_channels = InC::from_arg(target.1.clone()).size();
        (
            out_channels,
            in_channels,
            (target.0, target.1, K::from_arg(Default::default()).arg()),
            (Default::default(),),
        )
    }
}

impl<S: Conv1dShape, B: Backend> Conv1d<S, B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new() -> Result<Self>
    where
        (): crate::tensor::arg_into::ArgInto<S::Target>,
    {
        Self::new_dyn(())
    }

    pub fn new_dyn<A: crate::tensor::arg_into::ArgInto<S::Target>>(args: A) -> Result<Self> {
        let target = args.into_arg();
        let (_cout, _cin, w_args, b_args) = S::build_args(target);
        
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

        let weight = Param::<S::WeightShape, B>::zeros_raw(w_args_data)?;
        let bias = Param::<S::BiasShape, B>::zeros_raw(b_args_data)?;
        Ok(Self { weight, bias: Some(bias), _phantom: core::marker::PhantomData })
    }
}

impl<I, S, B, COut: Dim, CIn: Dim> Module<Tensor<I, B>> for Conv1d<S, B>
where
    S: Conv1dShape<OutC = COut, InC = CIn>,
    I: Shape + DynShape + crate::shapes::SpatialConv1d<COut, S::K, S::S, S::P, S::D> + crate::shapes::HasChannels1D<CIn>,
    B: Backend,
{
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let bias = match &self.bias {
            Some(b) => Some(b.as_tensor()?.detach()),
            None => None,
        };
        
        let x_shape = x.dims();
        let x_shape = x_shape.as_ref();
        let rank = x_shape.len();
        let batch_size: usize = x_shape[0..rank - 2].iter().product();
        let in_channels = x_shape[rank - 2];
        let length = x_shape[rank - 1];

        let x_inner = if rank > 3 {
            <B as Backend>::reshape(&x.inner, &[batch_size, in_channels, length])?
        } else {
            x.inner.clone()
        };

        let out = <B as Backend>::conv1d(
            &x_inner,
            &weight.inner,
            bias.as_ref().map(|b| b.inner()),
            S::S::USIZE,
            S::P::USIZE,
            S::D::USIZE,
        )?;

        let shape = <I as crate::shapes::SpatialConv1d<COut, S::K, S::S, S::P, S::D>>::compute_output_shape(
            x.shape_field(),
            weight.dims()[0],
        );
        
        let out_shape = <I::Output as DynShape>::dims(&shape);
        let out = if rank > 3 {
            <B as Backend>::reshape(&out, out_shape.as_ref())?
        } else {
            out
        };

        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            weight._device.clone(),
            x.grad_field().clone(),
        ))
    }
}



