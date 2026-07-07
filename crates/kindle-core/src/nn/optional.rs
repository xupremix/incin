use crate::prelude::{Backend, DynShape, Param, Result, Shape};
use crate::tensor::arg_into::TensorArgsData;
use crate::nn::init::Init;
use crate::tensor::grad::Grad;

/// Trait governing optional module parameters (e.g., bias tensors).
/// By plugging in different structs (`True`, `False`, `DynParam`),
/// memory and compilation constraints correctly update for the parameter.
pub trait OptionalField {
    type BuildArgs;

    fn build<S: Shape + DynShape, B: Backend>(
        args_data: <(S, B::DType, B::Device, Grad) as crate::prelude::TensorArgs<S, B::DType, B::Device, Grad>>::Args,
        init: Init,
        build_args: Self::BuildArgs,
    ) -> Result<Option<Param<S, B>>>;
}

/// Variant indicating the parameter should ALWAYS exist.
#[derive(Debug, Clone, Copy, Default)]
pub struct True;

impl OptionalField for True {
    type BuildArgs = ();

    #[inline(always)]
    fn build<S: Shape + DynShape, B: Backend>(
        args_data: <(S, B::DType, B::Device, Grad) as crate::prelude::TensorArgs<S, B::DType, B::Device, Grad>>::Args,
        init: Init,
        _build_args: (),
    ) -> Result<Option<Param<S, B>>> {
        Ok(Some(Param::<S, B>::new_init_raw(args_data, init)?))
    }
}

/// Variant indicating the parameter should NEVER exist (Zero-cost omitted).
#[derive(Debug, Clone, Copy, Default)]
pub struct False;

impl OptionalField for False {
    type BuildArgs = ();

    #[inline(always)]
    fn build<S: Shape + DynShape, B: Backend>(
        _args_data: <(S, B::DType, B::Device, Grad) as crate::prelude::TensorArgs<S, B::DType, B::Device, Grad>>::Args,
        _init: Init,
        _build_args: (),
    ) -> Result<Option<Param<S, B>>> {
        Ok(None)
    }
}

/// Variant indicating the parameter's existence is determined dynamically at runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct DynParam;

impl OptionalField for DynParam {
    type BuildArgs = bool;

    #[inline(always)]
    fn build<S: Shape + DynShape, B: Backend>(
        args_data: <(S, B::DType, B::Device, Grad) as crate::prelude::TensorArgs<S, B::DType, B::Device, Grad>>::Args,
        init: Init,
        has_param: bool,
    ) -> Result<Option<Param<S, B>>> {
        if has_param {
            Ok(Some(Param::<S, B>::new_init_raw(args_data, init)?))
        } else {
            Ok(None)
        }
    }
}
