use crate::prelude::*;
use alloc::vec::Vec;
use std::collections::HashMap;

/// A trait implemented by all Neural Network modules to manage their state (weights).
/// Usually automatically derived via `#[kindle::module]`.
pub trait StateDict<B: Backend> {
    /// Loads the module's state from a dictionary of dynamic tensors.
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &HashMap<String, Tensor<Dyn, B>>,
    ) -> Result<()>;

    /// Collects the module's state into a dictionary of dynamic tensors.
    fn state_dict(&self, prefix: &str, tensors: &mut HashMap<String, Tensor<Dyn, B>>);
    
    /// Helper: Serializes this module's state to a given serializer.
    fn save_to<S: crate::serialize::Serializer>(&self, serializer: &mut S) -> core::result::Result<(), S::Error> 
    where
        <<B as Backend>::DType as DType>::Field: Default,
        <<B as Backend>::Device as Device>::Field: Default {
        let mut map = HashMap::new();
        self.state_dict("", &mut map);
        serializer.serialize(&map)
    }
    
    /// Helper: Deserializes this module's state from a given deserializer.
    fn load_from<D: crate::serialize::Deserializer>(&mut self, deserializer: &mut D, device: &KindleDevice) -> Result<()> 
    where
        <<B as Backend>::DType as DType>::Field: Default,
        <<B as Backend>::Device as Device>::Field: Default {
        let map = deserializer.deserialize(device).map_err(|e| Error::ShapeMismatch { op: "Deserialization", expected: vec![], got: vec![], msg: format!("{:?}", e) })?;
        self.load_state_dict("", &map)
    }
}

/// A trait implemented by all Neural Network modules.
/// Usually automatically derived via `#[kindle::module]`.
pub trait Parameters<B: Backend> {
    /// Recursively extract all trainable parameters from this module.
    /// The parameters are returned as a list of backend-specific raw variables,
    /// which can be passed to an optimizer (e.g., `candle_nn::optim::SGD`).
    fn parameters(&self) -> Vec<B::RawVar>;
}

/// A trait to transfer ownership of a module to a new device.
pub trait ToDevice<B: Backend, NewD: Device> {
    type Output;
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output>;
}

impl<T: ToDevice<B, NewD>, B: Backend, NewD: Device> ToDevice<B, NewD> for Option<T> {
    type Output = Option<T::Output>;
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        self.map(|t| t.to_device(arg)).transpose()
    }
}

#[doc(hidden)]
pub trait AutorefParametersFallback<B: Backend> {
    fn maybe_parameters(&self, _phantom: core::marker::PhantomData<B>) -> Vec<B::RawVar> { Vec::new() }
}
impl<T, B: Backend> AutorefParametersFallback<B> for &&T {}

#[doc(hidden)]
pub trait AutorefParameters<B: Backend> {
    fn maybe_parameters(&self, _phantom: core::marker::PhantomData<B>) -> Vec<B::RawVar>;
}
impl<T: Parameters<B>, B: Backend> AutorefParameters<B> for &T {
    #[inline]
    fn maybe_parameters(&self, _phantom: core::marker::PhantomData<B>) -> Vec<B::RawVar> {
        (*self).parameters()
    }
}

#[doc(hidden)]
pub trait AutorefStateDictFallback<B: Backend> {
    fn maybe_load_state_dict(&mut self, _phantom: core::marker::PhantomData<B>, _prefix: &str, _tensors: &HashMap<String, Tensor<Dyn, B>>) -> Result<()> { Ok(()) }
    fn maybe_state_dict(&self, _phantom: core::marker::PhantomData<B>, _prefix: &str, _tensors: &mut HashMap<String, Tensor<Dyn, B>>) {}
}
impl<T, B: Backend> AutorefStateDictFallback<B> for &mut &mut T {}
impl<T, B: Backend> AutorefStateDictFallback<B> for &&T {}

#[doc(hidden)]
pub trait AutorefStateDict<B: Backend> {
    fn maybe_load_state_dict(&mut self, _phantom: core::marker::PhantomData<B>, prefix: &str, tensors: &HashMap<String, Tensor<Dyn, B>>) -> Result<()>;
    fn maybe_state_dict(&self, _phantom: core::marker::PhantomData<B>, prefix: &str, tensors: &mut HashMap<String, Tensor<Dyn, B>>);
}

// For mutable operations
impl<T: StateDict<B>, B: Backend> AutorefStateDict<B> for &mut T {
    #[inline]
    fn maybe_load_state_dict(&mut self, _phantom: core::marker::PhantomData<B>, prefix: &str, tensors: &HashMap<String, Tensor<Dyn, B>>) -> Result<()> {
        (*self).load_state_dict(prefix, tensors)
    }
    #[inline]
    fn maybe_state_dict(&self, _phantom: core::marker::PhantomData<B>, prefix: &str, tensors: &mut HashMap<String, Tensor<Dyn, B>>) {
        (**self).state_dict(prefix, tensors)
    }
}

// For immutable operations (state_dict uses &self)
impl<T: StateDict<B>, B: Backend> AutorefStateDict<B> for &T {
    #[inline]
    fn maybe_load_state_dict(&mut self, _phantom: core::marker::PhantomData<B>, _prefix: &str, _tensors: &HashMap<String, Tensor<Dyn, B>>) -> Result<()> {
        Ok(()) // Should not be called
    }
    #[inline]
    fn maybe_state_dict(&self, _phantom: core::marker::PhantomData<B>, prefix: &str, tensors: &mut HashMap<String, Tensor<Dyn, B>>) {
        (*self).state_dict(prefix, tensors)
    }
}

/// A generic Neural Network Layer or Module.
/// Capable of taking an input and returning an output or error.
/// 
/// `Module` is the fundamental building block of neural networks in Kindle.
/// Any layer, from a simple ReLU to an entire ResNet architecture, implements `Module`.
/// 
/// ## Deriving Modules
/// 
/// While you can implement `Module` manually, the recommended approach is to use the `#[module]` attribute macro
/// provided by `kindle-macros`. This automatically derives `Parameters`, `StateDict`, and generates structural boilerplate.
/// 
/// ```rust,ignore
/// use kindle::prelude::*;
/// 
/// #[module]
/// pub struct MyLayer<B: Backend> {
///     weight: Param<Tensor<s![128, 128], B>>,
///     bias: Param<Tensor<s![128], B>>,
/// }
/// 
/// impl<B: Backend> Module<Tensor<s![1, 128], B>> for MyLayer<B> {
///     type Output = Tensor<s![1, 128], B>;
///     type Error = Error;
/// 
///     fn forward(&self, x: Tensor<s![1, 128], B>) -> Result<Self::Output> {
///         // Custom logic here
///         Ok(x)
///     }
/// }
/// ```
pub trait Module<Input> {
    type Output;
    type Error;

    fn forward(&self, input: Input) -> core::result::Result<Self::Output, Self::Error>;
}

/// A sequential container for composing two modules.
/// `Sequential` automatically implements `Module` if the inner modules are compatible.
#[derive(Debug, Clone)]
pub struct Sequential<L1, L2>(pub L1, pub L2);

impl<I, L1, L2> Module<I> for Sequential<L1, L2>
where
    L1: Module<I>,
    L2: Module<L1::Output, Error = L1::Error>,
{
    type Output = L2::Output;
    type Error = L1::Error;

    #[inline]
    fn forward(&self, input: I) -> core::result::Result<Self::Output, Self::Error> {
        let out1 = self.0.forward(input)?;
        self.1.forward(out1)
    }
}

impl<B: Backend, NewD: Device, L1, L2> ToDevice<B, NewD> for Sequential<L1, L2>
where
    L1: ToDevice<B, NewD>,
    L2: ToDevice<B, NewD>,
{
    type Output = Sequential<L1::Output, L2::Output>;
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        Ok(Sequential(self.0.to_device(arg)?, self.1.to_device(arg)?))
    }
}

impl<B: Backend, L1, L2> Parameters<B> for Sequential<L1, L2>
where
    L1: Parameters<B>,
    L2: Parameters<B>,
{
    fn parameters(&self) -> Vec<B::RawVar> {
        let mut p = self.0.parameters();
        p.extend(self.1.parameters());
        p
    }
}

impl<B: Backend, L1, L2> StateDict<B> for Sequential<L1, L2>
where
    L1: StateDict<B>,
    L2: StateDict<B>,
{
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &HashMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        self.0.load_state_dict(&format!("{}0.", prefix), tensors)?;
        self.1.load_state_dict(&format!("{}1.", prefix), tensors)?;
        Ok(())
    }

    fn state_dict(&self, prefix: &str, tensors: &mut HashMap<String, Tensor<Dyn, B>>) {
        self.0.state_dict(&format!("{}0.", prefix), tensors);
        self.1.state_dict(&format!("{}1.", prefix), tensors);
    }
}

// Dummy implementations for primitive/marker types that are often fields in modules.
macro_rules! impl_dummy_state {
    ($($t:ty),+) => {
        $(
            impl<B: Backend> Parameters<B> for $t {
                fn parameters(&self) -> Vec<B::RawVar> {
                    Vec::new()
                }
            }

            impl<B: Backend> StateDict<B> for $t {
                fn load_state_dict(&mut self, _prefix: &str, _tensors: &HashMap<String, Tensor<Dyn, B>>) -> Result<()> {
                    Ok(())
                }

                fn state_dict(&self, _prefix: &str, _tensors: &mut HashMap<String, Tensor<Dyn, B>>) {
                }
            }
        )+
    };
}

impl_dummy_state!(usize, f32);

impl<T: ?Sized, B: Backend> Parameters<B> for core::marker::PhantomData<T>
where
    T: crate::prelude::DType,
{
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}
impl<T: ?Sized, B: Backend> StateDict<B> for core::marker::PhantomData<T>
where
    T: crate::prelude::DType,
{
    fn load_state_dict(
        &mut self,
        _prefix: &str,
        _tensors: &HashMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        Ok(())
    }
    fn state_dict(&self, _prefix: &str, _tensors: &mut HashMap<String, Tensor<Dyn, B>>) {}
}

impl<L: Parameters<B>, B: Backend> Parameters<B> for Option<L> {
    fn parameters(&self) -> Vec<B::RawVar> {
        match self {
            Some(v) => v.parameters(),
            None => Vec::new(),
        }
    }
}

impl<L: StateDict<B>, B: Backend> StateDict<B> for Option<L> {
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &HashMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        if let Some(v) = self {
            v.load_state_dict(prefix, tensors)?;
        }
        Ok(())
    }

    fn state_dict(&self, prefix: &str, tensors: &mut HashMap<String, Tensor<Dyn, B>>) {
        if let Some(v) = self {
            v.state_dict(prefix, tensors);
        }
    }
}

/// A macro to easily build Sequential models with many layers.
/// `seq!(L1, L2, L3)` expands to `Sequential(L1, Sequential(L2, L3))`.
#[macro_export]
macro_rules! seq {
    ($l1:expr) => {
        $l1
    };
    ($l1:expr, $($tail:expr),+ $(,)?) => {
        $crate::nn::Sequential($l1, $crate::seq!($($tail),+))
    };
}
