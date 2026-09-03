use crate::dist::placement::Local;
use crate::err::{Error, Result};
use crate::exec::capability::Capabilities;
use crate::exec::catalog::{DropoutAttributes, op};
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::nn::{Module, TrainMode};
use crate::shapes::{DynShape, Shape};
use crate::tensor::backend::Execute;
use crate::tensor::backend::SupportsDType;
use crate::tensor::base::Tensor;
use crate::tensor::device::ConstDevice;
use crate::tensor::dtype::BuiltinDType;
use crate::tensor::grad::RequiresGrad;

/// A Dropout layer.
///
/// Randomly zeroes some of the elements of the input tensor with probability `p` using samples
/// from a uniform distribution. The elements to zero are randomized on every forward call during training.
///
/// This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons.
/// Furthermore, the outputs are scaled by a factor of `1 / (1 - p)` during training. This means that during
/// evaluation the module simply computes an identity function.
#[derive(Debug, Clone)]
pub struct Dropout {
    /// The probability of an element to be zeroed.
    pub p: f32,
    /// Whether the module is in training mode.
    /// If false, dropout acts as an identity function.
    pub is_training: bool,
}

impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitParameters<B> for Dropout {
    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        _: &crate::nn::StatePath,
        _: &mut V,
    ) -> Result<()> {
        Ok(())
    }
}

impl Dropout {
    /// Creates a new Dropout module with the specified probability.
    /// By default, `is_training` is set to true.
    pub fn new(p: f32) -> Self {
        Self {
            p,
            is_training: true,
        }
    }
}

impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitState<B> for Dropout {
    fn visit_state<V: crate::nn::StateVisitor<B>>(
        &self,
        _: &crate::nn::StatePath,
        _: &mut V,
    ) -> Result<()> {
        Ok(())
    }
}

impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitStateMut<B> for Dropout {
    fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        _: &crate::nn::StatePath,
        _: &mut V,
    ) -> Result<()> {
        Ok(())
    }
}

impl crate::nn::NamedLayers for Dropout {
    fn layer_structure(&self, prefix: &str) -> alloc::vec::Vec<crate::nn::LayerNode> {
        alloc::vec![crate::nn::LayerNode {
            name: alloc::string::String::from(prefix),
            type_name: alloc::string::String::from("Dropout"),
            shape_info: alloc::format!("p={}", self.p),
            children: alloc::vec![],
        }]
    }
}

impl crate::nn::ShapeInfo for Dropout {
    fn shape_info(&self) -> Option<alloc::string::String> {
        None
    }
}

impl<B: crate::tensor::backend::VariableBackend, NewD: crate::tensor::device::Device>
    crate::tensor::transfer::ToDevice<B, NewD> for Dropout
{
    type Output = Dropout;
    fn to_device(self, _arg: &NewD::Arg) -> Result<Self::Output> {
        Ok(self)
    }
}

impl TrainMode for Dropout {
    /// Directly sets `is_training` - `Dropout`'s own forward already reads
    /// this flag to decide identity-vs-random-zeroing behavior.
    fn set_training(&mut self, training: bool) {
        self.is_training = training;
    }
}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend,
    K: BuiltinDType,
    G: RequiresGrad,
    L: crate::shapes::FreshDense<S>,
> Module<Tensor<S, B, K, G, Local, L>> for Dropout
where
    B: SupportsDType<K> + Capabilities + Execute<op::Dropout>,
    B::Device: ConstDevice,
    <B as Execute<op::Dropout>>::Output: Into<B::Storage<K>>,
{
    /// The operand's own layout, which is the one place in the crate where
    /// carrying it is right rather than merely typechecking.
    ///
    /// Every other shape-preserving operation writes a fresh buffer on every
    /// call, so returning the operand's claim describes memory the operand
    /// never touched. Dropout has a real identity path -- eval mode, or
    /// `p == 0` -- which hands back the very tensor it was given, strides and
    /// all, so for that branch the operand's layout is exactly right.
    ///
    /// The training branch is what constrains the bound. It writes a dense
    /// buffer, so the layout carried across both branches has to be one a fresh
    /// dense allocation also satisfies -- which is precisely
    /// [`FreshDense<S>`](crate::shapes::FreshDense), the sealed bound the
    /// constructors use, rather than [`Layout`](crate::shapes::Layout). Bounding
    /// on `Layout` would compile today, because `Dyn` and `RowMajor` are the
    /// only layouts and both are dense; it would start lying the day a
    /// `ChannelsLast` exists, and the compiler would not ask.
    ///
    /// Pinned by `dropout_carries_its_operand_layout_and_both_branches_earn_it`.
    type Output = Tensor<S, B, K, G, Local, L>;
    type Error = Error;

    #[inline]
    fn forward(
        &self,
        x: Tensor<S, B, K, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        if !self.is_training || self.p <= 0.0 {
            return Ok(x);
        }

        let input = TensorHandle::from_storage::<B, K, Local>(&x.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&x._grad)
            .with_training(self.is_training);
        let output = dispatch::execute_shaped::<op::Dropout, B, S>(
            &context,
            DropoutAttributes {
                probability: self.p as f64,
                training: self.is_training,
            },
            &[input],
            &x._shape,
        )
        .map_err(Error::from)?
        .into();
        Tensor::from_shape_value(
            output,
            x._shape.clone(),
            x._dtype.clone(),
            x._device.clone(),
            x._grad,
        )
    }
}
