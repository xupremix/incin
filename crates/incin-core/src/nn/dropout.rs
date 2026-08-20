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
> Module<Tensor<S, B, K, G>> for Dropout
where
    B: SupportsDType<K> + Capabilities + Execute<op::Dropout>,
    B::Device: ConstDevice,
    <B as Execute<op::Dropout>>::Output: Into<B::Storage<K>>,
{
    type Output = Tensor<S, B, K, G>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B, K, G>) -> core::result::Result<Tensor<S, B, K, G>, Error> {
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
