use crate::dist::placement::Local;
use crate::err::{Error, ErrorMessage, Result};
use crate::exec::capability::Capabilities;
use crate::exec::catalog::op;
use crate::nn::Module;
use crate::shapes::error::{OperationKind, RankExpectation};
use crate::shapes::{Dense, Dyn, DynShape, Layout, Shape, ShapeError};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::dtype::DType;
use crate::tensor::grad::NoGrad;

/// Interpolation mode for [`Upsample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsampleMode {
    /// Nearest-neighbor interpolation (supported).
    Nearest,
    /// Bilinear interpolation (refused: no bilinear op exists in the
    /// operation catalog, so this mode fails closed at construction and
    /// in `forward`).
    Bilinear,
}

/// Nearest-neighbor upsampling for rank-3 `[N, C, W]` and rank-4
/// `[N, C, H, W]` inputs.
///
/// The layer is a thin typed wrapper over the `repeat_interleave` tensor
/// op: a scale of `k` along an axis repeats every element `k` times in
/// place. Two specification forms exist, matching the PyTorch API:
///
/// * scale-factor form ([`Upsample::nearest`]): per-axis integer factors.
/// * size form ([`Upsample::nearest_size`]): target spatial extents, which
///   must be exact integer multiples of the input extents (nearest
///   interpolation cannot represent fractional scales).
///
/// Bilinear mode has no underlying catalog op and is refused with a typed
/// [`Error::InvalidModuleState`] both at construction and in `forward`, so
/// a future bilinear op can adopt this mode without changing the API.
/// This layer holds no parameters and has no training-dependent behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Upsample {
    /// Interpolation mode. Only [`UpsampleMode::Nearest`] executes.
    pub mode: UpsampleMode,
    /// Per-axis integer scale factors `[h, w]`, when built in scale form.
    pub scale_factor: Option<[usize; 2]>,
    /// Target spatial extents `[h, w]`, when built in size form.
    pub size: Option<[usize; 2]>,
}

impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitParameters<B> for Upsample {
    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        _: &crate::nn::StatePath,
        _: &mut V,
    ) -> Result<()> {
        Ok(())
    }
}

impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitState<B> for Upsample {
    fn visit_state<V: crate::nn::StateVisitor<B>>(
        &self,
        _: &crate::nn::StatePath,
        _: &mut V,
    ) -> Result<()> {
        Ok(())
    }
}

impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitStateMut<B> for Upsample {
    fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        _: &crate::nn::StatePath,
        _: &mut V,
    ) -> Result<()> {
        Ok(())
    }
}

impl crate::nn::NamedLayers for Upsample {
    fn layer_structure(&self, prefix: &str) -> alloc::vec::Vec<crate::nn::LayerNode> {
        alloc::vec![crate::nn::LayerNode {
            name: alloc::string::String::from(prefix),
            type_name: alloc::string::String::from("Upsample"),
            shape_info: alloc::format!("mode={:?}", self.mode),
            children: alloc::vec![],
        }]
    }
}

impl crate::nn::ShapeInfo for Upsample {
    fn shape_info(&self) -> Option<alloc::string::String> {
        None
    }
}

impl<B: crate::tensor::backend::VariableBackend, NewD: crate::tensor::device::Device>
    crate::tensor::transfer::ToDevice<B, NewD> for Upsample
{
    type Output = Upsample;
    fn to_device(self, _arg: &NewD::Arg) -> Result<Self::Output> {
        Ok(self)
    }
}

impl crate::nn::TrainMode for Upsample {}

impl Upsample {
    /// Refuses bilinear mode: no bilinear op exists in the catalog.
    fn check_mode(mode: UpsampleMode) -> Result<()> {
        if mode == UpsampleMode::Bilinear {
            return Err(Error::InvalidModuleState {
                operation: "upsample",
                reason: ErrorMessage::from(
                    "bilinear upsampling is unsupported: the operation catalog has no interpolate op",
                ),
            });
        }
        Ok(())
    }

    /// Rejects a zero scale factor before any backend runs.
    fn check_scale(scale: [usize; 2]) -> Result<()> {
        for factor in scale {
            if factor == 0 {
                return Err(Error::Shape(ShapeError::InvalidParameter {
                    operation: OperationKind::RepeatInterleave,
                    parameter: "scale_factor",
                    value: factor,
                }));
            }
        }
        Ok(())
    }

    /// Nearest-neighbor upsampling with per-axis integer scale factors.
    pub fn nearest(scale_h: usize, scale_w: usize) -> Result<Self> {
        Self::check_scale([scale_h, scale_w])?;
        Ok(Self {
            mode: UpsampleMode::Nearest,
            scale_factor: Some([scale_h, scale_w]),
            size: None,
        })
    }

    /// Nearest-neighbor upsampling to target spatial extents.
    pub fn nearest_size(out_h: usize, out_w: usize) -> Result<Self> {
        Self::check_scale([out_h, out_w])?;
        Ok(Self {
            mode: UpsampleMode::Nearest,
            scale_factor: None,
            size: Some([out_h, out_w]),
        })
    }

    /// General constructor over an explicit mode. Bilinear is refused.
    pub fn new(mode: UpsampleMode, scale_h: usize, scale_w: usize) -> Result<Self> {
        Self::check_mode(mode)?;
        Self::check_scale([scale_h, scale_w])?;
        Ok(Self {
            mode,
            scale_factor: Some([scale_h, scale_w]),
            size: None,
        })
    }

    /// General size-form constructor over an explicit mode. Bilinear is refused.
    pub fn with_size(mode: UpsampleMode, out_h: usize, out_w: usize) -> Result<Self> {
        Self::check_mode(mode)?;
        Self::check_scale([out_h, out_w])?;
        Ok(Self {
            mode,
            scale_factor: None,
            size: Some([out_h, out_w]),
        })
    }

    /// Resolves per-axis integer repeat factors for an input with the given
    /// spatial extents.
    fn factors(&self, spatial: &[usize]) -> Result<[usize; 2]> {
        Self::check_mode(self.mode)?;
        match (self.scale_factor, self.size) {
            (Some(scale), None) => {
                Self::check_scale(scale)?;
                Ok(scale)
            }
            (None, Some(size)) => {
                if spatial.len() != 2 {
                    return Err(Error::Shape(ShapeError::InvalidParameter {
                        operation: OperationKind::RepeatInterleave,
                        parameter: "size",
                        value: spatial.len(),
                    }));
                }
                let mut factors = [1usize; 2];
                for (i, (&target, &input)) in size.iter().zip(spatial.iter()).enumerate() {
                    if target < input || input == 0 || target % input != 0 {
                        return Err(Error::Shape(ShapeError::InvalidParameter {
                            operation: OperationKind::RepeatInterleave,
                            parameter: "size",
                            value: target,
                        }));
                    }
                    factors[i] = target / input;
                }
                Ok(factors)
            }
            _ => Err(Error::InvalidModuleState {
                operation: "upsample",
                reason: ErrorMessage::from("upsample needs exactly one of scale_factor or size"),
            }),
        }
    }
}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend, K: DType, L: Layout<S>>
    Module<Tensor<S, B, K, NoGrad, Local, L>> for Upsample
where
    B: Capabilities + Execute<op::RepeatInterleave>,
    <B as Execute<op::RepeatInterleave>>::Output: Into<B::Storage<K>>,
{
    /// `Dense<Dyn>`: the output extents multiply the input's by runtime factors.
    type Output = Dense<Dyn, B, K, NoGrad, Local>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<S, B, K, NoGrad, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        let dims = x.dims();
        let rank = dims.as_ref().len();
        if rank != 3 && rank != 4 {
            return Err(Error::Shape(ShapeError::RankMismatch {
                operation: OperationKind::RepeatInterleave,
                expected: RankExpectation::Between { min: 3, max: 4 },
                actual: rank,
            }));
        }
        let spatial: alloc::vec::Vec<usize> = dims.as_ref()[rank - 2..].to_vec();
        let factors = self.factors(&spatial)?;
        if rank == 3 && factors[0] != 1 {
            // A rank-3 input has one spatial axis; a height factor has
            // nothing to apply to, so refusing beats silently dropping it.
            return Err(Error::Shape(ShapeError::InvalidParameter {
                operation: OperationKind::RepeatInterleave,
                parameter: "scale_factor",
                value: factors[0],
            }));
        }
        // Factors are validated non-zero, so dispatching a factor of one is
        // a plain copy on every backend (only zero is refused) and keeps
        // both branches at the same `Dense<Dyn>` type.
        if rank == 4 {
            Ok(x.repeat_interleave(factors[0], -2)?
                .repeat_interleave(factors[1], -1)?)
        } else {
            Ok(x.repeat_interleave(factors[1], -1)?)
        }
    }
}
