use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelFamily {
    PointwiseUnary,
    PointwiseBinary,
    Reduction,
    Normalization,
}

impl KernelFamily {
    fn tag(self) -> &'static str {
        match self {
            Self::PointwiseUnary => "pointwise-unary",
            Self::PointwiseBinary => "pointwise-binary",
            Self::Reduction => "reduction",
            Self::Normalization => "normalization",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelAccess {
    Scalar { unroll_width: u8 },
    Packed { vector_width: u8 },
    WarpReduction,
    Welford,
}

impl KernelAccess {
    fn tag(self) -> String {
        match self {
            Self::Scalar { unroll_width } => format!("scalar-u{unroll_width}"),
            Self::Packed { vector_width } => format!("packed-v{vector_width}"),
            Self::WarpReduction => "warp-reduction".into(),
            Self::Welford => "welford".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum KernelDType {
    U8,
    U32,
    I64,
    BF16,
    F16,
    F32,
    F64,
    Q8_0,
}

impl KernelDType {
    fn from_id(dtype: DTypeId) -> Result<Self> {
        match dtype {
            DTypeId::U8 => Ok(Self::U8),
            DTypeId::U32 => Ok(Self::U32),
            DTypeId::I64 => Ok(Self::I64),
            DTypeId::BF16 => Ok(Self::BF16),
            DTypeId::F16 => Ok(Self::F16),
            DTypeId::F32 => Ok(Self::F32),
            DTypeId::F64 => Ok(Self::F64),
            DTypeId::Q8_0 => Ok(Self::Q8_0),
            _ => Err(Error::Msg(format!(
                "dtype {dtype:?} has no kernel-key encoding"
            ))),
        }
    }

    fn from_descriptor(dtype: incin_core::tensor::dtype::DTypeDescriptor) -> Result<Self> {
        let id = dtype.builtin_id().ok_or_else(|| {
            Error::Msg(format!(
                "custom dtype {:?} has no kernel-key encoding",
                dtype
            ))
        })?;
        Self::from_id(id)
    }

    fn tag(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Q8_0 => "q8_0",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum KernelIndexWidth {
    I32,
}

impl KernelIndexWidth {
    fn tag(self) -> &'static str {
        match self {
            Self::I32 => "i32",
        }
    }
}

use crate::tuning::signature::{AlignmentClass, RankClass, ShapeBucket};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelKey {
    schema_version: u8,
    pub family: KernelFamily,
    pub operation: String,
    storage: KernelDType,
    compute: KernelDType,
    pub(super) accumulator: KernelDType,
    output: KernelDType,
    pub layout: LayoutClass,
    pub access: KernelAccess,
    pub(crate) index_width: KernelIndexWidth,
    pub math_mode: MathMode,
    pub rank_class: RankClass,
    pub shape_bucket: ShapeBucket,
    pub alignment: AlignmentClass,
}

impl KernelKey {
    pub fn cuda(
        _policy_family: OperationKind,
        family: KernelFamily,
        operation: &str,
        dtype: DTypeId,
        layout: LayoutClass,
        access: KernelAccess,
    ) -> Result<Self> {
        Self::cuda_with_signature(
            _policy_family,
            family,
            operation,
            dtype,
            layout,
            access,
            RankClass::Vector,
            ShapeBucket::from_numel(1024),
            AlignmentClass::Align256,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_with_signature(
        _policy_family: OperationKind,
        family: KernelFamily,
        operation: &str,
        dtype: DTypeId,
        layout_class: LayoutClass,
        access: KernelAccess,
        rank_class: RankClass,
        shape_bucket: ShapeBucket,
        alignment: AlignmentClass,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let policy = {
            let req = PrecisionRequest::new(
                _policy_family,
                dtype.descriptor(),
                dtype.descriptor(),
                layout_class,
                1,
                false,
                MathMode::Fast,
            );
            crate::cuda::backend::native_precision(&req)?
        };
        #[cfg(not(feature = "cuda"))]
        let policy = {
            let compute = if matches!(dtype, DTypeId::F16 | DTypeId::BF16) {
                DTypeId::F32.descriptor()
            } else {
                dtype.descriptor()
            };
            incin_core::exec::ResolvedPrecision::new(
                dtype.descriptor(),
                compute,
                compute,
                dtype.descriptor(),
                incin_core::exec::LossScaling::None,
            )
        };
        Ok(Self {
            schema_version: KERNEL_KEY_SCHEMA_VERSION,
            family,
            operation: operation.into(),
            storage: KernelDType::from_descriptor(policy.storage)?,
            compute: KernelDType::from_descriptor(policy.compute)?,
            accumulator: KernelDType::from_descriptor(policy.accumulator)?,
            output: KernelDType::from_descriptor(policy.output)?,
            layout: layout_class,
            access,
            index_width: KernelIndexWidth::I32,
            math_mode: MathMode::Precise,
            rank_class,
            shape_bucket,
            alignment,
        })
    }

    pub fn cache_id(&self) -> String {
        format!(
            "k{}/cuda/{}/{}/s={}/c={}/a={}/o={}/layout={}/access={}/index={}/math={}/rank={}/bucket={}/align={}",
            self.schema_version,
            self.family.tag(),
            self.operation,
            self.storage.tag(),
            self.compute.tag(),
            self.accumulator.tag(),
            self.output.tag(),
            self.layout.as_str(),
            self.access.tag(),
            self.index_width.tag(),
            self.math_mode.as_str(),
            self.rank_class.tag(),
            self.shape_bucket.tag(),
            self.alignment.tag(),
        )
    }

    #[cfg(any(feature = "autotune", test))]
    pub fn tuning_problem_id(&self) -> String {
        format!(
            "k{}/cuda/{}/{}/s={}/c={}/a={}/o={}/layout={}/index={}/math={}/rank={}/bucket={}/align={}",
            self.schema_version,
            self.family.tag(),
            self.operation,
            self.storage.tag(),
            self.compute.tag(),
            self.accumulator.tag(),
            self.output.tag(),
            self.layout.as_str(),
            self.index_width.tag(),
            self.math_mode.as_str(),
            self.rank_class.tag(),
            self.shape_bucket.tag(),
            self.alignment.tag(),
        )
    }
}

/// A rendered kernel and the identity used by the backend module cache.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(any(feature = "cuda", test))]
pub(crate) struct RenderedKernel {
    pub(crate) entry_point: String,
    pub(crate) cache_key: String,
    pub(crate) source: String,
    pub(crate) dtype: DTypeId,
    pub(crate) element_size: usize,
    pub(crate) unroll_width: u8,
    pub(crate) vector_width: u8,
    pub(crate) key: KernelKey,
}

#[cfg(any(feature = "cuda", test))]
impl RenderedKernel {
    pub(crate) fn elements_per_thread(&self) -> u8 {
        self.unroll_width.max(self.vector_width)
    }
}
