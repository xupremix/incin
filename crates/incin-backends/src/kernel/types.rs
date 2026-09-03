use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Family of kernels sharing launch structure.
pub enum KernelFamily {
    /// One input, element-wise compute.
    PointwiseUnary,
    /// Two inputs combined element-wise.
    PointwiseBinary,
    /// Collapses axes into aggregates.
    Reduction,
    /// Normalizes over a feature set.
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
/// How a kernel accesses its inputs.
pub enum KernelAccess {
    /// Scalar access with an unroll width.
    Scalar {
        /// How many scalar iterations to unroll.
        unroll_width: u8,
    },
    /// Packed vector access with a vector width.
    Packed {
        /// SIMD lanes per packed load.
        vector_width: u8,
    },
    /// Warp-level reduction pattern.
    WarpReduction,
    /// Welford-style online variance pattern.
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
/// Stable cache key identifying one compiled kernel variant.
pub struct KernelKey {
    schema_version: u8,
    /// Kernel family of this variant.
    pub family: KernelFamily,
    /// Operation name the kernel implements.
    pub operation: String,
    storage: KernelDType,
    compute: KernelDType,
    pub(super) accumulator: KernelDType,
    output: KernelDType,
    /// Layout class the kernel was specialized for.
    pub layout: LayoutClass,
    /// Input access pattern of this variant.
    pub access: KernelAccess,
    pub(crate) index_width: KernelIndexWidth,
    /// Math mode this variant was compiled under.
    pub math_mode: MathMode,
    /// Rank class the shape was bucketed into.
    pub rank_class: RankClass,
    /// Shape bucket used for tile selection.
    pub shape_bucket: ShapeBucket,
    /// Alignment class of the operands.
    pub alignment: AlignmentClass,
}

impl KernelKey {
    /// Builds a CUDA kernel key from launch parameters.
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
    /// Builds a CUDA kernel key including an explicit signature.
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

    /// Canonical string id used by the persistent tuning cache.
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
    /// Stable problem id binding this kernel to its tuning record.
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

/// What the frontend proved about an operand's geometry, as far as kernel
/// specialization is concerned.
///
/// This is the backend-side projection of `incin_core::exec::ShapeEvidence`,
/// which rides along on every `Validated` descriptor and reports what
/// `Shape::PROOF` established from the shape *type* rather than from a runtime
/// measurement. The distinction is the whole point. A launcher always knows the
/// runtime element count, and could always specialize on it -- but doing so
/// would compile a fresh kernel per observed shape and turn the module cache
/// into a leak. A *statically* known count is a constant of the program, so the
/// number of specializations it can produce is bounded by the number of shape
/// types the program instantiates, which is finite and small.
///
/// So the proof is not a faster way to learn the numel. It is the thing that
/// makes specializing on the numel affordable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg(any(feature = "cuda", test))]
pub(crate) struct KernelSpecialization {
    /// Element count known before runtime, from the shape type.
    pub(crate) static_numel: Option<usize>,
    /// Per-axis extents the shape type settled, outermost first.
    ///
    /// Empty when the rank was not known. Otherwise rank-long, with a hole for
    /// each axis whose extent is only a runtime fact.
    pub(crate) static_extents: &'static [Option<usize>],
}

#[cfg(any(feature = "cuda", test))]
impl KernelSpecialization {
    /// The "nothing was proven" value, which specializes nothing.
    pub(crate) const NONE: Self = Self {
        static_numel: None,
        static_extents: &[],
    };

    /// Projects the facts this module can act on out of a frontend proof.
    ///
    /// `None` evidence means the call arrived through a path that carries no
    /// shape type -- `dispatch::execute` is generic over the operation but not
    /// over operand shapes, so it reports `Dynamic` for everything -- and is
    /// treated exactly like a dynamic proof.
    // The only caller is the CUDA executor, so a `test` build without the
    // `cuda` feature compiles this and reaches nothing.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn from_evidence(evidence: Option<incin_core::exec::ShapeEvidence>) -> Self {
        let Some(evidence) = evidence else {
            return Self::NONE;
        };
        // A `Mixed` proof has a static rank but at least one runtime axis, so
        // its numel is not a constant even when `static_numel` is populated.
        // Only `Static` licenses baking the count into the source.
        Self {
            // Only `Static` licenses a whole-shape element count; a `Mixed`
            // proof has a static rank but at least one runtime axis.
            static_numel: if evidence.proof().is_static() {
                evidence.static_numel()
            } else {
                None
            },
            // Extents are per-axis and already carry a hole where the type
            // settled nothing, so a `Mixed` shape can still contribute the axes
            // it does know.
            static_extents: evidence.static_extents(),
        }
    }

    /// Whether a packed kernel's ragged-tail branch is provably unreachable.
    ///
    /// The packed templates handle a trailing partial vector with a scalar
    /// `else` branch. When the element count is a proven multiple of the vector
    /// width, no packet is ever partial, so that branch is dead code: emitting
    /// it costs instruction cache and forces the compiler to keep a divergent
    /// path it can never take.
    pub(crate) fn packed_tail_is_dead(self, vector_width: u8) -> bool {
        let width = usize::from(vector_width);
        width != 0 && self.static_numel.is_some_and(|n| n % width == 0)
    }

    /// The extents to unroll a strided index computation over, if every axis is
    /// known *and* agrees with the shape the kernel will actually iterate.
    ///
    /// The agreement check is not paranoia. The proof describes the descriptor's
    /// output geometry, while the kernel iterates the iteration plan's output
    /// shape; these should be the same, but "should" is not a basis for emitting
    /// literal divisors. A mismatch silently produces wrong addresses rather
    /// than a failure, so a disagreement declines to specialize instead.
    // Reached only from the CUDA launcher, so a `test` build without the
    // `cuda` feature compiles this and calls nothing.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn unrollable_extents(self, iterated: &[usize]) -> Option<alloc::vec::Vec<usize>> {
        if self.static_extents.len() != iterated.len() || iterated.is_empty() {
            return None;
        }
        let mut extents = alloc::vec::Vec::with_capacity(iterated.len());
        for (proven, actual) in self.static_extents.iter().zip(iterated) {
            // A hole means this axis is a runtime fact; the loop needs every
            // divisor to be a literal, so one hole disqualifies the whole
            // unroll.
            if (*proven)? != *actual {
                return None;
            }
            extents.push(*actual);
        }
        Some(extents)
    }
}

/// The module-cache identity for a kernel, scoped to the text it compiles.
///
/// `KernelKey::cache_id` describes the *problem* a kernel solves -- family,
/// operation name, dtypes, layout, access pattern -- and deliberately says
/// nothing about the kernel body, because that is what makes it usable as a
/// tuning identity. It is not sufficient as a compilation identity: several
/// callers format a runtime value directly into their expression while passing a
/// fixed operation name, so `powf(x, 2.0)` and `powf(x, 3.0)` are both rendered
/// under `"powf"`, and `x * 0.25` and `x * 0.5` are both rendered under
/// `"mul_scalar"`. Keyed on the problem alone, the first of each pair to be
/// compiled would be served to the second, which is a silent wrong answer rather
/// than a miss.
///
/// Mixing a digest of the source into the key makes bodies that differ compile
/// separately and bodies that match continue to share, which is what the cache
/// is for. `tuning_problem_id` is left alone, so autotuning still groups the
/// variants as one problem.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn source_scoped_cache_id(key: &KernelKey, source: &str) -> String {
    format!("{}/src={:016x}", key.cache_id(), fnv1a64(source))
}

/// FNV-1a, 64-bit. Chosen for being short, dependency-free and stable across
/// runs; this is a cache discriminator, not a security boundary.
#[cfg(any(feature = "cuda", test))]
fn fnv1a64(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
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
