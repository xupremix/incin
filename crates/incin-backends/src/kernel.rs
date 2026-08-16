//! Internal kernel specialization vocabulary and source rendering.
//!
//! Kernels are described once and specialized lazily by dtype. This keeps
//! source maintenance proportional to operation families rather than to the
//! Cartesian product of operations, dtypes, layouts, and devices.

use alloc::{boxed::Box, string::String};
use incin_core::error::{Error, Result};
#[cfg(feature = "cuda")]
use incin_core::exec::PrecisionRequest;
use incin_core::exec::{LayoutClass, MathMode};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DTypeId;
const KERNEL_KEY_SCHEMA_VERSION: u8 = 1;

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
enum KernelDType {
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
// I64 is part of the stable key vocabulary before the large-index renderer is enabled.
#[allow(dead_code)]
pub(crate) enum KernelIndexWidth {
    I32,
    I64,
}

impl KernelIndexWidth {
    fn tag(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
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
    accumulator: KernelDType,
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

#[derive(Debug, Clone, Copy)]
#[cfg(any(feature = "cuda", test))]
struct CudaScalarSpec {
    suffix: &'static str,
    storage_type: &'static str,
    compute_type: &'static str,
    preamble: &'static str,
    load_prefix: &'static str,
    load_suffix: &'static str,
    store_prefix: &'static str,
    store_suffix: &'static str,
    element_size: usize,
}

#[cfg(any(feature = "cuda", test))]
impl CudaScalarSpec {
    fn for_float(dtype: DTypeId, op: &'static str) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let policy = {
            let req = PrecisionRequest::new(
                OperationKind::Pointwise,
                dtype.descriptor(),
                dtype.descriptor(),
                LayoutClass::Contiguous,
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
        match dtype {
            DTypeId::F16 => Ok(Self {
                suffix: "f16",
                storage_type: "__half",
                compute_type: "float",
                preamble: "#include <cuda_fp16.h>",
                load_prefix: "__half2float(",
                load_suffix: ")",
                store_prefix: "__float2half_rn(",
                store_suffix: ")",
                element_size: 2,
            }),
            DTypeId::BF16 => Ok(Self {
                suffix: "bf16",
                storage_type: "__nv_bfloat16",
                compute_type: "float",
                preamble: "#include <cuda_bf16.h>",
                load_prefix: "__bfloat162float(",
                load_suffix: ")",
                store_prefix: "__float2bfloat16_rn(",
                store_suffix: ")",
                element_size: 2,
            }),
            DTypeId::F32 => Ok(Self {
                suffix: "f32",
                storage_type: "float",
                compute_type: "float",
                preamble: "",
                load_prefix: "",
                load_suffix: "",
                store_prefix: "",
                store_suffix: "",
                element_size: 4,
            }),
            DTypeId::F64 => Ok(Self {
                suffix: "f64",
                storage_type: "double",
                compute_type: "double",
                preamble: "",
                load_prefix: "",
                load_suffix: "",
                store_prefix: "",
                store_suffix: "",
                element_size: 8,
            }),
            _ => Err(Error::UnsupportedDType {
                dtype: dtype.descriptor(),
                backend: "Cuda",
                op,
            }),
        }
        .inspect(|_| {
            debug_assert_eq!(
                policy.compute,
                if matches!(dtype, DTypeId::F16 | DTypeId::BF16) {
                    DTypeId::F32.descriptor()
                } else {
                    dtype.descriptor()
                }
            );
        })
    }
}

#[cfg(any(feature = "cuda", test))]
fn validate_identifier(identifier: &str) -> Result<()> {
    let mut chars = identifier.chars();
    let valid_start = chars
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic());
    if valid_start && chars.all(|character| character == '_' || character.is_ascii_alphanumeric()) {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "invalid kernel operation identifier: {identifier:?}"
        )))
    }
}

#[cfg(any(feature = "cuda", test))]
fn render_cuda(
    template: &str,
    family: &str,
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
    unroll_width: u8,
) -> Result<RenderedKernel> {
    if !matches!(unroll_width, 1 | 2 | 4) {
        return Err(Error::Msg(format!(
            "unsupported CUDA pointwise unroll width {unroll_width}"
        )));
    }
    validate_identifier(op_name)?;
    let scalar = CudaScalarSpec::for_float(dtype, "render_elementwise")?;
    let (key_family, layout) = match family {
        "elementwise_unary" => (KernelFamily::PointwiseUnary, LayoutClass::Strided),
        "elementwise_unary_contiguous" => (KernelFamily::PointwiseUnary, LayoutClass::Contiguous),
        "elementwise_binary" => (KernelFamily::PointwiseBinary, LayoutClass::Strided),
        "elementwise_binary_contiguous" => (KernelFamily::PointwiseBinary, LayoutClass::Contiguous),
        "elementwise_binary_scalar_left" => {
            (KernelFamily::PointwiseBinary, LayoutClass::ScalarLeft)
        }
        "elementwise_binary_scalar_right" => {
            (KernelFamily::PointwiseBinary, LayoutClass::ScalarRight)
        }
        _ => {
            return Err(Error::Msg(format!(
                "unknown CUDA pointwise kernel family {family:?}"
            )));
        }
    };
    let key = KernelKey::cuda(
        OperationKind::Pointwise,
        key_family,
        op_name,
        dtype,
        layout,
        KernelAccess::Scalar { unroll_width },
    )?;
    let entry_point = format!("incin_{family}_{}_u{unroll_width}_{op_name}", scalar.suffix);
    let source = template
        .replace("{PREAMBLE}", scalar.preamble)
        .replace("{ENTRY_POINT}", &entry_point)
        .replace("{STORAGE_TYPE}", scalar.storage_type)
        .replace("{COMPUTE_TYPE}", scalar.compute_type)
        .replace("{LOAD_PREFIX}", scalar.load_prefix)
        .replace("{LOAD_SUFFIX}", scalar.load_suffix)
        .replace("{STORE_PREFIX}", scalar.store_prefix)
        .replace("{STORE_SUFFIX}", scalar.store_suffix)
        .replace("{UNROLL_WIDTH}", &unroll_width.to_string())
        .replace("{OP}", op_expr);

    Ok(RenderedKernel {
        cache_key: key.cache_id(),
        entry_point,
        source,
        dtype,
        element_size: scalar.element_size,
        unroll_width,
        vector_width: 1,
        key,
    })
}

#[derive(Debug, Clone, Copy)]
#[cfg(any(feature = "cuda", test))]
struct CudaPackedSpec {
    scalar: CudaScalarSpec,
    storage_type: &'static str,
    compute_type: &'static str,
    width: u8,
    unpack_prefix: &'static str,
    unpack_suffix: &'static str,
    store_prefix: &'static str,
    store_suffix: &'static str,
    components: &'static [&'static str],
}

#[cfg(any(feature = "cuda", test))]
impl CudaPackedSpec {
    fn for_float(dtype: DTypeId) -> Result<Self> {
        let scalar = CudaScalarSpec::for_float(dtype, "render_packed_elementwise")?;
        let packed = match dtype {
            DTypeId::F16 => Self {
                scalar,
                storage_type: "__half2",
                compute_type: "float2",
                width: 2,
                unpack_prefix: "__half22float2(",
                unpack_suffix: ")",
                store_prefix: "__floats2half2_rn(",
                store_suffix: ")",
                components: &["x", "y"],
            },
            DTypeId::BF16 => Self {
                scalar,
                storage_type: "__nv_bfloat162",
                compute_type: "float2",
                width: 2,
                unpack_prefix: "__bfloat1622float2(",
                unpack_suffix: ")",
                store_prefix: "__floats2bfloat162_rn(",
                store_suffix: ")",
                components: &["x", "y"],
            },
            DTypeId::F32 => Self {
                scalar,
                storage_type: "float4",
                compute_type: "float4",
                width: 4,
                unpack_prefix: "",
                unpack_suffix: "",
                store_prefix: "",
                store_suffix: "",
                components: &["x", "y", "z", "w"],
            },
            DTypeId::F64 => Self {
                scalar,
                storage_type: "double2",
                compute_type: "double2",
                width: 2,
                unpack_prefix: "",
                unpack_suffix: "",
                store_prefix: "",
                store_suffix: "",
                components: &["x", "y"],
            },
            _ => unreachable!("CudaScalarSpec already rejected non-float dtype"),
        };
        Ok(packed)
    }

    fn packed_store_expr(self, value: &str) -> String {
        match self.width {
            2 if self.scalar.element_size == 2 => format!(
                "{}{value}.x, {value}.y{}",
                self.store_prefix, self.store_suffix
            ),
            _ => value.into(),
        }
    }
}

#[cfg(any(feature = "cuda", test))]
#[allow(clippy::too_many_arguments)]
fn render_cuda_packed(
    family: &str,
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
    arguments: &str,
    packed_loads: &str,
    lane_bindings: impl Fn(&str) -> String,
    scalar_tail: &str,
) -> Result<RenderedKernel> {
    validate_identifier(op_name)?;
    let packed = CudaPackedSpec::for_float(dtype)?;
    let (key_family, layout) = match family {
        "elementwise_unary_contiguous" => (KernelFamily::PointwiseUnary, LayoutClass::Contiguous),
        "elementwise_binary_contiguous" => (KernelFamily::PointwiseBinary, LayoutClass::Contiguous),
        "elementwise_binary_scalar_left" => {
            (KernelFamily::PointwiseBinary, LayoutClass::ScalarLeft)
        }
        "elementwise_binary_scalar_right" => {
            (KernelFamily::PointwiseBinary, LayoutClass::ScalarRight)
        }
        _ => {
            return Err(Error::Msg(format!(
                "unknown packed CUDA pointwise kernel family {family:?}"
            )));
        }
    };
    let key = KernelKey::cuda(
        OperationKind::Pointwise,
        key_family,
        op_name,
        dtype,
        layout,
        KernelAccess::Packed {
            vector_width: packed.width,
        },
    )?;
    let entry_point = format!(
        "incin_{family}_{}_v{}_{op_name}",
        packed.scalar.suffix, packed.width
    );
    let mut lanes = String::new();
    for component in packed.components {
        lanes.push_str("        {\n");
        lanes.push_str(&lane_bindings(component));
        lanes.push_str(&format!(
            "            {} out_val = {op_expr};\n            packed_output.{component} = out_val;\n",
            packed.scalar.compute_type
        ));
        lanes.push_str("        }\n");
    }
    let packed_store = packed.packed_store_expr("packed_output");
    let source = format!(
        r#"
{preamble}
extern "C" __global__ void {entry_point}(
{arguments}
) {{
    int packet_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = packet_idx * {width};
    if (base >= numel) {{
        return;
    }}
    if (base + {width} <= numel) {{
{packed_loads}
        {packed_compute_type} packed_output;
{lanes}
        reinterpret_cast<{packed_storage_type}*>(output)[packet_idx] = {packed_store};
    }} else {{
        #pragma unroll
        for (int lane = 0; lane < {width}; lane++) {{
            int idx = base + lane;
            if (idx < numel) {{
{scalar_tail}
            }}
        }}
    }}
}}
"#,
        preamble = packed.scalar.preamble,
        width = packed.width,
        packed_compute_type = packed.compute_type,
        packed_storage_type = packed.storage_type,
    );
    Ok(RenderedKernel {
        cache_key: key.cache_id(),
        entry_point,
        source,
        dtype,
        element_size: packed.scalar.element_size,
        unroll_width: 1,
        vector_width: packed.width,
        key,
    })
}

#[cfg(any(feature = "cuda", test))]
const CUDA_UNARY_TEMPLATE: &str = r#"
{PREAMBLE}
extern "C" __global__ void {ENTRY_POINT}(
    const {STORAGE_TYPE}* input,
    {STORAGE_TYPE}* output,
    const int* shape,
    const int* strides,
    int offset,
    int numel,
    int ndim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int flat_idx = offset;
        int temp = idx;
        for (int i = ndim - 1; i >= 0; i--) {
            int dim_idx = temp % shape[i];
            temp /= shape[i];
            flat_idx += dim_idx * strides[i];
        }
        {COMPUTE_TYPE} x = {LOAD_PREFIX}input[flat_idx]{LOAD_SUFFIX};
        {COMPUTE_TYPE} out_val = {OP};
        output[idx] = {STORE_PREFIX}out_val{STORE_SUFFIX};
    }
}
"#;

#[cfg(any(feature = "cuda", test))]
const CUDA_BINARY_TEMPLATE: &str = r#"
{PREAMBLE}
extern "C" __global__ void {ENTRY_POINT}(
    const {STORAGE_TYPE}* lhs,
    const {STORAGE_TYPE}* rhs,
    {STORAGE_TYPE}* output,
    const int* out_shape,
    const int* lhs_strides,
    const int* rhs_strides,
    int lhs_offset,
    int rhs_offset,
    int numel,
    int ndim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int temp = idx;
        int lhs_flat = lhs_offset;
        int rhs_flat = rhs_offset;
        for (int i = ndim - 1; i >= 0; i--) {
            int dim_idx = temp % out_shape[i];
            temp /= out_shape[i];
            lhs_flat += dim_idx * lhs_strides[i];
            rhs_flat += dim_idx * rhs_strides[i];
        }
        {COMPUTE_TYPE} a = {LOAD_PREFIX}lhs[lhs_flat]{LOAD_SUFFIX};
        {COMPUTE_TYPE} b = {LOAD_PREFIX}rhs[rhs_flat]{LOAD_SUFFIX};
        {COMPUTE_TYPE} out_val = {OP};
        output[idx] = {STORE_PREFIX}out_val{STORE_SUFFIX};
    }
}
"#;

#[cfg(any(feature = "cuda", test))]
const CUDA_UNARY_CONTIGUOUS_TEMPLATE: &str = r#"
{PREAMBLE}
extern "C" __global__ void {ENTRY_POINT}(
    const {STORAGE_TYPE}* input,
    {STORAGE_TYPE}* output,
    int offset,
    int numel
) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * {UNROLL_WIDTH};
    #pragma unroll
    for (int lane = 0; lane < {UNROLL_WIDTH}; lane++) {
        int idx = base + lane;
        if (idx < numel) {
            {COMPUTE_TYPE} x = {LOAD_PREFIX}input[offset + idx]{LOAD_SUFFIX};
            {COMPUTE_TYPE} out_val = {OP};
            output[idx] = {STORE_PREFIX}out_val{STORE_SUFFIX};
        }
    }
}
"#;

#[cfg(any(feature = "cuda", test))]
const CUDA_BINARY_DENSE_TEMPLATE: &str = r#"
{PREAMBLE}
extern "C" __global__ void {ENTRY_POINT}(
    const {STORAGE_TYPE}* lhs,
    const {STORAGE_TYPE}* rhs,
    {STORAGE_TYPE}* output,
    int lhs_offset,
    int rhs_offset,
    int numel
) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * {UNROLL_WIDTH};
    #pragma unroll
    for (int lane = 0; lane < {UNROLL_WIDTH}; lane++) {
        int idx = base + lane;
        if (idx < numel) {
            {COMPUTE_TYPE} a = {LOAD_PREFIX}lhs[{LHS_INDEX}]{LOAD_SUFFIX};
            {COMPUTE_TYPE} b = {LOAD_PREFIX}rhs[{RHS_INDEX}]{LOAD_SUFFIX};
            {COMPUTE_TYPE} out_val = {OP};
            output[idx] = {STORE_PREFIX}out_val{STORE_SUFFIX};
        }
    }
}
"#;

#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_unary(
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
) -> Result<RenderedKernel> {
    render_cuda(
        CUDA_UNARY_TEMPLATE,
        "elementwise_unary",
        op_name,
        op_expr,
        dtype,
        1,
    )
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_unary_for_layout(
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
    layout: LayoutClass,
    unroll_width: u8,
) -> Result<RenderedKernel> {
    match layout {
        LayoutClass::Contiguous => render_cuda(
            CUDA_UNARY_CONTIGUOUS_TEMPLATE,
            "elementwise_unary_contiguous",
            op_name,
            op_expr,
            dtype,
            unroll_width,
        ),
        LayoutClass::Strided if unroll_width == 1 => render_cuda_unary(op_name, op_expr, dtype),
        LayoutClass::Strided => Err(Error::Msg(
            "strided CUDA unary kernels require unroll width 1".into(),
        )),
        other => Err(Error::Msg(format!(
            "layout {} is not valid for a CUDA unary kernel",
            other.as_str()
        ))),
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_binary(
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
) -> Result<RenderedKernel> {
    render_cuda(
        CUDA_BINARY_TEMPLATE,
        "elementwise_binary",
        op_name,
        op_expr,
        dtype,
        1,
    )
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_binary_for_layout(
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
    layout: LayoutClass,
    unroll_width: u8,
) -> Result<RenderedKernel> {
    let (family, lhs_index, rhs_index) = match layout {
        LayoutClass::Contiguous => (
            "elementwise_binary_contiguous",
            "lhs_offset + idx",
            "rhs_offset + idx",
        ),
        LayoutClass::ScalarLeft => (
            "elementwise_binary_scalar_left",
            "lhs_offset",
            "rhs_offset + idx",
        ),
        LayoutClass::ScalarRight => (
            "elementwise_binary_scalar_right",
            "lhs_offset + idx",
            "rhs_offset",
        ),
        LayoutClass::Strided if unroll_width == 1 => {
            return render_cuda_binary(op_name, op_expr, dtype);
        }
        LayoutClass::Strided => {
            return Err(Error::Msg(
                "strided CUDA binary kernels require unroll width 1".into(),
            ));
        }
        other => {
            return Err(Error::Msg(format!(
                "layout {} is not valid for a CUDA binary kernel",
                other.as_str()
            )));
        }
    };
    let template = CUDA_BINARY_DENSE_TEMPLATE
        .replace("{LHS_INDEX}", lhs_index)
        .replace("{RHS_INDEX}", rhs_index);
    render_cuda(&template, family, op_name, op_expr, dtype, unroll_width)
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_unary_packed(
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
    layout: LayoutClass,
) -> Result<RenderedKernel> {
    if layout != LayoutClass::Contiguous {
        return Err(Error::Msg(
            "packed CUDA unary kernels require contiguous input".into(),
        ));
    }
    let packed = CudaPackedSpec::for_float(dtype)?;
    let scalar = packed.scalar;
    let arguments = format!(
        "    const {}* input,\n    {}* output,\n    int offset,\n    int numel",
        scalar.storage_type, scalar.storage_type
    );
    let packed_loads = format!(
        "        const {packed_storage} input_storage = reinterpret_cast<const {packed_storage}*>(input + offset)[packet_idx];\n        const {packed_compute} packed_input = {unpack_prefix}input_storage{unpack_suffix};",
        packed_storage = packed.storage_type,
        packed_compute = packed.compute_type,
        unpack_prefix = packed.unpack_prefix,
        unpack_suffix = packed.unpack_suffix,
    );
    let scalar_tail = format!(
        "                {compute} x = {load_prefix}input[offset + idx]{load_suffix};\n                {compute} out_val = {op_expr};\n                output[idx] = {store_prefix}out_val{store_suffix};",
        compute = scalar.compute_type,
        load_prefix = scalar.load_prefix,
        load_suffix = scalar.load_suffix,
        store_prefix = scalar.store_prefix,
        store_suffix = scalar.store_suffix,
    );
    render_cuda_packed(
        "elementwise_unary_contiguous",
        op_name,
        op_expr,
        dtype,
        &arguments,
        &packed_loads,
        |component| {
            format!(
                "            {} x = packed_input.{component};\n",
                scalar.compute_type
            )
        },
        &scalar_tail,
    )
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_binary_packed(
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
    layout: LayoutClass,
) -> Result<RenderedKernel> {
    let packed = CudaPackedSpec::for_float(dtype)?;
    let scalar = packed.scalar;
    let (family, lhs_index, rhs_index, packed_loads, lane_bindings) = match layout {
        LayoutClass::Contiguous => {
            let loads = format!(
                "        const {storage} lhs_storage = reinterpret_cast<const {storage}*>(lhs + lhs_offset)[packet_idx];\n        const {storage} rhs_storage = reinterpret_cast<const {storage}*>(rhs + rhs_offset)[packet_idx];\n        const {compute} packed_lhs = {unpack_prefix}lhs_storage{unpack_suffix};\n        const {compute} packed_rhs = {unpack_prefix}rhs_storage{unpack_suffix};",
                storage = packed.storage_type,
                compute = packed.compute_type,
                unpack_prefix = packed.unpack_prefix,
                unpack_suffix = packed.unpack_suffix,
            );
            let bindings = move |component: &str| {
                format!(
                    "            {compute} a = packed_lhs.{component};\n            {compute} b = packed_rhs.{component};\n",
                    compute = scalar.compute_type
                )
            };
            (
                "elementwise_binary_contiguous",
                "lhs_offset + idx",
                "rhs_offset + idx",
                loads,
                Box::new(bindings) as Box<dyn Fn(&str) -> String>,
            )
        }
        LayoutClass::ScalarLeft => {
            let loads = format!(
                "        const {scalar_compute} scalar_lhs = {load_prefix}lhs[lhs_offset]{load_suffix};\n        const {storage} rhs_storage = reinterpret_cast<const {storage}*>(rhs + rhs_offset)[packet_idx];\n        const {packed_compute} packed_rhs = {unpack_prefix}rhs_storage{unpack_suffix};",
                scalar_compute = scalar.compute_type,
                load_prefix = scalar.load_prefix,
                load_suffix = scalar.load_suffix,
                storage = packed.storage_type,
                packed_compute = packed.compute_type,
                unpack_prefix = packed.unpack_prefix,
                unpack_suffix = packed.unpack_suffix,
            );
            let bindings = move |component: &str| {
                format!(
                    "            {compute} a = scalar_lhs;\n            {compute} b = packed_rhs.{component};\n",
                    compute = scalar.compute_type
                )
            };
            (
                "elementwise_binary_scalar_left",
                "lhs_offset",
                "rhs_offset + idx",
                loads,
                Box::new(bindings) as Box<dyn Fn(&str) -> String>,
            )
        }
        LayoutClass::ScalarRight => {
            let loads = format!(
                "        const {storage} lhs_storage = reinterpret_cast<const {storage}*>(lhs + lhs_offset)[packet_idx];\n        const {packed_compute} packed_lhs = {unpack_prefix}lhs_storage{unpack_suffix};\n        const {scalar_compute} scalar_rhs = {load_prefix}rhs[rhs_offset]{load_suffix};",
                storage = packed.storage_type,
                packed_compute = packed.compute_type,
                unpack_prefix = packed.unpack_prefix,
                unpack_suffix = packed.unpack_suffix,
                scalar_compute = scalar.compute_type,
                load_prefix = scalar.load_prefix,
                load_suffix = scalar.load_suffix,
            );
            let bindings = move |component: &str| {
                format!(
                    "            {compute} a = packed_lhs.{component};\n            {compute} b = scalar_rhs;\n",
                    compute = scalar.compute_type
                )
            };
            (
                "elementwise_binary_scalar_right",
                "lhs_offset + idx",
                "rhs_offset",
                loads,
                Box::new(bindings) as Box<dyn Fn(&str) -> String>,
            )
        }
        LayoutClass::Strided => {
            return Err(Error::Msg(
                "packed CUDA binary kernels require dense or scalar-broadcast input".into(),
            ));
        }
        other => {
            return Err(Error::Msg(format!(
                "layout {} is not valid for a packed CUDA binary kernel",
                other.as_str()
            )));
        }
    };
    let arguments = format!(
        "    const {}* lhs,\n    const {}* rhs,\n    {}* output,\n    int lhs_offset,\n    int rhs_offset,\n    int numel",
        scalar.storage_type, scalar.storage_type, scalar.storage_type
    );
    let scalar_tail = format!(
        "                {compute} a = {load_prefix}lhs[{lhs_index}]{load_suffix};\n                {compute} b = {load_prefix}rhs[{rhs_index}]{load_suffix};\n                {compute} out_val = {op_expr};\n                output[idx] = {store_prefix}out_val{store_suffix};",
        compute = scalar.compute_type,
        load_prefix = scalar.load_prefix,
        load_suffix = scalar.load_suffix,
        store_prefix = scalar.store_prefix,
        store_suffix = scalar.store_suffix,
    );
    render_cuda_packed(
        family,
        op_name,
        op_expr,
        dtype,
        &arguments,
        &packed_loads,
        lane_bindings,
        &scalar_tail,
    )
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_reduction(
    op_name: &str,
    dtype: DTypeId,
    with_indices: bool,
    contiguous_last_axis: bool,
) -> Result<RenderedKernel> {
    let scalar = CudaScalarSpec::for_float(dtype, "render_reduction")?;
    #[cfg(feature = "cuda")]
    let policy = {
        let req = PrecisionRequest::new(
            incin_core::shapes::error::OperationKind::Reduction,
            dtype.descriptor(),
            dtype.descriptor(),
            LayoutClass::Contiguous,
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
    debug_assert_eq!(
        policy.accumulator,
        if matches!(dtype, DTypeId::F16 | DTypeId::BF16) {
            DTypeId::F32.descriptor()
        } else {
            dtype.descriptor()
        }
    );
    let (init, update, finish) = match op_name {
        "sum" => ("0.0", "acc += value;", "acc"),
        "mean" => ("0.0", "acc += value;", "acc / (ACC_TYPE)reduce_dim_size"),
        "max" => (
            "(-1.0 / 0.0)",
            "if (value > acc) { acc = value; best_idx = (unsigned int)in_flat; }",
            "acc",
        ),
        "min" => (
            "(1.0 / 0.0)",
            "if (value < acc) { acc = value; best_idx = (unsigned int)in_flat; }",
            "acc",
        ),
        _ => {
            return Err(Error::Msg(format!(
                "unsupported CUDA reduction operation {op_name:?}"
            )));
        }
    };
    if with_indices && !matches!(op_name, "max" | "min") {
        return Err(Error::Msg(format!(
            "CUDA reduction {op_name:?} does not produce indices"
        )));
    }
    if with_indices && contiguous_last_axis {
        return Err(Error::Msg(
            "indexed CUDA reductions do not yet use the block-parallel path".into(),
        ));
    }

    let layout = if contiguous_last_axis {
        "contiguous_last_axis"
    } else {
        "strided"
    };
    let index_suffix = if with_indices { "_indices" } else { "" };
    let entry_point = format!(
        "incin_reduce_{layout}_{}_{}{}",
        scalar.suffix, op_name, index_suffix
    );
    let key = KernelKey::cuda(
        OperationKind::Reduction,
        KernelFamily::Reduction,
        &format!("{op_name}{index_suffix}"),
        dtype,
        if contiguous_last_axis {
            LayoutClass::ContiguousLastAxis
        } else {
            LayoutClass::Strided
        },
        if contiguous_last_axis {
            KernelAccess::WarpReduction
        } else {
            KernelAccess::Scalar { unroll_width: 1 }
        },
    )?;
    let cache_key = key.cache_id();

    let source = if contiguous_last_axis {
        let combine = match op_name {
            "sum" | "mean" => "acc += other;",
            "max" => "if (other > acc) acc = other;",
            "min" => "if (other < acc) acc = other;",
            _ => unreachable!(),
        };
        format!(
            r#"
{preamble}
extern "C" __global__ void {entry_point}(
    const {storage_type}* __restrict__ input,
    {storage_type}* __restrict__ output,
    int in_offset,
    int reduce_dim_size,
    int out_numel)
{{
    int out_idx = blockIdx.x;
    if (out_idx >= out_numel) return;
    int tid = threadIdx.x;
    extern __shared__ unsigned char shared_raw[];
    {compute_type}* shared = reinterpret_cast<{compute_type}*>(shared_raw);
    {compute_type} acc = ({compute_type})({init});
    int row_start = in_offset + out_idx * reduce_dim_size;
    for (int i = tid; i < reduce_dim_size; i += blockDim.x) {{
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        {fast_update}
    }}
    unsigned int active = __activemask();
    for (int delta = 16; delta > 0; delta >>= 1) {{
        {compute_type} other = __shfl_down_sync(active, acc, delta);
        {combine}
    }}
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) shared[warp] = acc;
    __syncthreads();
    if (warp == 0) {{
        int warp_count = (blockDim.x + 31) >> 5;
        acc = lane < warp_count ? shared[lane] : ({compute_type})({init});
        active = __activemask();
        for (int delta = 16; delta > 0; delta >>= 1) {{
            {compute_type} other = __shfl_down_sync(active, acc, delta);
            {combine}
        }}
        if (lane == 0) {{
            {compute_type} out_value = {finish};
            output[out_idx] = {store_prefix}out_value{store_suffix};
        }}
    }}
}}
"#,
            preamble = scalar.preamble,
            storage_type = scalar.storage_type,
            compute_type = scalar.compute_type,
            fast_update = match op_name {
                "sum" | "mean" => "acc += value;",
                "max" => "if (value > acc) acc = value;",
                "min" => "if (value < acc) acc = value;",
                _ => unreachable!(),
            },
            load_prefix = scalar.load_prefix,
            load_suffix = scalar.load_suffix,
            store_prefix = scalar.store_prefix,
            store_suffix = scalar.store_suffix,
            finish = finish.replace("ACC_TYPE", scalar.compute_type),
        )
    } else {
        let index_argument = if with_indices {
            "    unsigned int* __restrict__ out_indices,\n"
        } else {
            ""
        };
        let best_index_decl = if with_indices {
            "    unsigned int best_idx = 0;\n"
        } else {
            ""
        };
        let index_store = if with_indices {
            "    out_indices[out_flat] = best_idx;\n"
        } else {
            ""
        };
        let update = if with_indices {
            update
        } else {
            match op_name {
                "max" => "if (value > acc) acc = value;",
                "min" => "if (value < acc) acc = value;",
                _ => update,
            }
        };
        format!(
            r#"
{preamble}
extern "C" __global__ void {entry_point}(
    const {storage_type}* __restrict__ input,
    {storage_type}* __restrict__ output,
{index_argument}    const int* __restrict__ in_strides,
    const int* __restrict__ out_shape,
    const int* __restrict__ out_strides,
    int in_offset,
    int out_offset,
    int reduce_axis,
    int reduce_dim_size,
    int ndim,
    int out_numel)
{{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_numel) return;
    int temp = out_idx;
    int out_flat = out_offset;
    int base_in_flat = in_offset;
    for (int i = ndim - 1; i >= 0; i--) {{
        int dim_idx = temp % out_shape[i];
        temp /= out_shape[i];
        out_flat += dim_idx * out_strides[i];
        if (i != reduce_axis) base_in_flat += dim_idx * in_strides[i];
    }}
    {compute_type} acc = ({compute_type})({init});
{best_index_decl}    for (int i = 0; i < reduce_dim_size; i++) {{
        int in_flat = base_in_flat + i * in_strides[reduce_axis];
        {compute_type} value = {load_prefix}input[in_flat]{load_suffix};
        {update}
    }}
    {compute_type} out_value = {finish};
    output[out_flat] = {store_prefix}out_value{store_suffix};
{index_store}}}
"#,
            preamble = scalar.preamble,
            storage_type = scalar.storage_type,
            compute_type = scalar.compute_type,
            load_prefix = scalar.load_prefix,
            load_suffix = scalar.load_suffix,
            store_prefix = scalar.store_prefix,
            store_suffix = scalar.store_suffix,
            finish = finish.replace("ACC_TYPE", scalar.compute_type),
        )
    };

    Ok(RenderedKernel {
        entry_point,
        cache_key,
        source,
        dtype,
        element_size: scalar.element_size,
        unroll_width: 1,
        vector_width: 1,
        key,
    })
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_normalization(op_name: &str, dtype: DTypeId) -> Result<RenderedKernel> {
    let scalar = CudaScalarSpec::for_float(dtype, "render_normalization")?;
    #[cfg(feature = "cuda")]
    let policy = {
        let req = PrecisionRequest::new(
            incin_core::shapes::error::OperationKind::Normalization,
            dtype.descriptor(),
            dtype.descriptor(),
            LayoutClass::Contiguous,
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
    debug_assert_eq!(policy.accumulator, policy.compute);
    let entry_point = format!("incin_normalization_{}_{}", scalar.suffix, op_name);
    let key = KernelKey::cuda(
        OperationKind::Normalization,
        KernelFamily::Normalization,
        op_name,
        dtype,
        if op_name == "layer_norm" {
            LayoutClass::RowWise
        } else {
            LayoutClass::ChannelWise
        },
        if op_name == "layer_norm" {
            KernelAccess::Welford
        } else {
            KernelAccess::Scalar { unroll_width: 1 }
        },
    )?;
    let cache_key = key.cache_id();
    let inverse_std = if dtype == DTypeId::F64 {
        "1.0 / sqrt(variance + (double)eps)"
    } else {
        "rsqrtf(variance + eps)"
    };
    let source = match op_name {
        "layer_norm" => format!(
            r#"
{preamble}
struct IncinWelford {{
    {compute_type} mean;
    {compute_type} m2;
    int count;
}};

__device__ __forceinline__ IncinWelford incin_welford_combine(
    IncinWelford left, IncinWelford right)
{{
    if (right.count == 0) return left;
    if (left.count == 0) return right;
    int count = left.count + right.count;
    {compute_type} delta = right.mean - left.mean;
    {compute_type} right_ratio = ({compute_type})right.count / ({compute_type})count;
    IncinWelford combined;
    combined.mean = left.mean + delta * right_ratio;
    combined.m2 = left.m2 + right.m2 + delta * delta
        * (({compute_type})left.count * ({compute_type})right.count / ({compute_type})count);
    combined.count = count;
    return combined;
}}

extern "C" __global__ void {entry_point}(
    const {storage_type}* __restrict__ input,
    const {storage_type}* __restrict__ gamma,
    const {storage_type}* __restrict__ beta,
    {storage_type}* __restrict__ output,
    float eps,
    int norm_size,
    int has_bias,
    int batch_size,
    int input_offset,
    int gamma_offset,
    int beta_offset)
{{
    int row = blockIdx.x;
    if (row >= batch_size) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    IncinWelford local = {{({compute_type})0.0, ({compute_type})0.0, 0}};
    int row_start = input_offset + row * norm_size;
    for (int i = tid; i < norm_size; i += blockDim.x) {{
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        local.count += 1;
        {compute_type} delta = value - local.mean;
        local.mean += delta / ({compute_type})local.count;
        {compute_type} delta2 = value - local.mean;
        local.m2 += delta * delta2;
    }}
    unsigned int active = __activemask();
    for (int delta = 16; delta > 0; delta >>= 1) {{
        IncinWelford other;
        other.mean = __shfl_down_sync(active, local.mean, delta);
        other.m2 = __shfl_down_sync(active, local.m2, delta);
        other.count = __shfl_down_sync(active, local.count, delta);
        if (lane + delta < 32) local = incin_welford_combine(local, other);
    }}
    extern __shared__ unsigned char shared_raw[];
    int warp_count = (blockDim.x + 31) >> 5;
    {compute_type}* shared_mean = reinterpret_cast<{compute_type}*>(shared_raw);
    {compute_type}* shared_m2 = shared_mean + warp_count;
    int* shared_count = reinterpret_cast<int*>(shared_m2 + warp_count);
    if (lane == 0) {{
        shared_mean[warp] = local.mean;
        shared_m2[warp] = local.m2;
        shared_count[warp] = local.count;
    }}
    __syncthreads();
    if (warp == 0) {{
        local.mean = lane < warp_count ? shared_mean[lane] : ({compute_type})0.0;
        local.m2 = lane < warp_count ? shared_m2[lane] : ({compute_type})0.0;
        local.count = lane < warp_count ? shared_count[lane] : 0;
        active = __activemask();
        for (int delta = 16; delta > 0; delta >>= 1) {{
            IncinWelford other;
            other.mean = __shfl_down_sync(active, local.mean, delta);
            other.m2 = __shfl_down_sync(active, local.m2, delta);
            other.count = __shfl_down_sync(active, local.count, delta);
            if (lane + delta < 32) local = incin_welford_combine(local, other);
        }}
        if (lane == 0) {{
            shared_mean[0] = local.mean;
            shared_m2[0] = local.m2 / ({compute_type})local.count;
        }}
    }}
    __syncthreads();
    {compute_type} mean = shared_mean[0];
    {compute_type} variance = shared_m2[0];
    {compute_type} inverse_std = {inverse_std};
    for (int i = tid; i < norm_size; i += blockDim.x) {{
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        {compute_type} scale = {load_prefix}gamma[gamma_offset + i]{load_suffix};
        {compute_type} shift = has_bias
            ? {load_prefix}beta[beta_offset + i]{load_suffix}
            : ({compute_type})0.0;
        {compute_type} normalized = (value - mean) * inverse_std;
        output[row * norm_size + i] = {store_prefix}(normalized * scale + shift){store_suffix};
    }}
}}
"#,
            preamble = scalar.preamble,
            compute_type = scalar.compute_type,
            storage_type = scalar.storage_type,
            load_prefix = scalar.load_prefix,
            load_suffix = scalar.load_suffix,
            store_prefix = scalar.store_prefix,
            store_suffix = scalar.store_suffix,
        ),
        "batch_norm" => format!(
            r#"
{preamble}
extern "C" __global__ void {entry_point}(
    const {storage_type}* __restrict__ input,
    const {storage_type}* __restrict__ weight,
    const {storage_type}* __restrict__ bias,
    const {storage_type}* __restrict__ running_mean,
    const {storage_type}* __restrict__ running_variance,
    {storage_type}* __restrict__ output,
    float eps,
    int num_channels,
    int spatial_size,
    int total_elements,
    int has_weight,
    int has_bias,
    int has_running_mean,
    int has_running_variance,
    int input_offset,
    int weight_offset,
    int bias_offset,
    int mean_offset,
    int variance_offset)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int channel = (idx / spatial_size) % num_channels;
    {compute_type} mean = has_running_mean
        ? {load_prefix}running_mean[mean_offset + channel]{load_suffix}
        : ({compute_type})0.0;
    {compute_type} variance = has_running_variance
        ? {load_prefix}running_variance[variance_offset + channel]{load_suffix}
        : ({compute_type})1.0;
    {compute_type} scale = has_weight
        ? {load_prefix}weight[weight_offset + channel]{load_suffix}
        : ({compute_type})1.0;
    {compute_type} shift = has_bias
        ? {load_prefix}bias[bias_offset + channel]{load_suffix}
        : ({compute_type})0.0;
    {compute_type} value = {load_prefix}input[input_offset + idx]{load_suffix};
    {compute_type} inverse_std = {inverse_std};
    {compute_type} normalized = (value - mean) * inverse_std;
    output[idx] = {store_prefix}(normalized * scale + shift){store_suffix};
}}
"#,
            preamble = scalar.preamble,
            compute_type = scalar.compute_type,
            storage_type = scalar.storage_type,
            load_prefix = scalar.load_prefix,
            load_suffix = scalar.load_suffix,
            store_prefix = scalar.store_prefix,
            store_suffix = scalar.store_suffix,
        ),
        _ => {
            return Err(Error::Msg(format!(
                "unsupported CUDA normalization operation {op_name:?}"
            )));
        }
    };
    Ok(RenderedKernel {
        entry_point,
        cache_key,
        source,
        dtype,
        element_size: scalar.element_size,
        unroll_width: 1,
        vector_width: 1,
        key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_keys_separate_binary_specializations_from_tuning_problems() {
        let scalar = KernelKey::cuda(
            OperationKind::Pointwise,
            KernelFamily::PointwiseUnary,
            "neg",
            DTypeId::F32,
            LayoutClass::Contiguous,
            KernelAccess::Scalar { unroll_width: 4 },
        )
        .unwrap();
        let mut packed = scalar.clone();
        packed.access = KernelAccess::Packed { vector_width: 4 };
        assert_ne!(scalar.cache_id(), packed.cache_id());
        assert_eq!(scalar.tuning_problem_id(), packed.tuning_problem_id());

        let mut strided = scalar.clone();
        strided.layout = LayoutClass::Strided;
        assert_ne!(scalar.cache_id(), strided.cache_id());
        assert_ne!(scalar.tuning_problem_id(), strided.tuning_problem_id());

        let mut wider_accumulator = scalar.clone();
        wider_accumulator.accumulator = KernelDType::F64;
        assert_ne!(scalar.cache_id(), wider_accumulator.cache_id());
        assert_ne!(
            scalar.tuning_problem_id(),
            wider_accumulator.tuning_problem_id()
        );
        assert!(
            scalar
                .cache_id()
                .starts_with("k1/cuda/pointwise-unary/neg/")
        );
    }

    #[test]
    fn cuda_float_specializations_share_a_template_but_not_cache_keys() {
        let f16 = render_cuda_unary("relu", "x > 0.0f ? x : 0.0f", DTypeId::F16).unwrap();
        let f32 = render_cuda_unary("relu", "x > 0.0f ? x : 0.0f", DTypeId::F32).unwrap();
        let f64 = render_cuda_unary("relu", "x > 0.0 ? x : 0.0", DTypeId::F64).unwrap();

        assert_ne!(f16.cache_key, f32.cache_key);
        assert_ne!(f32.cache_key, f64.cache_key);
        assert_eq!(f16.dtype, DTypeId::F16);
        assert_eq!(f32.dtype, DTypeId::F32);
        assert_eq!(f64.dtype, DTypeId::F64);
        assert_eq!(
            f16.element_size,
            DTypeId::F16.encoding().scalar_bytes().unwrap()
        );
        assert_eq!(
            f32.element_size,
            DTypeId::F32.encoding().scalar_bytes().unwrap()
        );
        assert_eq!(
            f64.element_size,
            DTypeId::F64.encoding().scalar_bytes().unwrap()
        );
        assert!(f16.source.contains("const __half* input"));
        assert!(f16.source.contains("__half2float(input[flat_idx])"));
        assert!(f16.source.contains("__float2half_rn(out_val)"));
        assert!(f32.source.contains("const float* input"));
        assert!(f64.source.contains("const double* input"));
    }

    #[test]
    fn cuda_bfloat16_uses_f32_compute_and_bfloat16_storage() {
        let rendered = render_cuda_binary("add", "a + b", DTypeId::BF16).unwrap();
        assert_eq!(rendered.element_size, 2);
        assert!(rendered.source.contains("#include <cuda_bf16.h>"));
        assert!(rendered.source.contains("const __nv_bfloat16* lhs"));
        assert!(rendered.source.contains("float a = __bfloat162float"));
        assert!(rendered.source.contains("__float2bfloat16_rn(out_val)"));
        assert!(!rendered.source.contains("lhs_shape"));
        assert!(!rendered.source.contains("rhs_shape"));
    }

    #[test]
    fn renderer_rejects_non_float_dtypes_and_invalid_identifiers() {
        assert!(matches!(
            render_cuda_unary("relu", "x", DTypeId::U32),
            Err(Error::UnsupportedDType { .. })
        ));
        assert!(render_cuda_unary("relu;bad", "x", DTypeId::F32).is_err());
    }

    #[test]
    fn layout_specializations_share_expressions_but_have_distinct_abis_and_keys() {
        let contiguous =
            render_cuda_binary_for_layout("sub", "a - b", DTypeId::F32, LayoutClass::Contiguous, 4)
                .unwrap();
        let scalar_left =
            render_cuda_binary_for_layout("sub", "a - b", DTypeId::F32, LayoutClass::ScalarLeft, 2)
                .unwrap();
        let strided =
            render_cuda_binary_for_layout("sub", "a - b", DTypeId::F32, LayoutClass::Strided, 1)
                .unwrap();

        assert_ne!(contiguous.cache_key, scalar_left.cache_key);
        assert_ne!(contiguous.cache_key, strided.cache_key);
        assert!(contiguous.source.contains("lhs[lhs_offset + idx]"));
        assert!(scalar_left.source.contains("lhs[lhs_offset]"));
        assert!(contiguous.source.contains("lane < 4"));
        assert!(contiguous.source.contains("if (idx < numel)"));
        assert_eq!(contiguous.unroll_width, 4);
        assert_eq!(contiguous.vector_width, 1);
        assert_eq!(contiguous.elements_per_thread(), 4);
        assert_eq!(
            contiguous.key.access,
            KernelAccess::Scalar { unroll_width: 4 }
        );
        assert!(contiguous.cache_key.contains("access=scalar-u4"));
        assert!(!contiguous.source.contains("out_shape"));
        assert!(strided.source.contains("out_shape"));

        let unary =
            render_cuda_unary_for_layout("neg", "-x", DTypeId::BF16, LayoutClass::Contiguous, 2)
                .unwrap();
        assert!(unary.source.contains("input[offset + idx]"));
        assert!(!unary.source.contains("strides"));
        assert_eq!(unary.unroll_width, 2);
        assert!(
            render_cuda_unary_for_layout("neg", "-x", DTypeId::F32, LayoutClass::Strided, 4,)
                .is_err()
        );
    }

    #[test]
    fn packed_templates_use_vector_storage_and_mask_scalar_tails() {
        let unary =
            render_cuda_unary_packed("neg", "-x", DTypeId::F32, LayoutClass::Contiguous).unwrap();
        assert_eq!(unary.unroll_width, 1);
        assert_eq!(unary.vector_width, 4);
        assert_eq!(unary.elements_per_thread(), 4);
        assert_eq!(unary.key.access, KernelAccess::Packed { vector_width: 4 });
        assert!(unary.cache_key.contains("access=packed-v4"));
        assert!(
            unary
                .source
                .contains("reinterpret_cast<const float4*>(input + offset)[packet_idx]")
        );
        assert!(unary.source.contains("if (base + 4 <= numel)"));
        assert!(unary.source.contains("input[offset + idx]"));

        let half = render_cuda_binary_packed("add", "a + b", DTypeId::F16, LayoutClass::Contiguous)
            .unwrap();
        assert_eq!(half.vector_width, 2);
        assert!(half.source.contains("const __half2 lhs_storage"));
        assert!(half.source.contains("__half22float2(lhs_storage)"));
        assert!(
            half.source
                .contains("__floats2half2_rn(packed_output.x, packed_output.y)")
        );

        let scalar_left =
            render_cuda_binary_packed("sub", "a - b", DTypeId::BF16, LayoutClass::ScalarLeft)
                .unwrap();
        assert!(scalar_left.source.contains("const float scalar_lhs"));
        assert!(
            scalar_left
                .source
                .contains("const __nv_bfloat162 rhs_storage")
        );
        assert!(scalar_left.source.contains("a = scalar_lhs"));
        assert!(
            render_cuda_binary_packed("add", "a + b", DTypeId::F32, LayoutClass::Strided,).is_err()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a locally installed NVRTC shared library"]
    fn packed_templates_compile_with_nvrtc_for_every_float_family() {
        for dtype in [DTypeId::F16, DTypeId::BF16, DTypeId::F32, DTypeId::F64] {
            let unary =
                render_cuda_unary_packed("neg", "-x", dtype, LayoutClass::Contiguous).unwrap();
            crate::cuda::gpu::compile_ptx_with_cuda_includes(&unary.source)
                .unwrap_or_else(|error| panic!("NVRTC rejected packed unary {dtype:?}: {error:?}"));

            for layout in [
                LayoutClass::Contiguous,
                LayoutClass::ScalarLeft,
                LayoutClass::ScalarRight,
            ] {
                let binary = render_cuda_binary_packed("add", "a + b", dtype, layout).unwrap();
                crate::cuda::gpu::compile_ptx_with_cuda_includes(&binary.source).unwrap_or_else(
                    |error| panic!("NVRTC rejected packed binary {dtype:?}/{layout:?}: {error:?}"),
                );
            }
        }
    }

    #[test]
    fn reduction_templates_share_structure_and_apply_accumulator_policy() {
        let half_fast = render_cuda_reduction("sum", DTypeId::F16, false, true).unwrap();
        assert_eq!(half_fast.key.layout, LayoutClass::ContiguousLastAxis);
        assert_eq!(half_fast.key.access, KernelAccess::WarpReduction);
        assert!(half_fast.cache_key.contains("/reduction/sum/s=f16"));
        assert!(half_fast.cache_key.contains("layout=contiguous-last-axis"));
        assert!(
            half_fast
                .source
                .contains("const __half* __restrict__ input")
        );
        assert!(half_fast.source.contains("float* shared"));
        assert!(
            half_fast
                .source
                .contains("__half2float(input[row_start + i])")
        );
        assert!(half_fast.source.contains("__float2half_rn(out_value)"));
        assert!(half_fast.source.contains("__shfl_down_sync"));
        assert!(half_fast.source.contains("shared[warp] = acc"));

        let double_mean = render_cuda_reduction("mean", DTypeId::F64, false, false).unwrap();
        assert_eq!(double_mean.key.layout, LayoutClass::Strided);
        assert!(double_mean.cache_key.contains("/reduction/mean/s=f64"));
        assert!(double_mean.source.contains("double acc"));
        assert!(double_mean.source.contains("acc / (double)reduce_dim_size"));
        assert!(!double_mean.source.contains("out_indices"));

        let indexed = render_cuda_reduction("max", DTypeId::BF16, true, false).unwrap();
        assert!(
            indexed
                .source
                .contains("unsigned int* __restrict__ out_indices")
        );
        assert!(indexed.source.contains("unsigned int best_idx"));
        assert!(indexed.source.contains("__bfloat162float(input[in_flat])"));
        assert!(indexed.source.contains("out_indices[out_flat] = best_idx"));
        assert!(render_cuda_reduction("sum", DTypeId::F32, true, false).is_err());
        assert!(render_cuda_reduction("unknown", DTypeId::F32, false, false).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a locally installed NVRTC shared library"]
    fn reduction_templates_compile_with_nvrtc_for_every_float_family() {
        for dtype in [DTypeId::F16, DTypeId::BF16, DTypeId::F32, DTypeId::F64] {
            for op in ["sum", "mean", "max", "min"] {
                for fast in [false, true] {
                    let kernel = render_cuda_reduction(op, dtype, false, fast).unwrap();
                    crate::cuda::gpu::compile_ptx_with_cuda_includes(&kernel.source)
                        .unwrap_or_else(|error| {
                            panic!("NVRTC rejected reduction {dtype:?}/{op}/fast={fast}: {error:?}")
                        });
                }
            }
            for op in ["max", "min"] {
                let kernel = render_cuda_reduction(op, dtype, true, false).unwrap();
                crate::cuda::gpu::compile_ptx_with_cuda_includes(&kernel.source).unwrap_or_else(
                    |error| panic!("NVRTC rejected indexed reduction {dtype:?}/{op}: {error:?}"),
                );
            }
        }
    }

    #[test]
    fn normalization_templates_use_welford_and_dtype_specific_compute() {
        let half = render_cuda_normalization("layer_norm", DTypeId::F16).unwrap();
        assert!(half.source.contains("struct IncinWelford"));
        assert!(half.source.contains("__shfl_down_sync"));
        assert!(half.source.contains("float mean"));
        assert!(half.source.contains("__half2float(input[row_start + i])"));
        assert!(
            half.source
                .contains("__float2half_rn((normalized * scale + shift))")
        );

        let double = render_cuda_normalization("layer_norm", DTypeId::F64).unwrap();
        assert!(double.source.contains("1.0 / sqrt(variance + (double)eps)"));
        assert!(double.source.contains("double m2"));

        let bfloat_batch = render_cuda_normalization("batch_norm", DTypeId::BF16).unwrap();
        assert!(
            bfloat_batch
                .source
                .contains("const __nv_bfloat16* __restrict__ input")
        );
        assert!(
            bfloat_batch
                .source
                .contains("__float2bfloat16_rn((normalized * scale + shift))")
        );
        assert!(render_cuda_normalization("unknown", DTypeId::F32).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a locally installed NVRTC shared library"]
    fn normalization_templates_compile_with_nvrtc_for_every_float_family() {
        for dtype in [DTypeId::F16, DTypeId::BF16, DTypeId::F32, DTypeId::F64] {
            for op in ["layer_norm", "batch_norm"] {
                let kernel = render_cuda_normalization(op, dtype).unwrap();
                crate::cuda::gpu::compile_ptx_with_cuda_includes(&kernel.source).unwrap_or_else(
                    |error| panic!("NVRTC rejected normalization {dtype:?}/{op}: {error:?}"),
                );
            }
        }
    }
}
