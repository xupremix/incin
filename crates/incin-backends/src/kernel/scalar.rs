// Every item below is `#[cfg(any(feature = "cuda", test))]`; on a
// non-cuda, non-test build this file compiles empty, so this import is
// unused there.
#[allow(unused_imports)]
use super::*;

#[derive(Debug, Clone, Copy)]
#[cfg(any(feature = "cuda", test))]
pub(super) struct CudaScalarSpec {
    pub(super) suffix: &'static str,
    pub(super) storage_type: &'static str,
    pub(super) compute_type: &'static str,
    pub(super) preamble: &'static str,
    pub(super) load_prefix: &'static str,
    pub(super) load_suffix: &'static str,
    pub(super) store_prefix: &'static str,
    pub(super) store_suffix: &'static str,
    pub(super) element_size: usize,
}

#[cfg(any(feature = "cuda", test))]
impl CudaScalarSpec {
    pub(super) fn for_float(dtype: DTypeId, op: &'static str) -> Result<Self> {
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
pub(super) fn validate_identifier(identifier: &str) -> Result<()> {
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
const CUDA_UNARY_TEMPLATE: &str = r#"
{PREAMBLE}
extern "C" __global__ void {ENTRY_POINT}(
    const {STORAGE_TYPE}* input,
    {STORAGE_TYPE}* output,
{SHAPE_PARAM}    const int* strides,
    int offset,
    int numel{NDIM_PARAM}
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int flat_idx = offset;
        int temp = idx;
{INDEX_CALC}
        {COMPUTE_TYPE} x = {LOAD_PREFIX}input[flat_idx]{LOAD_SUFFIX};
{PROLOGUE}        {COMPUTE_TYPE} out_val = {OP};
        output[idx] = {STORE_PREFIX}out_val{STORE_SUFFIX};
    }
}
"#;

/// The general runtime index walk: one modulo and one division per axis, both
/// by values read from device memory.
///
/// Integer division is among the most expensive operations on the device, and
/// a divisor that only exists in a buffer cannot be strength-reduced by the
/// compiler.
#[cfg(any(feature = "cuda", test))]
const DYNAMIC_INDEX_CALC: &str = "\
        for (int i = ndim - 1; i >= 0; i--) {
            int dim_idx = temp % shape[i];
            temp /= shape[i];
            flat_idx += dim_idx * strides[i];
        }";

/// The same walk with the extents substituted as literals.
///
/// Strides stay dynamic: they describe the *view*, not the shape, so a
/// transposed or narrowed tensor has strides the type cannot settle. Extents
/// are what the shape type knows, and they are the divisors -- which is the
/// half that matters, because nvcc lowers a division by a literal to a
/// multiply-and-shift while a division by a loaded value stays a division.
///
/// The loop is emitted fully unrolled rather than as a counted loop so every
/// divisor is a distinct constant at its use site.
#[cfg(any(feature = "cuda", test))]
fn static_index_calc(extents: &[usize]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "        // extents proven by the shape type: {extents:?}"
    );
    for axis in (0..extents.len()).rev() {
        let extent = extents[axis];
        let _ = writeln!(
            out,
            "        flat_idx += (temp % {extent}) * strides[{axis}];"
        );
        // The outermost axis consumes what is left, so its division is dead.
        // Its modulo is kept: dropping it would make correctness depend on
        // `numel` equalling the product of the extents, and the guard in
        // `unrollable_extents` deliberately proves agreement with the iterated
        // shape rather than with the element count.
        if axis > 0 {
            let _ = writeln!(out, "        temp /= {extent};");
        }
    }
    // Trim the trailing newline; the slot supplies its own line break.
    while out.ends_with('\n') {
        out.pop();
    }
    out
}

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
{PROLOGUE}        {COMPUTE_TYPE} out_val = {OP};
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
{PROLOGUE}            {COMPUTE_TYPE} out_val = {OP};
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
{PROLOGUE}            {COMPUTE_TYPE} out_val = {OP};
            output[idx] = {STORE_PREFIX}out_val{STORE_SUFFIX};
        }
    }
}
"#;

#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
fn render_cuda(
    template: &str,
    family: &str,
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
    unroll_width: u8,
) -> Result<RenderedKernel> {
    render_cuda_body(
        template,
        family,
        op_name,
        &ScalarFragment::literal(op_expr),
        dtype,
        unroll_width,
        None,
    )
}

/// The indentation the `{PROLOGUE}` slot's own line carries.
///
/// The slot sits at column zero and the result statement follows it on the same
/// line, so prologue statements have to supply the indentation the template
/// would otherwise have provided. Reading it back out of the template keeps the
/// two in step instead of hard-coding a width per template.
#[cfg(any(feature = "cuda", test))]
fn prologue_indent(template: &str) -> String {
    template
        .split("{PROLOGUE}")
        .nth(1)
        .map(|rest| rest.chars().take_while(|c| *c == ' ').collect())
        .unwrap_or_default()
}

/// Renders a pointwise template from a body that may carry its own bindings.
///
/// A hand-written expression arrives as a `ScalarFragment` with an empty
/// prologue and is emitted exactly as before. An expression lowered from the IR
/// arrives in SSA form, where the fragment's value names a temporary that its
/// prologue defines; see `crate::codegen::fragment`. Both share every other step
/// -- identifier validation, precision policy, entry point and cache key -- so
/// the paths cannot diverge in how a kernel is identified or compiled.
#[cfg(any(feature = "cuda", test))]
fn render_cuda_body(
    template: &str,
    family: &str,
    op_name: &str,
    body: &ScalarFragment,
    dtype: DTypeId,
    unroll_width: u8,
    index_calc: Option<alloc::vec::Vec<usize>>,
) -> Result<RenderedKernel> {
    let prologue = body.prologue_block(&prologue_indent(template));
    let prologue = prologue.as_str();
    let op_expr = body.value.as_str();
    // Only the strided templates carry an index walk; the dense ones address
    // linearly and leave the slot absent.
    // Baking the extents also retires the `shape` array and `ndim`: nothing in
    // the unrolled walk reads them, so the parameters come off the signature and
    // the launcher stops uploading a buffer per launch to feed them.
    let (index_calc, shape_param, ndim_param) = match &index_calc {
        Some(extents) => (static_index_calc(extents), "", ""),
        None => (
            DYNAMIC_INDEX_CALC.into(),
            "    const int* shape,\n",
            ",\n    int ndim",
        ),
    };
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
        .replace("{PROLOGUE}", prologue)
        .replace("{INDEX_CALC}", &index_calc)
        .replace("{SHAPE_PARAM}", shape_param)
        .replace("{NDIM_PARAM}", ndim_param)
        .replace("{OP}", op_expr);

    Ok(RenderedKernel {
        cache_key: source_scoped_cache_id(&key, &source),
        entry_point,
        source,
        dtype,
        element_size: scalar.element_size,
        unroll_width,
        vector_width: 1,
        key,
    })
}

#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
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
#[allow(dead_code)]
pub(crate) fn render_cuda_unary_for_layout(
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
    layout: LayoutClass,
    unroll_width: u8,
) -> Result<RenderedKernel> {
    render_cuda_unary_for_layout_body(
        op_name,
        &ScalarFragment::literal(op_expr),
        dtype,
        layout,
        unroll_width,
        None,
    )
}

/// The layout-dispatching unary renderer over a body that may carry bindings.
///
/// # Errors
///
/// Returns an error for a layout the unary templates do not cover, or when the
/// underlying template rendering fails.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_unary_for_layout_body(
    op_name: &str,
    body: &ScalarFragment,
    dtype: DTypeId,
    layout: LayoutClass,
    unroll_width: u8,
    index_extents: Option<alloc::vec::Vec<usize>>,
) -> Result<RenderedKernel> {
    match layout {
        LayoutClass::Contiguous => render_cuda_body(
            CUDA_UNARY_CONTIGUOUS_TEMPLATE,
            "elementwise_unary_contiguous",
            op_name,
            body,
            dtype,
            unroll_width,
            None,
        ),
        LayoutClass::Strided if unroll_width == 1 => render_cuda_body(
            CUDA_UNARY_TEMPLATE,
            "elementwise_unary",
            op_name,
            body,
            dtype,
            1,
            index_extents,
        ),
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
#[allow(dead_code)]
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
#[allow(dead_code)]
pub(crate) fn render_cuda_binary_for_layout(
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
    layout: LayoutClass,
    unroll_width: u8,
) -> Result<RenderedKernel> {
    render_cuda_binary_for_layout_body(
        op_name,
        &ScalarFragment::literal(op_expr),
        dtype,
        layout,
        unroll_width,
    )
}

/// The layout-dispatching binary renderer over a body that may carry bindings.
///
/// # Errors
///
/// Returns an error for a layout the binary templates do not cover, or when the
/// underlying template rendering fails.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_binary_for_layout_body(
    op_name: &str,
    body: &ScalarFragment,
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
            return render_cuda_body(
                CUDA_BINARY_TEMPLATE,
                "elementwise_binary",
                op_name,
                body,
                dtype,
                1,
                None,
            );
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
    render_cuda_body(&template, family, op_name, body, dtype, unroll_width, None)
}

/// The compute type an IR fragment for `dtype` must be lowered in.
///
/// `CudaScalarSpec` computes half-precision tensors in `float`, so the fragment
/// has to use single-precision math there; lowering against the storage type
/// would emit `__half` arithmetic the template never declares.
#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
fn fragment_compute_dtype(dtype: DTypeId) -> DTypeId {
    match dtype {
        DTypeId::F16 | DTypeId::BF16 => DTypeId::F32,
        other => other,
    }
}

/// Lowers an IR expression against the unary templates' operand name.
///
/// # Errors
///
/// Returns an error when the IR cannot be lowered for `dtype`.
#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
pub(crate) fn lower_unary_body(
    expr: &crate::codegen::ir::IrExpr,
    dtype: DTypeId,
) -> Result<ScalarFragment> {
    crate::codegen::lower_scalar(expr, &["x"], fragment_compute_dtype(dtype))
}

/// Lowers an IR expression against the binary templates' operand names.
///
/// # Errors
///
/// Returns an error when the IR cannot be lowered for `dtype`.
#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
pub(crate) fn lower_binary_body(
    expr: &crate::codegen::ir::IrExpr,
    dtype: DTypeId,
) -> Result<ScalarFragment> {
    crate::codegen::lower_scalar(expr, &["a", "b"], fragment_compute_dtype(dtype))
}
