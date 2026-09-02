// Every item below is `#[cfg(any(feature = "cuda", test))]`; on a
// non-cuda, non-test build this file compiles empty, so this import is
// unused there.
#[allow(unused_imports)]
use super::*;

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
    body: &ScalarFragment,
    dtype: DTypeId,
    arguments: &str,
    packed_loads: &str,
    lane_bindings: impl Fn(&str) -> String,
    scalar_tail: &str,
) -> Result<RenderedKernel> {
    let op_expr = body.value.as_str();
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
        // Each lane is its own `{ }` scope, so an SSA prologue's temporaries are
        // scoped per lane and cannot collide across the unrolled components.
        lanes.push_str(&body.prologue_block("            "));
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
        cache_key: source_scoped_cache_id(&key, &source),
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
#[allow(dead_code)]
pub(crate) fn render_cuda_unary_packed(
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
    layout: LayoutClass,
) -> Result<RenderedKernel> {
    render_cuda_unary_packed_body(op_name, &ScalarFragment::literal(op_expr), dtype, layout)
}

/// The packed unary renderer over a body that may carry its own bindings.
///
/// # Errors
///
/// Returns an error when `layout` is not contiguous, or when the packed
/// specification rejects `dtype`.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_unary_packed_body(
    op_name: &str,
    body: &ScalarFragment,
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
        "                {compute} x = {load_prefix}input[offset + idx]{load_suffix};\n{prologue}                {compute} out_val = {op_expr};\n                output[idx] = {store_prefix}out_val{store_suffix};",
        compute = scalar.compute_type,
        load_prefix = scalar.load_prefix,
        load_suffix = scalar.load_suffix,
        store_prefix = scalar.store_prefix,
        store_suffix = scalar.store_suffix,
        prologue = body.prologue_block("                "),
        op_expr = body.value,
    );
    render_cuda_packed(
        "elementwise_unary_contiguous",
        op_name,
        body,
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
#[allow(dead_code)]
pub(crate) fn render_cuda_binary_packed(
    op_name: &str,
    op_expr: &str,
    dtype: DTypeId,
    layout: LayoutClass,
) -> Result<RenderedKernel> {
    render_cuda_binary_packed_body(op_name, &ScalarFragment::literal(op_expr), dtype, layout)
}

/// The packed binary renderer over a body that may carry its own bindings.
///
/// # Errors
///
/// Returns an error when `layout` is strided or otherwise not packable, or when
/// the packed specification rejects `dtype`.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_binary_packed_body(
    op_name: &str,
    body: &ScalarFragment,
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
        "                {compute} a = {load_prefix}lhs[{lhs_index}]{load_suffix};\n                {compute} b = {load_prefix}rhs[{rhs_index}]{load_suffix};\n{prologue}                {compute} out_val = {op_expr};\n                output[idx] = {store_prefix}out_val{store_suffix};",
        compute = scalar.compute_type,
        load_prefix = scalar.load_prefix,
        load_suffix = scalar.load_suffix,
        store_prefix = scalar.store_prefix,
        store_suffix = scalar.store_suffix,
        prologue = body.prologue_block("                "),
        op_expr = body.value,
    );
    render_cuda_packed(
        family,
        op_name,
        body,
        dtype,
        &arguments,
        &packed_loads,
        lane_bindings,
        &scalar_tail,
    )
}
