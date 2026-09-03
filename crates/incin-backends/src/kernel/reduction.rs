// Every item below is `#[cfg(any(feature = "cuda", test))]`; on a
// non-cuda, non-test build this file compiles empty, so this import is
// unused there.
#[allow(unused_imports)]
use super::*;

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
        "prod" => ("1.0", "acc *= value;", "acc"),
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

    let source = if contiguous_last_axis {
        // `combine` folds one warp lane into another; `fast_update` folds one
        // loaded element into the accumulator. Both must cover exactly the set
        // of operations accepted above, and the two are written out separately
        // because the warp-shuffle path names its operand `other` and the load
        // path names it `value`. Keeping them as two matches over the same set
        // is what let `prod` be present in one and missing from the other.
        let combine = match op_name {
            "sum" | "mean" => "acc += other;",
            "max" => "if (other > acc) acc = other;",
            "min" => "if (other < acc) acc = other;",
            "prod" => "acc *= other;",
            _ => unreachable!("op_name was validated against this same set above"),
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
                "prod" => "acc *= value;",
                _ => unreachable!("op_name was validated against this same set above"),
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
        cache_key: source_scoped_cache_id(&key, &source),
        source,
        dtype,
        element_size: scalar.element_size,
        unroll_width: 1,
        vector_width: 1,
        key,
    })
}
