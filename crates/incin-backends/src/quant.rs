//! The in-memory layout of a quantized block, shared by every backend that
//! stores one.
//!
//! Q8_0 is a data format, not a device capability: the CPU packs blocks, CUDA
//! sizes device allocations from the same struct, and WGPU writes the identical
//! byte sequence by hand. Housing the struct in the CPU backend made it look
//! like a CPU detail, which is why `--features cuda` could not build without
//! `cpu` also enabled.

/// One GGUF-style Q8_0 block: 32 `i8` quants sharing a single `f16` scale.
///
/// The field order is the byte order. [`incin_core::tensor::dtype::DTypeId::Q8_0`]'s
/// `block_bytes` reports the same 34 bytes, and the two are asserted equal at
/// every site that allocates by block count.
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8_0 {
    pub(crate) d: half::f16,
    pub(crate) qs: [i8; 32],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// The struct and the dtype table must agree on a block's size.
    ///
    /// Both are used to size allocations — the CPU by `size_of`, CUDA and WGPU
    /// by `block_bytes` — so a disagreement would under-allocate on one side of
    /// a transfer rather than fail to compile.
    fn block_matches_the_dtype_tables_block_size() {
        use incin_core::tensor::dtype::ConstDType;
        assert_eq!(
            core::mem::size_of::<BlockQ8_0>(),
            incin_core::tensor::dtype::Q8_0::DESCRIPTOR
                .encoding()
                .bytes_per_block()
        );
    }
}
