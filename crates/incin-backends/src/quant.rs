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

/// One NVFP4 block (issue #95): 16 E2M1 values sharing one FP8-E4M3 scale.
///
/// Interleaved scale-first wire order: 1 scale byte, then 8 packed data
/// bytes (byte `j` holds element `2j` low, `2j+1` high — the pinned nibble
/// order in [`incin_core::tensor::dtype::fp4`]). 9 bytes, 1-byte aligned,
/// matching `DTypeId::NVFP4`'s `block(16, 9, 1)`.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockNVFP4 {
    pub(crate) scale: u8,
    pub(crate) data: [u8; 8],
}

/// One MXFP4 block (issue #95, OCP MX): 32 E2M1 values sharing one E8M0
/// power-of-two scale byte. 17 bytes, 1-byte aligned, matching
/// `DTypeId::MXFP4`'s `block(32, 17, 1)`. Same scale-first interleaving and
/// nibble order as [`BlockNVFP4`].
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockMXFP4 {
    pub(crate) scale: u8,
    pub(crate) data: [u8; 16],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// The struct and the dtype table must agree on a block's size.
    ///
    /// Both are used to size allocations - the CPU by `size_of`, CUDA and WGPU
    /// by `block_bytes` - so a disagreement would under-allocate on one side of
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

    #[test]
    /// Issue #95: the FP4 structs and the dtype table must agree on a
    /// block's size, for the same allocation-sizing reason as Q8_0 above.
    /// Both are `repr(C)` scale-first, so the offsets agree too.
    fn fp4_blocks_match_the_dtype_tables_block_size() {
        use incin_core::tensor::dtype::ConstDType;
        use incin_core::tensor::dtype::{MXFP4, NVFP4};
        assert_eq!(
            core::mem::size_of::<BlockNVFP4>(),
            NVFP4::DESCRIPTOR.encoding().bytes_per_block()
        );
        assert_eq!(
            core::mem::size_of::<BlockMXFP4>(),
            MXFP4::DESCRIPTOR.encoding().bytes_per_block()
        );
        assert_eq!(core::mem::size_of::<BlockNVFP4>(), 9);
        assert_eq!(core::mem::size_of::<BlockMXFP4>(), 17);
        assert_eq!(core::mem::align_of::<BlockNVFP4>(), 1);
        assert_eq!(core::mem::align_of::<BlockMXFP4>(), 1);
        // Scale-first field order = wire order.
        assert_eq!(core::mem::offset_of!(BlockNVFP4, scale), 0);
        assert_eq!(core::mem::offset_of!(BlockNVFP4, data), 1);
        assert_eq!(core::mem::offset_of!(BlockMXFP4, scale), 0);
        assert_eq!(core::mem::offset_of!(BlockMXFP4, data), 1);
    }
}
