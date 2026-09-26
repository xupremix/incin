//! Block-scaled 4-bit floating-point codecs (issue #95: NVFP4 + MXFP4).
//!
//! Two block dtypes sharing one element format, distinguished the way the
//! hardware distinguishes them — by block size and scale encoding, in the
//! type, never as a runtime field:
//!
//! * [`NVFP4`](super::NVFP4): E2M1 elements, 16-value blocks, one FP8 **E4M3**
//!   scale byte per block (`StorageEncoding::block(16, 9, 1)` interleaved:
//!   scale byte first, then 8 packed data bytes), plus an optional per-tensor
//!   FP32 global scale that lives **outside** the block (snapshot side tensor,
//!   same category as
//!   [`to_dtype_scaled`](crate::tensor::base::Tensor::to_dtype_scaled)'s
//!   companion scale — never an encoding field).
//! * [`MXFP4`](super::MXFP4): E2M1 elements, 32-value blocks, one **E8M0**
//!   (power-of-two) scale byte per block (`StorageEncoding::block(32, 17, 1)`).
//!   No per-tensor scale in the format (OCP MX v1.0).
//!
//! # Format facts (with sources)
//!
//! * E2M1 grid (signed): `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`, max **6.0**,
//!   no infinities, no NaN — every nibble is a value ([bitsandbytes OCP
//!   table](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/851):
//!   bias 1, `S 11 1 = 2^2 * 1.5 = 6.0`; CUDA Math API `__nv_fp4_e2m1`:
//!   "does not support Inf/NaN").
//! * NVFP4: `x = x_e2m1 * s_block * s_global`, `s_block` FP8 E4M3 per 16
//!   elements, `s_global = global_amax / (448 * 6)` FP32 per tensor
//!   ([Transformer Engine NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)).
//!   The CPU reference in this module is that recipe with `s_global = 1.0`:
//!   `s_block = E4M3(block_amax / 6)`, which keeps quantize self-contained
//!   (no side channel through the `Quantize` signature) and dequantize exact
//!   up to the stated bound. A tensor-surface global, when a scale-carrying
//!   API uses one, persists as a snapshot side tensor, not in the encoding.
//! * MXFP4 scale is E8M0: 8 exponent bits, bias 127, `0xFF` = NaN, value
//!   `2^(bits - 127)` (OCP MX v1.0). The quantizer takes the smallest power
//!   of two `>= block_amax / 6`, so normalized elements land in `[-6, 6]`.
//! * Nibble packing (pinned, part of the checkpoint format): byte `j` holds
//!   elements `2j` (low nibble) and `2j + 1` (high nibble). Scale-first
//!   interleaving mirrors `Q8_0`'s scale-first block layout.
//!
//! # Conversion semantics
//!
//! * E2M1 encode is **correctly rounded** (round-to-nearest-even over the
//!   8-level grid, single rounding straight from `f32`/`f64` — never a
//!   narrow-then-narrow double rounding) and **saturating**: magnitudes past
//!   6.0 (including `±Inf`) yield `±6.0`. NaN has no encoding, so NaN input
//!   saturates to `+6.0`/`-6.0` by sign (unsigned NaN → `+6.0`).
//! * E2M1 decode is exact (every nibble is a representable `f32`).
//! * E8M0 encode takes the ceiling power of two (NaN → `0xFF`, `+Inf` →
//!   `0xFE`, non-positive → `0x00`); decode is exact.
//!
//! # Error bounds (4-bit ⇒ state the bound, prove it)
//!
//! * **NVFP4**: per element, `|x - x_hat| <= 0.25 * block_amax`, provided the
//!   block scale neither saturates nor flushes (`block_amax / 6` inside
//!   E4M3's nonzero range `[2^-9, 448]` — true for every sane tensor; below
//!   `2^-9` the scale underflows and the block decodes as zeros with
//!   `|err| <= block_amax`). Proof: encode stores
//!   `e = RNE(x / s)` with `s = E4M3(block_amax / 6)`; the grid's widest half
//!   step is `1.0` (the 4→6 binade), so `|x - e*s| <= 1.0*s`; the E4M3 scale
//!   itself carries relative error `|δ| <= 1/16` (half an ulp of 3 mantissa
//!   bits), contributing at most `6*s/16` on a full-scale element. Total
//!   `<= 1.375*s = 1.375*block_amax/6 < 0.23*block_amax < 0.25*block_amax`.
//! * **MXFP4**: per element, `|x - x_hat| <= 0.5 * block_amax` (for finite
//!   inputs whose ceiling power-of-two scale fits E8M0). Proof: `s` is the
//!   smallest power of two `>= block_amax / 6`, so `s < 2*block_amax/6`
//!   (equality only when exact, and then `s = block_amax/6`); with the same
//!   widest half step `1.0`, `|x - e*s| <= s <= block_amax/3 < 0.5*block_amax`.
//!   E8M0 scales are exact powers of two — no scale-rounding term at all.
//!
//! `no_std` clean: pure integer/float arithmetic, no allocation.

/// E2M1 magnitude grid, indexed by the 3 magnitude bits (exp_hi, exp_lo, mant).
const E2M1_MAGNITUDE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Largest E2M1 magnitude.
pub const E2M1_MAX: f32 = 6.0;

/// Decode one E2M1 nibble (low 4 bits) to `f32`. Exact.
#[must_use]
pub const fn e2m1_to_f32(nibble: u8) -> f32 {
    let mag = E2M1_MAGNITUDE[(nibble & 0x07) as usize];
    if nibble & 0x08 != 0 { -mag } else { mag }
}

/// Encode `f32` to one E2M1 nibble: correctly rounded (ties to the even bit
/// pattern), saturating at ±6.0. NaN has no encoding and saturates by sign.
#[must_use]
pub fn e2m1_from_f32(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x08 } else { 0x00 };
    let m = value.abs();
    if m.is_nan() {
        // No NaN pattern exists; saturate by sign (unsigned NaN → +6.0).
        return sign | 0x07;
    }
    // Midpoints between adjacent grid levels; equality goes to the even
    // pattern (0, 1, 2, 4 have even LSBs; 0.5, 1.5, 3, 6 have odd LSBs).
    // Every threshold is exactly representable, so the comparison is exact.
    let mag = if m <= 0.25 {
        0x00
    } else if m < 0.75 {
        0x01
    } else if m <= 1.25 {
        0x02
    } else if m < 1.75 {
        0x03
    } else if m <= 2.5 {
        0x04
    } else if m < 3.5 {
        0x05
    } else if m <= 5.0 {
        0x06
    } else {
        0x07
    };
    sign | mag
}

/// Encode `f64` to one E2M1 nibble natively (not via `f32`): same
/// ties-to-even grid, single rounding. Thresholds are exact in `f64` too.
#[must_use]
pub fn e2m1_from_f64(value: f64) -> u8 {
    let sign = if value.is_sign_negative() { 0x08 } else { 0x00 };
    let m = value.abs();
    if m.is_nan() {
        return sign | 0x07;
    }
    let mag = if m <= 0.25 {
        0x00
    } else if m < 0.75 {
        0x01
    } else if m <= 1.25 {
        0x02
    } else if m < 1.75 {
        0x03
    } else if m <= 2.5 {
        0x04
    } else if m < 3.5 {
        0x05
    } else if m <= 5.0 {
        0x06
    } else {
        0x07
    };
    sign | mag
}

/// Decode one E8M0 scale byte to `f32`. Exact: `0xFF` → NaN, anything else
/// `2^(bits - 127)` (including `0x00` = `2^-127`, a representable f32
/// subnormal).
#[must_use]
pub fn e8m0_to_f32(bits: u8) -> f32 {
    if bits == 0xFF {
        return f32::NAN;
    }
    2f32.powi(i32::from(bits) - 127)
}

/// Encode a non-negative magnitude to one E8M0 scale byte: the ceiling power
/// of two (`2^ceil(log2(v))`), biased by 127. NaN → `0xFF`, `+Inf` → `0xFE`
/// (max), non-positive → `0x00` (min). Callers pass `block_amax / 6`; the
/// result is the MXFP4 block scale.
#[must_use]
pub fn e8m0_from_f32(value: f32) -> u8 {
    if value.is_nan() {
        return 0xFF;
    }
    if value.is_infinite() {
        return if value.is_sign_positive() { 0xFE } else { 0x00 };
    }
    if value <= 0.0 {
        return 0x00;
    }
    // Ceiling binary exponent of a positive finite f32: frexp-grade exact.
    let bits = value.to_bits();
    let exp8 = ((bits >> 23) & 0xFF) as i32 - 127;
    let frac = bits & 0x7F_FFFF;
    // Exact powers of two need no rounding up; anything else does.
    let e = if frac == 0 { exp8 } else { exp8 + 1 };
    // E8M0 holds 2^-127..=2^127, i.e. biased bytes 0x00..=0xFE.
    if e >= 127 {
        0xFE
    } else if e <= -127 {
        0x00
    } else {
        (e + 127) as u8
    }
}

/// Pack 16 E2M1 nibbles (element order) into 8 bytes: byte `j` holds element
/// `2j` low, element `2j+1` high. The pinned checkpoint nibble order.
#[must_use]
pub fn pack_nibbles(elements: &[u8; 16]) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (j, byte) in out.iter_mut().enumerate() {
        *byte = (elements[2 * j] & 0x0F) | ((elements[2 * j + 1] & 0x0F) << 4);
    }
    out
}

/// Unpack 8 bytes into 16 E2M1 nibbles (element order): inverse of
/// [`pack_nibbles`].
#[must_use]
pub fn unpack_nibbles(packed: &[u8; 8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (j, byte) in packed.iter().enumerate() {
        out[2 * j] = byte & 0x0F;
        out[2 * j + 1] = (byte >> 4) & 0x0F;
    }
    out
}

/// Pack 32 E2M1 nibbles into 16 bytes (same low-even/high-odd order as
/// [`pack_nibbles`], twice the block).
#[must_use]
pub fn pack_nibbles32(elements: &[u8; 32]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (j, byte) in out.iter_mut().enumerate() {
        *byte = (elements[2 * j] & 0x0F) | ((elements[2 * j + 1] & 0x0F) << 4);
    }
    out
}

/// Unpack 16 bytes into 32 E2M1 nibbles: inverse of [`pack_nibbles32`].
#[must_use]
pub fn unpack_nibbles32(packed: &[u8; 16]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (j, byte) in packed.iter().enumerate() {
        out[2 * j] = byte & 0x0F;
        out[2 * j + 1] = (byte >> 4) & 0x0F;
    }
    out
}

/// Encode one NVFP4 block: 16 `f32` values → `(e4m3_scale_byte, packed_data)`.
///
/// `s_block = E4M3(block_amax / 6)` (the TE recipe with `s_global = 1.0`;
/// saturating, so absurd magnitudes clip rather than wrap), elements
/// `RNE(x_i / s_block)`. An all-zero block encodes as scale `0x00` + zero
/// data, so it decodes bit-exactly.
pub fn encode_nvfp4_block(values: &[f32; 16]) -> (u8, [u8; 8]) {
    let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return (0x00, [0u8; 8]);
    }
    let scale = super::F8E4M3::from_f32(amax / E2M1_MAX);
    let s = scale.to_f32();
    if s == 0.0 {
        // E4M3-underflowed scale (sub-2^-9-block magnitudes): emit zeros
        // rather than dividing by zero into saturated ±6s.
        return (scale.to_bits(), [0u8; 8]);
    }
    let mut nibbles = [0u8; 16];
    for (i, v) in values.iter().enumerate() {
        nibbles[i] = e2m1_from_f32(v / s);
    }
    (scale.to_bits(), pack_nibbles(&nibbles))
}

/// Decode one NVFP4 block: `(e4m3_scale_byte, packed_data)` → 16 `f32`
/// values (`e2m1(nibble) * s_block`).
pub fn decode_nvfp4_block(scale_bits: u8, packed: &[u8; 8]) -> [f32; 16] {
    let s = super::F8E4M3::from_bits(scale_bits).to_f32();
    let nibbles = unpack_nibbles(packed);
    let mut out = [0.0f32; 16];
    for (i, n) in nibbles.iter().enumerate() {
        out[i] = e2m1_to_f32(*n) * s;
    }
    out
}

/// Encode one MXFP4 block: 32 `f32` values → `(e8m0_scale_byte, packed_data)`.
///
/// `s = min E8M0 power of two >= block_amax / 6` (exact, never rounded —
/// MXFP4 has no scale-rounding term), elements `RNE(x_i / s)`.
pub fn encode_mxfp4_block(values: &[f32; 32]) -> (u8, [u8; 16]) {
    let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return (0x00, [0u8; 16]);
    }
    let scale_bits = e8m0_from_f32(amax / E2M1_MAX);
    let s = e8m0_to_f32(scale_bits);
    let mut nibbles = [0u8; 32];
    for (i, v) in values.iter().enumerate() {
        nibbles[i] = e2m1_from_f32(v / s);
    }
    (scale_bits, pack_nibbles32(&nibbles))
}

/// Decode one MXFP4 block: `(e8m0_scale_byte, packed_data)` → 32 `f32`
/// values (`e2m1(nibble) * s`).
pub fn decode_mxfp4_block(scale_bits: u8, packed: &[u8; 16]) -> [f32; 32] {
    let s = e8m0_to_f32(scale_bits);
    let nibbles = unpack_nibbles32(packed);
    let mut out = [0.0f32; 32];
    for (i, n) in nibbles.iter().enumerate() {
        out[i] = e2m1_to_f32(*n) * s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        decode_nvfp4_block, e2m1_from_f32, e2m1_from_f64, e2m1_to_f32, e8m0_from_f32, e8m0_to_f32,
        encode_mxfp4_block, encode_nvfp4_block, pack_nibbles, unpack_nibbles,
    };
    use crate::tensor::dtype::F8E4M3;

    #[test]
    fn e2m1_grid_decodes_to_the_spec_levels() {
        let grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (mag, expected) in grid.iter().enumerate() {
            let mag = mag as u8;
            assert_eq!(e2m1_to_f32(mag), *expected, "mag {mag:#x}");
            assert_eq!(e2m1_to_f32(mag | 0x08), -*expected, "mag {mag:#x} signed");
        }
        // Signed zero is preserved.
        assert_eq!(e2m1_to_f32(0x08), -0.0);
        assert!(e2m1_to_f32(0x08).is_sign_negative());
    }

    #[test]
    fn e2m1_encode_rounds_to_nearest_ties_to_even() {
        // Exact grid points (and signed partners) encode to themselves.
        for (mag, v) in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0].iter().enumerate() {
            assert_eq!(e2m1_from_f32(*v), mag as u8, "grid {v}");
            assert_eq!(e2m1_from_f32(-*v), mag as u8 | 0x08, "grid -{v}");
        }
        // Midpoints go to the even pattern: 0.25→0, 0.75→1, 1.25→1,
        // 1.75→2, 2.5→2, 3.5→4, 5.0→4.
        for (tie, even) in [
            (0.25, 0x00),
            (0.75, 0x02),
            (1.25, 0x02),
            (1.75, 0x04),
            (2.5, 0x04),
            (3.5, 0x06),
            (5.0, 0x06),
        ] {
            assert_eq!(e2m1_from_f32(tie), even, "tie {tie}");
            assert_eq!(e2m1_from_f32(-tie), even | 0x08, "tie -{tie}");
        }
        // Just off the tie goes to the nearer side.
        assert_eq!(e2m1_from_f32(0.250_001), 0x01);
        assert_eq!(e2m1_from_f32(0.249_999), 0x00);
        assert_eq!(e2m1_from_f32(5.000_001), 0x07);
        assert_eq!(e2m1_from_f32(4.999_999), 0x06);
        // f64 path agrees (single rounding, no f32 detour).
        assert_eq!(e2m1_from_f64(0.75), 0x02);
        assert_eq!(e2m1_from_f64(5.0), 0x06);
        assert_eq!(e2m1_from_f64(-6.5), 0x0F);
    }

    #[test]
    fn e2m1_encode_saturates_without_nan_or_inf() {
        // No NaN/Inf patterns exist: overflow clips to ±6, NaN to signed 6.
        assert_eq!(e2m1_from_f32(7.0), 0x07);
        assert_eq!(e2m1_from_f32(-100.0), 0x0F);
        assert_eq!(e2m1_from_f32(f32::INFINITY), 0x07);
        assert_eq!(e2m1_from_f32(f32::NEG_INFINITY), 0x0F);
        assert_eq!(e2m1_from_f32(f32::NAN), 0x07);
        assert_eq!(e2m1_from_f32(-f32::NAN), 0x0F);
        // Decode of every nibble is finite (never NaN/Inf).
        for n in 0u8..16 {
            assert!(e2m1_to_f32(n).is_finite(), "nibble {n:#x}");
        }
        // Round-trip: decode→encode is the identity on all 16 nibbles.
        for n in 0u8..16 {
            assert_eq!(e2m1_from_f32(e2m1_to_f32(n)), n, "nibble {n:#x}");
        }
    }

    #[test]
    fn e8m0_round_trips_powers_of_two_and_refuses_nan() {
        for exp in -126..=126i32 {
            let v = 2f32.powi(exp);
            let bits = e8m0_from_f32(v);
            assert_eq!(bits, (exp + 127) as u8, "2^{exp}");
            assert_eq!(e8m0_to_f32(bits), v, "decode 2^{exp}");
        }
        // Ceiling: just above a power of two rounds up.
        assert_eq!(e8m0_from_f32(3.0), e8m0_from_f32(4.0));
        assert_eq!(e8m0_to_f32(e8m0_from_f32(3.0)), 4.0);
        assert_eq!(e8m0_from_f32(0.75), e8m0_from_f32(1.0));
        // Extremes clamp, NaN is the only non-finite decode.
        assert_eq!(e8m0_from_f32(f32::INFINITY), 0xFE);
        assert_eq!(e8m0_from_f32(0.0), 0x00);
        assert_eq!(e8m0_from_f32(-1.0), 0x00);
        assert_eq!(e8m0_from_f32(f32::NAN), 0xFF);
        assert!(e8m0_to_f32(0xFF).is_nan());
        assert!(e8m0_to_f32(0xFE).is_finite());
    }

    #[test]
    fn nibble_packing_pins_low_even_high_odd() {
        let elements: [u8; 16] = core::array::from_fn(|i| i as u8);
        let packed = pack_nibbles(&elements);
        assert_eq!(packed[0], 0x10, "byte 0 = high(1) low(0)");
        assert_eq!(packed[7], 0xFE, "byte 7 = high(15) low(14)");
        assert_eq!(unpack_nibbles(&packed), elements);
    }

    #[test]
    fn nvfp4_block_scale_follows_the_te_recipe_at_global_one() {
        // block_amax = 12 → s_block = E4M3(2.0) = 0x40... but checked
        // through the F8E4M3 codec, not a hardcoded pattern.
        let mut values = [0.0f32; 16];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as f32 - 8.0) * 1.5; // amax 12
        }
        let (scale_bits, _) = encode_nvfp4_block(&values);
        assert_eq!(scale_bits, F8E4M3::from_f32(2.0).to_bits());
        assert_eq!(F8E4M3::from_bits(scale_bits).to_f32(), 2.0);
        // Zero block is exact zeros, not a denormal-scale artifact.
        let (zs, zp) = encode_nvfp4_block(&[0.0; 16]);
        assert_eq!((zs, zp), (0x00, [0u8; 8]));
        assert_eq!(decode_nvfp4_block(zs, &zp), [0.0; 16]);
    }

    #[test]
    fn mxfp4_block_scale_is_the_ceiling_power_of_two() {
        // block_amax = 12 → amax/6 = 2 → E8M0(2.0) = 0x80.
        let mut values = [0.0f32; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as f32 - 16.0) * 0.75; // amax 12
        }
        let (scale_bits, _) = encode_mxfp4_block(&values);
        assert_eq!(scale_bits, 0x80);
        assert_eq!(e8m0_to_f32(scale_bits), 2.0);
        // amax just over 12 needs the next power of two up.
        values[0] = 12.5;
        let (up, _) = encode_mxfp4_block(&values);
        assert_eq!(e8m0_to_f32(up), 4.0);
    }
}
