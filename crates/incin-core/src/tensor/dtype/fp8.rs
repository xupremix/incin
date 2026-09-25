//! OCP OFP8 8-bit floating-point element types (`F8E4M3`, `F8E5M2`).
//!
//! The two formats training hardware actually uses, kept distinct all the way
//! through the type system (issue #94): `e4m3` (4 exponent bits, 3 mantissa
//! bits) for weights and activations, `e5m2` (5 exponent bits, 2 mantissa
//! bits) for gradients. A single `f8` dtype would let the forward/backward
//! split become a runtime value, which is precisely the mistake that produces
//! silent divergence.
//!
//! Both are 1-byte POD newtypes over `u8`, following the `bf16` precedent: a
//! real [`TensorElement`](super::TensorElement) scalar dtype with a
//! per-element Rust representation, not the `Q8_0` block precedent. The
//! [`TensorElement`] set is open (D-110), so no seal change was needed — the
//! blanket impl covers any `NoUninit + Zeroable + Copy` type.
//!
//! # Conversion semantics (OCP OFP8 v1.0, saturating mode)
//!
//! * `E4M3`: bias 7, max normal **448** (`0x7E`), no infinities, NaN is only
//!   `0x7F`/`0xFF`.
//! * `E5M2`: bias 15, max normal **57344** (`0x7B`), infinities `0x7C`/`0xFC`,
//!   NaN is exponent `0x1F` with nonzero mantissa.
//! * `f32`/`f64` → fp8 is **correctly rounded** (round-to-nearest-even, single
//!   rounding via exact integer arithmetic — never a widen-then-narrow double
//!   rounding) and **saturating**: overflow, including `±Inf` input, yields the
//!   signed max normal (`±448` / `±57344`), never NaN or infinity. NaN input
//!   yields a NaN with the input's sign bit.
//! * fp8 → `f32`/`f64` is exact: every fp8 value is representable in `f32`.
//! * Subnormals round to nearest (ties to even) and are never flushed to zero
//!   except by rounding; signed zero is preserved.
//!
//! Per-tensor scaling lives outside this module: an fp8 tensor without a
//! scale is not usable, so the scale is a constructor argument of the
//! scaled-cast surface
//! ([`to_dtype_scaled`](crate::tensor::base::Tensor::to_dtype_scaled) /
//! [`from_dtype_scaled`](crate::tensor::base::Tensor::from_dtype_scaled))
//! rather than a convention.
//!
//! `no_std` clean: pure integer/float arithmetic, no allocation.

/// 8-bit floating point, OCP E4M3 format: 1 sign bit, 4 exponent bits
/// (bias 7), 3 mantissa bits. Range ±448, no infinities.
///
/// The forward-pass format: narrower range, more precision. Stored as one
/// byte; every byte pattern is a valid value (NaN payloads canonicalize on
/// conversion, see [`Self::from_f32`]).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct F8E4M3(u8);

// SAFETY: `repr(transparent)` over `u8`; every byte pattern is a valid value.
unsafe impl bytemuck::NoUninit for F8E4M3 {}
// SAFETY: the all-zero byte is the valid value +0.0.
unsafe impl bytemuck::Zeroable for F8E4M3 {}

/// 8-bit floating point, OCP E5M2 format: 1 sign bit, 5 exponent bits
/// (bias 15), 2 mantissa bits. Range ±57344, infinities representable.
///
/// The backward-pass format: gradients span a far wider dynamic range than
/// activations, so using `e4m3` for them is the standard way to get silent
/// divergence. Stored as one byte; every byte pattern is a valid value.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct F8E5M2(u8);

// SAFETY: `repr(transparent)` over `u8`; every byte pattern is a valid value.
unsafe impl bytemuck::NoUninit for F8E5M2 {}
// SAFETY: the all-zero byte is the valid value +0.0.
unsafe impl bytemuck::Zeroable for F8E5M2 {}

impl core::fmt::Debug for F8E4M3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "F8E4M3({})", self.to_f32())
    }
}

impl core::fmt::Debug for F8E5M2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "F8E5M2({})", self.to_f32())
    }
}

/// Shared encode/decode parameters for one OFP8 format.
struct Fp8Params {
    /// Exponent field width in bits.
    exp_bits: u32,
    /// Mantissa field width in bits.
    mant_bits: u32,
    /// Exponent bias.
    bias: i32,
    /// Bit pattern of the max normal magnitude (sign excluded).
    max_bits: u8,
    /// Bit pattern of the canonical NaN magnitude (sign excluded).
    nan_bits: u8,
    /// Bit pattern of infinity magnitude, if the format has infinities.
    inf_bits: Option<u8>,
}

const E4M3: Fp8Params = Fp8Params {
    exp_bits: 4,
    mant_bits: 3,
    bias: 7,
    max_bits: 0x7E,
    nan_bits: 0x7F,
    inf_bits: None,
};

const E5M2: Fp8Params = Fp8Params {
    exp_bits: 5,
    mant_bits: 2,
    bias: 15,
    max_bits: 0x7B,
    nan_bits: 0x7F,
    inf_bits: Some(0x7C),
};

/// Decode one fp8 byte to `f32`. Exact: every fp8 value is representable.
fn decode(bits: u8, p: &Fp8Params) -> f32 {
    let negative = bits & 0x80 != 0;
    let sign = if negative { -1.0f32 } else { 1.0f32 };
    let exp_mask = (1u8 << p.exp_bits) - 1;
    let exp = (bits >> p.mant_bits) & exp_mask;
    let mant = bits & ((1u8 << p.mant_bits) - 1);
    let max_enc = (1i32 << p.exp_bits) - 1;
    // NaN is built from bits, never via arithmetic on `f32::NAN`: a
    // multiply like `-1.0 * NAN` is canonicalized to +NAN by LLVM and
    // would drop the sign.
    let nan = f32::from_bits(if negative { 0xFFC0_0000 } else { 0x7FC0_0000 });
    if exp as i32 == max_enc {
        // Top exponent row: NaN, infinity, or (e4m3 only) extended normals.
        if p.inf_bits.is_some() {
            if mant == 0 {
                return sign * f32::INFINITY;
            }
            return nan;
        }
        // E4M3: mantissa 7 is the only NaN; 0..=6 are normal (emax 8).
        if mant == (1u8 << p.mant_bits) - 1 {
            return nan;
        }
    }
    if exp == 0 {
        if mant == 0 {
            // Signed zero. `sign * 0.0` preserves -0.0.
            return sign * 0.0;
        }
        // Subnormal: mant * 2^(1 - bias - mant_bits), exact in f32.
        let scale = 2f32.powi(1 - p.bias - p.mant_bits as i32);
        return sign * f32::from(mant) * scale;
    }
    // Normal: (2^mant_bits + mant) * 2^(E - bias - mant_bits), exact in f32.
    let significand = f32::from((1u8 << p.mant_bits) | mant);
    sign * significand * 2f32.powi(exp as i32 - p.bias - p.mant_bits as i32)
}

/// Round `sig` (value `sig * 2^-shift`) to an integer, round-to-nearest-even.
///
/// `shift >= 1`. Exact integer arithmetic: the dropped-bit comparison is the
/// single rounding, so the result is correctly rounded with no double-round
/// hazard from a wider intermediate.
fn round_shift(sig: u64, shift: u32) -> u64 {
    debug_assert!(shift >= 1);
    if shift >= 64 {
        return 0;
    }
    let keep = sig >> shift;
    let dropped = sig & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    if dropped > half || (dropped == half && keep & 1 == 1) {
        keep + 1
    } else {
        keep
    }
}

/// Pack a normalized source `(sign, unbiased exponent e, significand)` into
/// fp8, where `value = sig * 2^(e - src_mant)` and
/// `sig < 2^(src_mant + 1)`.
///
/// Saturating (OCP SAT mode): magnitudes above the max normal yield the
/// signed max normal, never NaN or infinity.
fn pack(sign_bit: u8, e: i32, sig: u64, src_mant: u32, p: &Fp8Params) -> u8 {
    let max_enc = (1i32 << p.exp_bits) - 1;
    let exp = e + p.bias;
    if exp > max_enc || (exp == max_enc && p.inf_bits.is_some()) {
        // Above the max normal (e5m2's top row is inf/NaN, not normal).
        return sign_bit | p.max_bits;
    }
    if exp <= 0 {
        // Subnormal (or underflow): round to a multiple of 2^(1-bias-mant).
        let lsb = 1 - p.bias - p.mant_bits as i32;
        let shift = (e - src_mant as i32) - lsb;
        let m = if shift >= 0 {
            sig << (shift as u32).min(63)
        } else {
            round_shift(sig, (-shift) as u32)
        };
        if m >= (1u64 << p.mant_bits) {
            // Rounded up into the smallest normal.
            return sign_bit | (1u8 << p.mant_bits);
        }
        if m == 0 {
            return sign_bit;
        }
        return sign_bit | (m as u8);
    }
    // Normal row. `src_mant > mant_bits` always (23/52 vs 2/3). Round the
    // fraction only: the hidden leading bit is not part of the mantissa
    // field, and rounding it would carry every value into the next binade.
    let frac = sig & ((1u64 << src_mant) - 1);
    let m = round_shift(frac, src_mant - p.mant_bits);
    let (mut exp_out, mut mant_out) = (exp, m);
    if mant_out >= (1u64 << p.mant_bits) {
        mant_out = 0;
        exp_out += 1;
        if exp_out > max_enc || (exp_out == max_enc && p.inf_bits.is_some()) {
            return sign_bit | p.max_bits;
        }
    }
    if exp_out == max_enc {
        // E4M3's top row: mantissa 7 is NaN, so 7 means overflow.
        debug_assert!(p.inf_bits.is_none());
        if mant_out >= ((1u64 << p.mant_bits) - 1) {
            return sign_bit | p.max_bits;
        }
    }
    sign_bit | ((exp_out as u8) << p.mant_bits) | (mant_out as u8)
}

/// Encode an `f32` bit pattern.
fn encode_f32(bits: u32, p: &Fp8Params) -> u8 {
    let sign_bit = ((bits >> 24) & 0x80) as u8;
    let exp8 = (bits >> 23) & 0xFF;
    let frac = bits & 0x7F_FFFF;
    if exp8 == 0xFF {
        if frac == 0 {
            // OCP SAT mode: infinities saturate to the signed max normal.
            return sign_bit | p.max_bits;
        }
        return sign_bit | p.nan_bits;
    }
    if exp8 == 0 {
        if frac == 0 {
            return sign_bit;
        }
        return pack(sign_bit, -126, frac as u64, 23, p);
    }
    pack(
        sign_bit,
        exp8 as i32 - 127,
        (frac | 0x0080_0000) as u64,
        23,
        p,
    )
}

/// Encode an `f64` bit pattern, natively (not via `f32`).
///
/// Going through `f32` first would double-round: an `f64` just above an fp8
/// rounding boundary rounds down to the boundary in `f32`, then ties-to-even
/// the wrong way. The integer path below rounds once.
fn encode_f64(bits: u64, p: &Fp8Params) -> u8 {
    let sign_bit = ((bits >> 56) & 0x80) as u8;
    let exp11 = ((bits >> 52) & 0x7FF) as u32;
    let frac = bits & 0x000F_FFFF_FFFF_FFFF;
    if exp11 == 0x7FF {
        if frac == 0 {
            return sign_bit | p.max_bits;
        }
        return sign_bit | p.nan_bits;
    }
    if exp11 == 0 {
        if frac == 0 {
            return sign_bit;
        }
        return pack(sign_bit, -1022, frac, 52, p);
    }
    pack(
        sign_bit,
        exp11 as i32 - 1023,
        frac | 0x0010_0000_0000_0000,
        52,
        p,
    )
}

impl F8E4M3 {
    /// Largest finite magnitude: 448.
    pub const MAX: Self = Self(E4M3.max_bits);
    /// Canonical quiet NaN (`0x7F`).
    pub const NAN: Self = Self(E4M3.nan_bits);

    /// Raw byte reinterpretation.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Raw byte reinterpretation.
    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Convert from `f32`: correctly rounded (RNE), saturating.
    ///
    /// Overflow and `±Inf` yield `±MAX` (448); NaN yields NaN. This is OCP
    /// SAT mode, the mode training scaling wants: a value that does not fit
    /// clips to the representable extreme instead of becoming NaN/Inf.
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        Self(encode_f32(value.to_bits(), &E4M3))
    }

    /// Convert from `f64` natively: correctly rounded (RNE), saturating.
    ///
    /// Not `from_f32(value as f32)`: that would round twice (see the
    /// module docs).
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Self(encode_f64(value.to_bits(), &E4M3))
    }

    /// Convert to `f32`. Exact.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        decode(self.0, &E4M3)
    }

    /// Convert to `f64`. Exact.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    /// Whether the value is NaN (`0x7F`/`0xFF` only, per OCP).
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.0 == 0x7F || self.0 == 0xFF
    }
}

impl F8E5M2 {
    /// Largest finite magnitude: 57344.
    pub const MAX: Self = Self(E5M2.max_bits);
    /// Canonical quiet NaN (`0x7F`).
    pub const NAN: Self = Self(E5M2.nan_bits);

    /// Raw byte reinterpretation.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Raw byte reinterpretation.
    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Convert from `f32`: correctly rounded (RNE), saturating.
    ///
    /// Overflow and `±Inf` yield `±MAX` (57344) — OCP SAT mode, even though
    /// the format can represent infinity. NaN yields NaN.
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        Self(encode_f32(value.to_bits(), &E5M2))
    }

    /// Convert from `f64` natively: correctly rounded (RNE), saturating.
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Self(encode_f64(value.to_bits(), &E5M2))
    }

    /// Convert to `f32`. Exact (`0x7C` decodes to infinity).
    #[must_use]
    pub fn to_f32(self) -> f32 {
        decode(self.0, &E5M2)
    }

    /// Convert to `f64`. Exact.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    /// Whether the value is NaN (exponent `0x1F`, nonzero mantissa).
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.0 & 0x7C == 0x7C && self.0 & 0x03 != 0
    }

    /// Whether the value is infinite (`0x7C`/`0xFC`).
    #[must_use]
    pub fn is_infinite(self) -> bool {
        self.0 == 0x7C || self.0 == 0xFC
    }
}

impl From<F8E4M3> for f32 {
    fn from(value: F8E4M3) -> Self {
        value.to_f32()
    }
}

impl From<F8E5M2> for f32 {
    fn from(value: F8E5M2) -> Self {
        value.to_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::{F8E4M3, F8E5M2, Fp8Params, decode};
    use crate::tensor::dtype::TensorElement;

    fn params_e4m3() -> Fp8Params {
        Fp8Params {
            exp_bits: 4,
            mant_bits: 3,
            bias: 7,
            max_bits: 0x7E,
            nan_bits: 0x7F,
            inf_bits: None,
        }
    }

    fn params_e5m2() -> Fp8Params {
        Fp8Params {
            exp_bits: 5,
            mant_bits: 2,
            bias: 15,
            max_bits: 0x7B,
            nan_bits: 0x7F,
            inf_bits: Some(0x7C),
        }
    }

    #[test]
    fn spec_maximums_decode_exactly() {
        assert_eq!(F8E4M3::from_bits(0x7E).to_f32(), 448.0);
        assert_eq!(F8E4M3::from_bits(0xFE).to_f32(), -448.0);
        assert_eq!(F8E5M2::from_bits(0x7B).to_f32(), 57344.0);
        assert_eq!(F8E5M2::from_bits(0xFB).to_f32(), -57344.0);
    }

    #[test]
    fn spec_specials_decode() {
        // E4M3: no infinities; only 0x7F/0xFF are NaN; 0x7E is finite.
        assert!(F8E4M3::from_bits(0x7F).to_f32().is_nan());
        assert!(F8E4M3::from_bits(0xFF).to_f32().is_nan());
        assert!(F8E4M3::from_bits(0x7E).to_f32().is_finite());
        assert!(!F8E4M3::from_bits(0x7E).is_nan());
        assert!(F8E4M3::from_bits(0x7F).is_nan());
        assert!(F8E4M3::from_bits(0xFF).is_nan());
        // E5M2: infinities and NaNs.
        assert_eq!(F8E5M2::from_bits(0x7C).to_f32(), f32::INFINITY);
        assert_eq!(F8E5M2::from_bits(0xFC).to_f32(), f32::NEG_INFINITY);
        assert!(F8E5M2::from_bits(0x7D).to_f32().is_nan());
        assert!(F8E5M2::from_bits(0x7F).to_f32().is_nan());
        assert!(F8E5M2::from_bits(0xFF).to_f32().is_nan());
        assert!(F8E5M2::from_bits(0x7C).is_infinite());
        // Signed zeros survive.
        assert_eq!(F8E4M3::from_bits(0x00).to_f32(), 0.0);
        assert!(F8E4M3::from_bits(0x80).to_f32().is_sign_negative());
        assert_eq!(F8E5M2::from_bits(0x80).to_f32(), -0.0);
        // OCP subnormal extremes.
        assert_eq!(F8E4M3::from_bits(0x01).to_f32(), 2f32.powi(-9));
        assert_eq!(F8E4M3::from_bits(0x07).to_f32(), 0.875 * 2f32.powi(-6));
        assert_eq!(F8E5M2::from_bits(0x01).to_f32(), 2f32.powi(-16));
        assert_eq!(F8E5M2::from_bits(0x03).to_f32(), 0.75 * 2f32.powi(-14));
        assert_eq!(F8E4M3::from_bits(0x08).to_f32(), 2f32.powi(-6));
        assert_eq!(F8E5M2::from_bits(0x04).to_f32(), 2f32.powi(-14));
    }

    /// Every non-NaN pattern round-trips through decode→encode; every NaN
    /// pattern canonicalizes to the sign-preserving `0x7F`-magnitude NaN.
    #[test]
    fn all_patterns_round_trip_or_canonicalize() {
        for bits in 0u8..=255 {
            let e4 = F8E4M3::from_f32(F8E4M3::from_bits(bits).to_f32());
            let e5 = F8E5M2::from_f32(F8E5M2::from_bits(bits).to_f32());
            if F8E4M3::from_bits(bits).to_f32().is_nan() {
                assert_eq!(e4.to_bits(), bits & 0x80 | 0x7F, "e4m3 {bits:#x}");
            } else {
                assert_eq!(e4.to_bits(), bits, "e4m3 {bits:#x}");
            }
            let v5 = F8E5M2::from_bits(bits).to_f32();
            if v5.is_nan() {
                assert_eq!(e5.to_bits(), bits & 0x80 | 0x7F, "e5m2 {bits:#x}");
            } else if v5.is_infinite() {
                // SAT mode: infinity saturates to max on the way back in.
                assert_eq!(
                    e5.to_bits(),
                    bits & 0x80 | 0x7B,
                    "e5m2 inf {bits:#x} saturates"
                );
            } else {
                assert_eq!(e5.to_bits(), bits, "e5m2 {bits:#x}");
            }
        }
    }

    #[test]
    fn saturation_not_nan_or_inf() {
        // Overflow clips to the signed max, per OCP SAT mode.
        assert_eq!(F8E4M3::from_f32(449.0).to_bits(), 0x7E);
        assert_eq!(F8E4M3::from_f32(-1000.0).to_bits(), 0xFE);
        assert_eq!(F8E4M3::from_f32(f32::INFINITY).to_bits(), 0x7E);
        assert_eq!(F8E4M3::from_f32(f32::NEG_INFINITY).to_bits(), 0xFE);
        assert_eq!(F8E5M2::from_f32(100_000.0).to_bits(), 0x7B);
        assert_eq!(F8E5M2::from_f32(f32::INFINITY).to_bits(), 0x7B);
        assert_eq!(F8E5M2::from_f32(f32::NEG_INFINITY).to_bits(), 0xFB);
        // NaN stays NaN with its sign.
        assert_eq!(F8E4M3::from_f32(f32::NAN).to_bits() & 0x7F, 0x7F);
        assert_eq!(F8E5M2::from_f32(f32::NAN).to_bits() & 0x7F, 0x7F);
        assert!(F8E4M3::from_f32(f32::NAN).is_nan());
        // f64 path saturates identically.
        assert_eq!(F8E4M3::from_f64(1e300).to_bits(), 0x7E);
        assert_eq!(F8E5M2::from_f64(f64::INFINITY).to_bits(), 0x7B);
        assert_eq!(F8E5M2::from_f64(f64::NAN).to_bits() & 0x7F, 0x7F);
    }

    /// Independent oracle: the nearest of the 256 patterns by `f64` distance,
    /// ties to the even significand, overflow to max, NaN to NaN. Different
    /// code path from the integer rounding under test (float distances vs
    /// bit shifts), so agreement is real evidence.
    fn oracle_f32(x: f32, e4: bool) -> u8 {
        if x.is_nan() {
            return if x.is_sign_negative() { 0xFF } else { 0x7F };
        }
        if x == 0.0 {
            // Signed zero is preserved, not rounded.
            return if x.is_sign_negative() { 0x80 } else { 0x00 };
        }
        let params = if e4 { params_e4m3() } else { params_e5m2() };
        let mut best = 0u8;
        let mut best_dist = f64::INFINITY;
        let mut best_even = false;
        let mut best_same_sign = false;
        let x_neg = x.is_sign_negative();
        for bits in 0u16..256 {
            let b = bits as u8;
            let v = decode(b, &params);
            if v.is_nan() {
                continue;
            }
            let target = if v.is_infinite() {
                // SAT mode never produces these; skip so the oracle can only
                // choose what the implementation may emit.
                continue;
            } else {
                f64::from(v)
            };
            let dist = (f64::from(x) - target).abs();
            // Even significand = even trailing pattern bits at a tie; and a
            // tie across zero (±0 are equidistant) keeps the input's sign,
            // which is what the sign-preserving implementation does.
            let even = b & 0x01 == 0;
            let same_sign = (b & 0x80 != 0) == x_neg;
            let better = dist < best_dist
                || (dist == best_dist
                    && (same_sign && !best_same_sign
                        || (same_sign == best_same_sign && even && !best_even)));
            if better {
                best = b;
                best_dist = dist;
                best_even = even;
                best_same_sign = same_sign;
            }
        }
        // Overflow: x beyond max must choose max (finite nearest is max).
        best
    }

    #[test]
    fn sampled_values_match_the_distance_oracle() {
        let mut samples = alloc::vec::Vec::new();
        // Spec boundaries and near-ties.
        for &x in &[
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            1.0625,
            1.125,
            3.5,
            7.0,
            8.0,
            100.0,
            240.0,
            400.0,
            440.0,
            447.0,
            447.9,
            448.0,
            449.0,
            500.0,
            1000.0,
            30000.0,
            57343.0,
            57344.0,
            57345.0,
            60000.0,
            1e10,
            2f32.powi(-9),
            2f32.powi(-10),
            2f32.powi(-14),
            2f32.powi(-15),
            2f32.powi(-16),
            2f32.powi(-17),
            1e-8,
            1e-7,
            0.1,
            0.33,
            2.5,
            15.75,
            31.5,
        ] {
            samples.push(x);
            samples.push(-x);
            samples.push(x * 1.0001);
            samples.push(x * 0.9999);
        }
        // Dense sweep across magnitudes.
        let mut v = 2f32.powi(-20);
        while v < 1e6 {
            samples.push(v);
            samples.push(-v);
            v *= 1.03;
        }
        // f32 subnormals must flush through the fp8 subnormal path to zero.
        samples.push(f32::MIN_POSITIVE / 2.0);
        samples.push(1e-40);
        for &x in &samples {
            assert_eq!(
                F8E4M3::from_f32(x).to_bits(),
                oracle_f32(x, true),
                "e4m3 mismatch at {x:e}"
            );
            assert_eq!(
                F8E5M2::from_f32(x).to_bits(),
                oracle_f32(x, false),
                "e5m2 mismatch at {x:e}"
            );
        }
        assert_eq!(F8E4M3::from_f32(f32::INFINITY).to_bits(), 0x7E);
        assert_eq!(F8E5M2::from_f32(f32::NEG_INFINITY).to_bits(), 0xFB);
    }

    /// `from_f64` rounds once: `1.0625 + 2^-30` is above the e4m3 midpoint
    /// between 1.0 and 1.125, so it must round up — but a detour through
    /// `f32` collapses it onto the midpoint and ties-to-even down to 1.0.
    #[test]
    fn f64_path_does_not_double_round() {
        let x = 1.0625f64 + 2f64.powi(-30);
        assert_eq!(F8E4M3::from_f64(x).to_f64(), 1.125);
        // The f32 detour would give 1.0; prove the test is sensitive to that.
        assert_eq!(F8E4M3::from_f32(x as f32).to_f64(), 1.0);
        // Exact ties go to the even significand in both widths.
        assert_eq!(F8E4M3::from_f64(1.0625).to_f64(), 1.0);
        assert_eq!(F8E5M2::from_f64(57344.0 + 4096.0).to_f64(), 57344.0);
    }

    #[test]
    fn pod_guarantees_hold() {
        // The TensorElement blanket impl's bounds, spelled out: any byte is a
        // valid value (required for host extraction of stored bytes).
        fn assert_element<T: TensorElement>() {}
        assert_element::<F8E4M3>();
        assert_element::<F8E5M2>();
        assert_eq!(core::mem::size_of::<F8E4M3>(), 1);
        assert_eq!(core::mem::size_of::<F8E5M2>(), 1);
        assert_eq!(core::mem::align_of::<F8E4M3>(), 1);
        // Zeroable: all-zero bytes are a valid (zero) value of each type.
        assert_eq!(F8E4M3::from_bits(0).to_f32(), 0.0);
        assert_eq!(F8E5M2::from_bits(0).to_f32(), 0.0);
    }
}
