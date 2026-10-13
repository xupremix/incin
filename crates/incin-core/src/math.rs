//! Scalar floating-point math that works with and without `std`.
//!
//! The inherent `f32`/`f64` math methods (`sqrt`, `ln`, `pow`, `trunc`,
//! `fract`, `powi`) live in `std`, so a `no_std` build cannot call them.
//! Every helper here forwards to the inherent method under `std`
//! (bit-identical behavior where the test suite runs) and to a `no_std`
//! fallback otherwise:
//!
//! - `sqrt`, `ln`, `pow` need a real implementation and use `libm`, which
//!   is pure Rust and `no_std`-compatible.
//! - `trunc`, `fract`, and `powi` are exact integer-domain operations and
//!   are implemented by hand, so the `no_std` path asks nothing of a
//!   third-party approximation where exactness is available.

/// Square root of an `f64` scalar.
#[inline]
pub(crate) fn sqrt_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(x)
    }
}

/// Natural logarithm of an `f64` scalar.
#[inline]
pub(crate) fn ln_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.ln()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::log(x)
    }
}

/// `x` raised to a floating-point power.
#[inline]
pub(crate) fn pow_f64(x: f64, y: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.powf(y)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::pow(x, y)
    }
}

/// Truncation toward zero, exact: `as` casts truncate, and every `f64`
/// outside `±2^52` is already integral.
#[inline]
pub(crate) fn trunc_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.trunc()
    }
    #[cfg(not(feature = "std"))]
    {
        const BOUND: f64 = 4503599627370496.0; // 2^52
        if x != x {
            return x; // NaN has no truncation; keep it a NaN.
        }
        if x < BOUND && x > -BOUND {
            (x as i64) as f64
        } else {
            x
        }
    }
}

/// Fractional part, exact: one subtraction off an exact truncation.
#[inline]
pub(crate) fn fract_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.fract()
    }
    #[cfg(not(feature = "std"))]
    {
        x - trunc_f64(x)
    }
}

/// Integer power by squaring, exact for the powers of two the dtype
/// decoders ask for. Handles negative exponents and `i32::MIN` through
/// the unsigned magnitude.
#[inline]
pub(crate) fn powi_f64(x: f64, n: i32) -> f64 {
    #[cfg(feature = "std")]
    {
        x.powi(n)
    }
    #[cfg(not(feature = "std"))]
    {
        if n == 0 {
            return 1.0;
        }
        let mut exp = (n as i64).unsigned_abs();
        let mut base = x;
        let mut acc = 1.0;
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            base *= base;
            exp >>= 1;
        }
        if n < 0 { 1.0 / acc } else { acc }
    }
}

/// `f32` integer power, same treatment as [`powi_f64`].
#[inline]
pub(crate) fn powi_f32(x: f32, n: i32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.powi(n)
    }
    #[cfg(not(feature = "std"))]
    {
        if n == 0 {
            return 1.0;
        }
        let mut exp = (n as i64).unsigned_abs();
        let mut base = x;
        let mut acc = 1.0;
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            base *= base;
            exp >>= 1;
        }
        if n < 0 { 1.0 / acc } else { acc }
    }
}
