/// `binary`.
pub mod binary;
/// `index`.
pub mod index;
/// `loss`.
pub mod loss;
/// `manipulation`.
pub mod manipulation;
/// `module`.
pub mod module;
/// `reduce`.
pub mod reduce;
/// `unary`.
pub mod unary;

pub use index::{DTypeEq, IndexArgs, IndexSpec, ShapeEq};

/// Resolves the deliberately infallible Rust operator surface.
///
/// The named tensor methods retain [`crate::err::Result`] so callers can
/// recover from backend, device, or dynamic-shape failures.  Rust's operator
/// traits cannot express that result type while still composing naturally, so
/// this is the single, explicit process-boundary conversion used by `+`, `-`,
/// `*`, `/`, scalar forms, and unary `-`.
///
/// Do not include the error value here: it can contain backend-provided text.
/// The panic text is intentionally fixed and bounded, and identifies only the
/// operator spelling that crossed the convenience boundary.
#[cold]
#[track_caller]
pub(crate) fn operator_or_panic<T>(operator: &'static str, result: crate::err::Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(_) => panic!("incin tensor operator `{operator}` failed"),
    }
}
