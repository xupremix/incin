//! Execution policy shared by capability queries and kernel cache keys.

/// Floating-point transformation policy.
///
/// Determinism is deliberately orthogonal and lands with execution contexts;
/// a deterministic request must not alias either numerical mode in a cache.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MathMode {
    #[default]
    Precise,
    Fast,
}

impl MathMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Precise => "precise",
            Self::Fast => "fast",
        }
    }
}
