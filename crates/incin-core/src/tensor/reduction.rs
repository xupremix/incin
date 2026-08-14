//! Runtime reduction choices shared by tensor and neural-network operations.

/// How an operation combines values across its reduction dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    /// Average the reduced values.
    #[default]
    Mean,
    /// Add the reduced values.
    Sum,
    /// Preserve the unreduced values.
    None,
}
