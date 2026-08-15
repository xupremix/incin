/// Anonymous semantic-axis marker.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct Anon;

/// Canonical static extent spelling. The underlying type is the raw typenum
/// unsigned integer emitted by the shape macro.
pub type Static<N> = N;

/// Canonical runtime extent spelling. Runtime values live in `ShapeBuf`.
pub type Runtime = usize;

/// Canonical named-axis dimension spelling.
pub type NamedAxis<Name, Extent> = crate::shapes::dim::NamedDim<Name, Extent>;
