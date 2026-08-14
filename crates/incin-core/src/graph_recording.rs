//! Crate-private graph recording vocabulary shared by tracing adapters.
//!
//! Keeping this bridge at the crate boundary prevents the tensor implementation
//! from depending directly on the graph representation.

pub(crate) use crate::graph::{AttributeValue, Graph, ValueId};
