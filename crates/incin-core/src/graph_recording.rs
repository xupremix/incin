//! Crate-private graph recording vocabulary shared by tracing adapters.
//!
//! Keeping this bridge at the crate boundary prevents the tensor implementation
//! from depending directly on the graph representation.

pub(crate) use crate::graph::{AttributeValue, Graph, ValueId};

use spin::{LazyLock, Mutex};

/// Process-wide graph state belongs to the graph-recording boundary rather
/// than to the tensor runtime. The tracing backend only emits records into
/// this store; graph extraction and lifecycle remain graph concerns.
pub(crate) static TRACING_GRAPH: LazyLock<Mutex<Graph>> =
    LazyLock::new(|| Mutex::new(Graph::new()));
