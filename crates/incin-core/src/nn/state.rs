//! Backend-neutral, owned model state artifacts.
//!
//! A state snapshot is deliberately separate from live tensors and backend
//! variable handles.  It is therefore safe to serialize, inspect, validate,
//! and stage before touching a module's live state.

use alloc::{collections::BTreeMap, string::String, vec::Vec};
use core::fmt;

use crate::{
    err::{Error, ErrorMessage, Result},
    shapes::ShapeBuf,
    tensor::dtype::DTypeDescriptor,
};

/// Durable, hierarchical state name.  This is a serialization path, not a
/// parameter/runtime-variable identity or alias identifier.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct StatePath(String);

impl StatePath {
    /// The root path.
    #[must_use]
    pub fn root() -> Self {
        Self(String::new())
    }

    /// Creates a path from its canonical dotted representation.
    pub fn new(path: impl Into<String>) -> Result<Self> {
        let path = path.into();
        if path.is_empty()
            || path.starts_with('.')
            || path.ends_with('.')
            || path
                .split('.')
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(Error::InvalidModuleState {
                operation: "state path",
                reason: ErrorMessage::new("path must contain non-empty dotted components"),
            });
        }
        Ok(Self(path))
    }

    /// Adds a named child component.
    #[must_use]
    pub fn child(&self, name: &str) -> Self {
        let component = name.trim_matches('.');
        if self.0.is_empty() {
            Self(component.into())
        } else {
            Self(format!("{}.{}", self.0, component))
        }
    }

    /// Adds a flat positional child component used by `Sequential`.
    #[must_use]
    pub fn index(&self, index: usize) -> Self {
        self.child(&index.to_string())
    }

    /// Returns the canonical dotted representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StatePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Whether a state value is trainable or a persistent non-trainable buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StateRole {
    Parameter,
    Buffer,
}

/// One owned, exact-dtype state value.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StateValue {
    shape: ShapeBuf,
    dtype: DTypeDescriptor,
    bytes: Vec<u8>,
    role: StateRole,
}

impl StateValue {
    /// Constructs and validates an owned value.
    pub fn new(
        shape: ShapeBuf,
        dtype: DTypeDescriptor,
        bytes: Vec<u8>,
        role: StateRole,
    ) -> Result<Self> {
        let elements = shape.numel().ok_or_else(|| Error::InvalidModuleState {
            operation: "state value",
            reason: ErrorMessage::new("shape element count overflow"),
        })?;
        let expected = dtype
            .size_bytes(elements, crate::shapes::error::OperationKind::Storage)
            .map_err(|error| Error::InvalidModuleState {
                operation: "state value",
                reason: ErrorMessage::new(error.to_string()),
            })?;
        if bytes.len() != expected {
            return Err(Error::MalformedArtifact {
                operation: "state value",
                artifact: "model state",
                reason: ErrorMessage::new(format!(
                    "byte length {} does not match shape {:?} and dtype {} (expected {})",
                    bytes.len(),
                    shape.dims(),
                    dtype.name(),
                    expected
                )),
            });
        }
        Ok(Self {
            shape,
            dtype,
            bytes,
            role,
        })
    }

    /// Runtime shape.
    #[must_use]
    pub fn shape(&self) -> &ShapeBuf {
        &self.shape
    }
    /// Exact logical and physical dtype descriptor.
    #[must_use]
    pub fn dtype(&self) -> DTypeDescriptor {
        self.dtype
    }
    /// Native bytes, owned by this value.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// State role.
    #[must_use]
    pub fn role(&self) -> StateRole {
        self.role
    }
}

/// An owned collection of heterogeneous model state.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StateSnapshot(BTreeMap<StatePath, StateValue>);

impl StateSnapshot {
    /// Creates an empty snapshot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Inserts a value, rejecting duplicate paths.
    pub fn insert(&mut self, path: StatePath, value: StateValue) -> Result<()> {
        if self.0.insert(path, value).is_some() {
            return Err(Error::InvalidModuleState {
                operation: "state snapshot",
                reason: ErrorMessage::new("duplicate state path"),
            });
        }
        Ok(())
    }
    /// Looks up a path.
    #[must_use]
    pub fn get(&self, path: &StatePath) -> Option<&StateValue> {
        self.0.get(path)
    }
    /// Number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Whether the snapshot is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Deterministically ordered entries.
    pub fn iter(&self) -> impl Iterator<Item = (&StatePath, &StateValue)> {
        self.0.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shapes::ShapeBuf, tensor::dtype::DTypeId};

    #[test]
    fn validates_exact_native_byte_length() {
        let value = StateValue::new(
            ShapeBuf::from_slice(&[2, 3]),
            DTypeId::F16.descriptor(),
            vec![0; 12],
            StateRole::Parameter,
        )
        .expect("valid f16 state");
        assert_eq!(value.dtype(), DTypeId::F16.descriptor());
        assert!(
            StateValue::new(
                ShapeBuf::from_slice(&[2, 3]),
                DTypeId::F16.descriptor(),
                vec![0; 24],
                StateRole::Parameter,
            )
            .is_err()
        );
    }

    #[test]
    fn paths_are_ordered_and_distinct_from_runtime_identity() {
        let root = StatePath::root();
        assert_eq!(
            root.child("q_proj").child("weight").as_str(),
            "q_proj.weight"
        );
        assert_eq!(root.index(3).as_str(), "3");
        assert!(StatePath::new("q_proj.weight").is_ok());
        assert!(StatePath::new("q_proj..weight").is_err());
    }
}
