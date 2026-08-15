//! Backend-neutral, owned model state artifacts.
//!
//! A state snapshot is deliberately separate from live tensors and backend
//! variable handles.  It is therefore safe to serialize, inspect, validate,
//! and stage before touching a module's live state.

use alloc::{
    collections::BTreeMap,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;

use crate::{
    err::{Error, ErrorMessage, Result},
    shapes::ShapeBuf,
    tensor::dtype::DTypeDescriptor,
};

/// Durable, hierarchical state name.  This is a serialization path, not a
/// parameter/runtime-variable identity or alias identifier.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize,
)]
pub struct StatePath(String);

impl<'de> serde::Deserialize<'de> for StatePath {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = <String as serde::Deserialize>::deserialize(deserializer)?;
        Self::new(raw).map_err(|error| serde::de::Error::custom(error.to_string()))
    }
}

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

/// Receives typed state leaves during module traversal.
pub trait StateVisitor<B: crate::tensor::backend::VariableBackend> {
    fn visit_param<S, K, Train>(
        &mut self,
        path: &StatePath,
        param: &crate::nn::param::Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop,
        Train: crate::nn::param::TrainState;

    fn visit_buffer<S, K>(
        &mut self,
        path: &StatePath,
        buffer: &crate::nn::param::Buffer<S, B, K>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop;
}

/// Structural traversal of typed parameter and buffer leaves.
pub trait VisitState<B: crate::tensor::backend::VariableBackend> {
    /// Number of flat module slots occupied by this subtree.
    fn flat_width() -> usize
    where
        Self: Sized,
    {
        1
    }

    fn visit_state<V: StateVisitor<B>>(
        &self,
        path: &StatePath,
        visitor: &mut V,
    ) -> Result<()>;
}

/// Receives mutable typed state leaves while restoring a snapshot.
pub trait StateMutVisitor<B: crate::tensor::backend::VariableBackend> {
    fn visit_param<S, K, Train>(
        &mut self,
        path: &StatePath,
        param: &mut crate::nn::param::Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop,
        Train: crate::nn::param::TrainState;

    fn visit_buffer<S, K>(
        &mut self,
        path: &StatePath,
        buffer: &mut crate::nn::param::Buffer<S, B, K>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop;
}

/// Structural mutable traversal used by snapshot restoration.
pub trait VisitStateMut<B: crate::tensor::backend::VariableBackend> {
    /// Number of flat module slots occupied by this subtree.
    fn flat_width() -> usize
    where
        Self: Sized,
    {
        1
    }

    fn visit_state_mut<V: StateMutVisitor<B>>(
        &mut self,
        path: &StatePath,
        visitor: &mut V,
    ) -> Result<()>;
}

/// Collects typed leaves into the durable, backend-neutral snapshot format.
#[derive(Debug, Default)]
pub struct StateSnapshotVisitor {
    snapshot: StateSnapshot,
}

impl StateSnapshotVisitor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn into_snapshot(self) -> StateSnapshot {
        self.snapshot
    }
}

impl<B: crate::tensor::backend::VariableBackend> StateVisitor<B> for StateSnapshotVisitor {
    fn visit_param<S, K, Train>(
        &mut self,
        path: &StatePath,
        param: &crate::nn::param::Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop,
        Train: crate::nn::param::TrainState,
    {
        self.snapshot.insert(path.clone(), param.snapshot_state_value(path)?)
    }

    fn visit_buffer<S, K>(
        &mut self,
        path: &StatePath,
        buffer: &crate::nn::param::Buffer<S, B, K>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop,
    {
        self.snapshot.insert(path.clone(), buffer.snapshot_state_value(path)?)
    }
}

/// Runs typed state traversal and returns an owned snapshot.
pub fn collect_state<B, M>(module: &M) -> Result<StateSnapshot>
where
    B: crate::tensor::backend::VariableBackend,
    M: VisitState<B>,
{
    let mut visitor = StateSnapshotVisitor::new();
    module.visit_state(&StatePath::root(), &mut visitor)?;
    Ok(visitor.into_snapshot())
}

struct StatePreparation<'a> {
    snapshot: &'a StateSnapshot,
}

impl<'a, B: crate::tensor::backend::VariableBackend> StateMutVisitor<B> for StatePreparation<'a> {
    fn visit_param<S, K, Train>(
        &mut self,
        path: &StatePath,
        param: &mut crate::nn::param::Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop,
        Train: crate::nn::param::TrainState,
    {
        param.prepare_state_value(path, self.snapshot)
    }

    fn visit_buffer<S, K>(
        &mut self,
        path: &StatePath,
        buffer: &mut crate::nn::param::Buffer<S, B, K>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop,
    {
        buffer.prepare_state_value(path, self.snapshot)
    }
}

struct StateCommit;

impl<B: crate::tensor::backend::VariableBackend> StateMutVisitor<B> for StateCommit {
    fn visit_param<S, K, Train>(
        &mut self,
        _path: &StatePath,
        param: &mut crate::nn::param::Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop,
        Train: crate::nn::param::TrainState,
    {
        param.commit_prepared_state();
        Ok(())
    }

    fn visit_buffer<S, K>(
        &mut self,
        _path: &StatePath,
        buffer: &mut crate::nn::param::Buffer<S, B, K>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop,
    {
        buffer.commit_prepared_state();
        Ok(())
    }
}

struct StateClear;

impl<B: crate::tensor::backend::VariableBackend> StateMutVisitor<B> for StateClear {
    fn visit_param<S, K, Train>(
        &mut self,
        _path: &StatePath,
        param: &mut crate::nn::param::Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop,
        Train: crate::nn::param::TrainState,
    {
        param.clear_prepared_state();
        Ok(())
    }

    fn visit_buffer<S, K>(
        &mut self,
        _path: &StatePath,
        buffer: &mut crate::nn::param::Buffer<S, B, K>,
    ) -> Result<()>
    where
        S: crate::shapes::Shape,
        K: crate::tensor::dtype::DType<Arg = ()>,
        B: crate::tensor::backend::SupportsDType<K>
            + crate::exec::Capabilities
            + crate::tensor::backend::HostInterop,
    {
        buffer.clear_prepared_state();
        Ok(())
    }
}

/// Restores a complete snapshot through typed mutable leaf visits.
pub fn load_state<B, M>(module: &mut M, snapshot: &StateSnapshot) -> Result<()>
where
    B: crate::tensor::backend::VariableBackend,
    M: VisitState<B> + VisitStateMut<B>,
{
    let current = collect_state::<B, _>(&*module)?;
    let expected: alloc::collections::BTreeSet<_> =
        current.iter().map(|(path, _)| path).collect();
    let provided: alloc::collections::BTreeSet<_> =
        snapshot.iter().map(|(path, _)| path).collect();
    if expected != provided {
        let missing = expected
            .difference(&provided)
            .map(ToString::to_string)
            .collect::<alloc::vec::Vec<_>>();
        let unexpected = provided
            .difference(&expected)
            .map(ToString::to_string)
            .collect::<alloc::vec::Vec<_>>();
        return Err(Error::InvalidModuleState {
            operation: "load state",
            reason: ErrorMessage::new(format!(
                "state paths differ: missing {:?}, unexpected {:?}",
                missing, unexpected
            )),
        });
    }
    let mut visitor = StatePreparation { snapshot };
    if let Err(error) = module.visit_state_mut(&StatePath::root(), &mut visitor) {
        let mut clear = StateClear;
        let _ = module.visit_state_mut(&StatePath::root(), &mut clear);
        return Err(error);
    }

    let mut commit = StateCommit;
    module.visit_state_mut(&StatePath::root(), &mut commit)
}

/// One owned, exact-dtype state value.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct StateValue {
    shape: ShapeBuf,
    dtype: DTypeDescriptor,
    bytes: Vec<u8>,
    role: StateRole,
}

#[derive(serde::Deserialize)]
struct StateValueWire {
    shape: ShapeBuf,
    dtype: DTypeDescriptor,
    bytes: Vec<u8>,
    role: StateRole,
}

impl<'de> serde::Deserialize<'de> for StateValue {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = <StateValueWire as serde::Deserialize>::deserialize(deserializer)?;
        Self::new(wire.shape, wire.dtype, wire.bytes, wire.role)
            .map_err(|error| serde::de::Error::custom(error.to_string()))
    }
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
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct StateSnapshot(BTreeMap<StatePath, StateValue>);

impl<'de> serde::Deserialize<'de> for StateSnapshot {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct SnapshotVisitor;

        impl<'de> serde::de::Visitor<'de> for SnapshotVisitor {
            type Value = StateSnapshot;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a map of unique state paths to state values")
            }

            fn visit_map<A>(self, mut map: A) -> core::result::Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut snapshot = StateSnapshot::new();
                while let Some((path, value)) = map.next_entry()? {
                    snapshot
                        .insert(path, value)
                        .map_err(|error| serde::de::Error::custom(error.to_string()))?;
                }
                Ok(snapshot)
            }
        }

        deserializer.deserialize_map(SnapshotVisitor)
    }
}

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

    #[test]
    fn preserves_supported_native_dtype_payloads() {
        for dtype in [
            DTypeId::F32,
            DTypeId::F16,
            DTypeId::BF16,
            DTypeId::I64,
            DTypeId::U32,
            DTypeId::U8,
            DTypeId::Bool,
            DTypeId::Q8_0,
        ] {
            let descriptor = dtype.descriptor();
            let bytes = descriptor
                .size_bytes(32, crate::shapes::error::OperationKind::Storage)
                .expect("test dtype has a storage size");
            let value = StateValue::new(
                ShapeBuf::from_slice(&[32]),
                descriptor,
                (0..bytes).map(|index| index as u8).collect(),
                StateRole::Buffer,
            )
            .expect("native payload should validate");
            assert_eq!(value.dtype(), descriptor);
            assert_eq!(value.bytes().len(), bytes);
        }
    }

    #[test]
    fn rejects_overflow_and_malformed_payloads() {
        assert!(
            StateValue::new(
                ShapeBuf::from_slice(&[usize::MAX, 2]),
                DTypeId::F32.descriptor(),
                Vec::new(),
                StateRole::Parameter,
            )
            .is_err()
        );
        assert!(
            StateValue::new(
                ShapeBuf::from_slice(&[2]),
                DTypeId::F32.descriptor(),
                vec![0; 3],
                StateRole::Parameter,
            )
            .is_err()
        );
    }

    #[test]
    fn wire_deserialization_revalidates_paths_and_payload_lengths() {
        let invalid_path = postcard::to_allocvec(&"layer..weight".to_string()).unwrap();
        assert!(postcard::from_bytes::<StatePath>(&invalid_path).is_err());

        #[derive(serde::Serialize)]
        struct Wire {
            shape: ShapeBuf,
            dtype: crate::tensor::dtype::DTypeDescriptor,
            bytes: Vec<u8>,
            role: StateRole,
        }

        let invalid_value = postcard::to_allocvec(&Wire {
            shape: ShapeBuf::from_slice(&[2]),
            dtype: DTypeId::F32.descriptor(),
            bytes: vec![0; 3],
            role: StateRole::Parameter,
        })
        .unwrap();
        assert!(postcard::from_bytes::<StateValue>(&invalid_value).is_err());
    }
}
