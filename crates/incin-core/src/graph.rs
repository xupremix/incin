use crate::err::{Error, Result};
use crate::exec::{ExecutionSite, LayoutClass, OperationIdentity, ShapeExpr};
use crate::shapes::error::OperationKind;
use crate::tensor::device::DeviceId;
use crate::tensor::dtype::DTypeDescriptor;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

/// Identifies a tensor value in a graph.
pub type ValueId = usize;
/// Identifies an operation node in a graph.
pub type NodeId = usize;

/// Metadata for one graph value. Concrete shape vectors are retained for
/// eager tracing; compiler-facing symbolic metadata is attached by capture.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Value {
    /// Value identifier within the graph.
    pub id: ValueId,
    /// Concrete extents when statically known.
    pub shape: Vec<usize>,
    /// Symbolic shape supporting dynamic dims.
    pub shape_expr: ShapeExpr,
    /// Element dtype of the value.
    pub dtype: DTypeDescriptor,
    /// Logical placement when known at capture time.
    #[serde(default)]
    pub device: Option<DeviceId>,
    /// Layout fact when the capture backend can establish one.
    #[serde(default)]
    pub layout: Option<LayoutClass>,
    /// Human-readable name, when supplied.
    pub name: Option<String>,
}

/// A canonical operation node. Built-in identity comes directly from the
/// operation catalog. Custom operations use their namespaced OperationKey.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Node {
    /// Node identifier.
    pub id: NodeId,
    /// Operation identity executing here.
    pub operation: OperationIdentity,
    /// Where the node executes, when resolved.
    pub execution_site: Option<ExecutionSite>,
    /// Identifiers of consumed values.
    pub inputs: Vec<ValueId>,
    /// Identifiers of produced values.
    pub outputs: Vec<ValueId>,
    /// Named operation attributes.
    pub attributes: BTreeMap<String, AttributeValue>,
    /// Versioned bytes of the canonical typed descriptor, when capture runs
    /// with the standard serialization support enabled.
    pub descriptor_payload: Option<DescriptorPayload>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Encoded canonical descriptor carried by a captured node.
pub struct DescriptorPayload {
    /// Schema version of the payload.
    pub schema: u32,
    /// Encoded bytes.
    pub payload: Vec<u8>,
}

/// A graph attribute value. The canonical typed descriptor remains the source
/// of semantic validation; this is its stable graph serialization form.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AttributeValue {
    /// 64-bit signed integer attribute.
    Int(i64),
    /// Single-precision float attribute.
    Float(f32),
    /// UTF-8 string attribute.
    String(String),
    /// List of signed integers.
    Ints(Vec<i64>),
    /// List of single-precision floats.
    Floats(Vec<f32>),
}

#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Directed acyclic computation graph.
pub struct Graph {
    #[serde(with = "string_key_map")]
    /// All values by id.
    pub values: BTreeMap<ValueId, Value>,
    /// Topologically ordered nodes.
    pub nodes: Vec<Node>,
    /// Identifiers of consumed values.
    pub inputs: Vec<ValueId>,
    /// Identifiers of produced values.
    pub outputs: Vec<ValueId>,
    #[serde(with = "string_key_map")]
    /// Initializer payloads keyed by value id.
    pub initializers: BTreeMap<ValueId, Vec<u8>>,
    next_value_id: usize,
    next_node_id: usize,
    #[serde(default)]
    named_symbols: BTreeMap<String, crate::exec::SymbolId>,
    #[serde(default)]
    next_symbol_id: u32,
}

mod string_key_map {
    use alloc::collections::BTreeMap;
    use alloc::string::{String, ToString};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<K, V, S>(map: &BTreeMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        K: ToString,
        V: Serialize,
        S: Serializer,
    {
        let string_map: BTreeMap<String, &V> = map
            .iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect();
        string_map.serialize(serializer)
    }

    pub fn deserialize<'de, K, V, D>(deserializer: D) -> Result<BTreeMap<K, V>, D::Error>
    where
        K: core::str::FromStr + core::hash::Hash + Eq + Ord,
        K::Err: core::fmt::Display,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let string_map: BTreeMap<String, V> = BTreeMap::deserialize(deserializer)?;
        string_map
            .into_iter()
            .map(|(key, value)| {
                key.parse::<K>()
                    .map_err(serde::de::Error::custom)
                    .map(|key| (key, value))
            })
            .collect()
    }
}

impl Graph {
    /// Creates an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a value with shape/dtype validation.
    pub fn add_value<D: Into<DTypeDescriptor>>(
        &mut self,
        shape: Vec<usize>,
        dtype: D,
        name: Option<String>,
    ) -> ValueId {
        let id = self.next_value_id;
        self.next_value_id += 1;
        let shape_expr = ShapeExpr::concrete(&shape);
        self.values.insert(
            id,
            Value {
                id,
                shape,
                shape_expr,
                dtype: dtype.into(),
                device: None,
                layout: None,
                name,
            },
        );
        id
    }

    /// Records physical metadata that was available for a captured value.
    pub fn set_value_placement(
        &mut self,
        value_id: ValueId,
        device: Option<DeviceId>,
        layout: Option<LayoutClass>,
    ) -> Result<()> {
        let value = self.values.get_mut(&value_id).ok_or_else(|| {
            Error::Msg(alloc::format!(
                "cannot set metadata for unknown graph value {value_id}"
            ))
        })?;
        value.device = device;
        value.layout = layout;
        Ok(())
    }

    /// Adds a node validating references exist.
    pub fn add_node(
        &mut self,
        operation: OperationKind,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        attributes: BTreeMap<String, AttributeValue>,
    ) -> NodeId {
        self.add_node_with_identity(
            OperationIdentity::Builtin(operation),
            inputs,
            outputs,
            attributes,
        )
    }

    /// Adds a node with an explicit identity override.
    pub fn add_node_with_identity(
        &mut self,
        operation: OperationIdentity,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        attributes: BTreeMap<String, AttributeValue>,
    ) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        let execution_site = operation.execution_site();
        self.nodes.push(Node {
            id,
            operation,
            execution_site,
            inputs,
            outputs,
            attributes,
            descriptor_payload: None,
        });
        id
    }

    /// Adds a node carrying an encoded descriptor payload.
    pub fn add_node_with_descriptor_payload(
        &mut self,
        operation: OperationIdentity,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        attributes: BTreeMap<String, AttributeValue>,
        descriptor_payload: Option<DescriptorPayload>,
    ) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        let execution_site = operation.execution_site();
        self.nodes.push(Node {
            id,
            operation,
            execution_site,
            inputs,
            outputs,
            attributes,
            descriptor_payload,
        });
        id
    }

    /// Marks a value as a graph input.
    pub fn mark_input(&mut self, value_id: ValueId) {
        if !self.inputs.contains(&value_id) {
            self.inputs.push(value_id);
        }
        if let Some(shape) = self.values.get(&value_id).map(|value| value.shape.clone()) {
            let shape = ShapeExpr::symbolic(&shape, 1);
            let remapped = self.remap_input_symbols(shape);
            if let Some(value) = self.values.get_mut(&value_id) {
                value.shape_expr = remapped;
            }
        }
    }

    /// Marks an input while retaining the typed frontend shape proof that
    /// produced it. Runtime axes are assigned graph-local symbols; static
    /// axes remain constants and therefore do not receive redundant guards.
    pub fn mark_input_with_shape<S>(&mut self, value_id: ValueId)
    where
        S: crate::shapes::Shape + crate::exec::shape_projection::ShapeProjection,
    {
        if !self.inputs.contains(&value_id) {
            self.inputs.push(value_id);
        }
        let shape_expr =
            self.remap_input_symbols(crate::exec::shape_projection::shape_expr::<S>(1));
        if let Some(value) = self.values.get_mut(&value_id) {
            value.shape_expr = shape_expr;
        }
    }

    /// Marks a value as a graph output.
    pub fn mark_output(&mut self, value_id: ValueId) {
        if !self.outputs.contains(&value_id) {
            self.outputs.push(value_id);
        }
    }

    fn remap_input_symbols(&mut self, expr: ShapeExpr) -> ShapeExpr {
        let mut anonymous = BTreeMap::new();
        let mut used = BTreeSet::new();
        for value in self.values.values() {
            collect_shape_symbols(&value.shape_expr, &mut used);
        }
        used.extend(self.named_symbols.values().copied());
        let mut next = self.next_symbol_id;
        let mut named_next = next;
        let remapped = remap_shape_symbols(
            expr,
            &mut anonymous,
            &mut self.named_symbols,
            &mut named_next,
            || loop {
                let candidate = crate::exec::SymbolId(next);
                next = next.saturating_add(1);
                if used.insert(candidate) {
                    break candidate;
                }
            },
        );
        self.next_symbol_id = next;
        remapped
    }
}

fn remap_shape_symbols<F: FnMut() -> crate::exec::SymbolId>(
    expr: ShapeExpr,
    anonymous: &mut BTreeMap<crate::exec::SymbolId, crate::exec::SymbolId>,
    names: &mut BTreeMap<String, crate::exec::SymbolId>,
    next_id: &mut u32,
    mut fresh: F,
) -> ShapeExpr {
    fn dim(
        expr: crate::exec::DimExpr,
        anonymous: &mut BTreeMap<crate::exec::SymbolId, crate::exec::SymbolId>,
        names: &mut BTreeMap<String, crate::exec::SymbolId>,
        next_id: &mut u32,
        fresh: &mut impl FnMut() -> crate::exec::SymbolId,
    ) -> crate::exec::DimExpr {
        use crate::exec::DimExpr;
        match expr {
            DimExpr::Fresh(source) => {
                let source = crate::exec::SymbolId(source);
                let mapped = if let Some(mapped) = anonymous.get(&source) {
                    *mapped
                } else {
                    let mapped = fresh();
                    anonymous.insert(source, mapped);
                    mapped
                };
                DimExpr::Symbol(mapped)
            }
            DimExpr::Symbol(id) => {
                let mapped = if let Some(mapped) = anonymous.get(&id) {
                    *mapped
                } else {
                    let mapped = fresh();
                    anonymous.insert(id, mapped);
                    mapped
                };
                DimExpr::Symbol(mapped)
            }
            DimExpr::NamedSymbol { name, identity, .. } => {
                let id = if let Some(id) = names.get(&identity) {
                    *id
                } else {
                    let id = fresh();
                    *next_id = (*next_id).max(id.0.saturating_add(1));
                    names.insert(identity.clone(), id);
                    id
                };
                DimExpr::NamedSymbol { id, name, identity }
            }
            DimExpr::NamedFresh {
                source,
                name,
                identity,
            } => {
                let id = if let Some(id) = names.get(&identity) {
                    *id
                } else {
                    let id = fresh();
                    *next_id = (*next_id).max(id.0.saturating_add(1));
                    names.insert(identity.clone(), id);
                    id
                };
                let _ = source;
                DimExpr::NamedSymbol { id, name, identity }
            }
            DimExpr::NamedExpr {
                expr,
                id: _,
                name,
                identity,
            } => {
                let id = if let Some(id) = names.get(&identity) {
                    *id
                } else {
                    let id = fresh();
                    *next_id = (*next_id).max(id.0.saturating_add(1));
                    names.insert(identity.clone(), id);
                    id
                };
                let expr = dim(*expr, anonymous, names, next_id, fresh);
                match expr {
                    DimExpr::Symbol(_) | DimExpr::NamedSymbol { .. } => {
                        DimExpr::NamedSymbol { id, name, identity }
                    }
                    DimExpr::Const(value) => DimExpr::NamedExpr {
                        expr: Box::new(DimExpr::Const(value)),
                        id,
                        name,
                        identity,
                    },
                    expr => DimExpr::NamedExpr {
                        expr: Box::new(expr),
                        id,
                        name,
                        identity,
                    },
                }
            }
            DimExpr::Add(lhs, rhs) => DimExpr::Add(
                Box::new(dim(*lhs, anonymous, names, next_id, fresh)),
                Box::new(dim(*rhs, anonymous, names, next_id, fresh)),
            ),
            DimExpr::Sub(lhs, rhs) => DimExpr::Sub(
                Box::new(dim(*lhs, anonymous, names, next_id, fresh)),
                Box::new(dim(*rhs, anonymous, names, next_id, fresh)),
            ),
            DimExpr::Mul(lhs, rhs) => DimExpr::Mul(
                Box::new(dim(*lhs, anonymous, names, next_id, fresh)),
                Box::new(dim(*rhs, anonymous, names, next_id, fresh)),
            ),
            DimExpr::ExactDiv(lhs, rhs) => DimExpr::ExactDiv(
                Box::new(dim(*lhs, anonymous, names, next_id, fresh)),
                Box::new(dim(*rhs, anonymous, names, next_id, fresh)),
            ),
            DimExpr::Broadcast(lhs, rhs) => DimExpr::Broadcast(
                Box::new(dim(*lhs, anonymous, names, next_id, fresh)),
                Box::new(dim(*rhs, anonymous, names, next_id, fresh)),
            ),
            DimExpr::Min(lhs, rhs) => DimExpr::Min(
                Box::new(dim(*lhs, anonymous, names, next_id, fresh)),
                Box::new(dim(*rhs, anonymous, names, next_id, fresh)),
            ),
            DimExpr::Max(lhs, rhs) => DimExpr::Max(
                Box::new(dim(*lhs, anonymous, names, next_id, fresh)),
                Box::new(dim(*rhs, anonymous, names, next_id, fresh)),
            ),
            other => other,
        }
    }

    ShapeExpr {
        rank: expr.rank,
        dims: expr
            .dims
            .into_iter()
            .map(|value| dim(value, anonymous, names, next_id, &mut fresh))
            .collect(),
        constraints: expr
            .constraints
            .into_iter()
            .map(|constraint| match constraint {
                crate::exec::Constraint::Equal { lhs, rhs } => crate::exec::Constraint::Equal {
                    lhs: dim(lhs, anonymous, names, next_id, &mut fresh),
                    rhs: dim(rhs, anonymous, names, next_id, &mut fresh),
                },
                crate::exec::Constraint::BroadcastCompatible { lhs, rhs } => {
                    crate::exec::Constraint::BroadcastCompatible {
                        lhs: dim(lhs, anonymous, names, next_id, &mut fresh),
                        rhs: dim(rhs, anonymous, names, next_id, &mut fresh),
                    }
                }
                crate::exec::Constraint::LowerBound { value, bound } => {
                    crate::exec::Constraint::LowerBound {
                        value: dim(value, anonymous, names, next_id, &mut fresh),
                        bound,
                    }
                }
                crate::exec::Constraint::UpperBound { value, bound } => {
                    crate::exec::Constraint::UpperBound {
                        value: dim(value, anonymous, names, next_id, &mut fresh),
                        bound,
                    }
                }
                crate::exec::Constraint::Divisible { value, divisor } => {
                    crate::exec::Constraint::Divisible {
                        value: dim(value, anonymous, names, next_id, &mut fresh),
                        divisor,
                    }
                }
            })
            .collect(),
    }
}

fn collect_shape_symbols(expr: &ShapeExpr, symbols: &mut BTreeSet<crate::exec::SymbolId>) {
    fn dim(expr: &crate::exec::DimExpr, symbols: &mut BTreeSet<crate::exec::SymbolId>) {
        match expr {
            crate::exec::DimExpr::Symbol(id) | crate::exec::DimExpr::NamedSymbol { id, .. } => {
                symbols.insert(*id);
            }
            crate::exec::DimExpr::Add(lhs, rhs)
            | crate::exec::DimExpr::Sub(lhs, rhs)
            | crate::exec::DimExpr::Mul(lhs, rhs)
            | crate::exec::DimExpr::ExactDiv(lhs, rhs)
            | crate::exec::DimExpr::Broadcast(lhs, rhs)
            | crate::exec::DimExpr::Min(lhs, rhs)
            | crate::exec::DimExpr::Max(lhs, rhs) => {
                dim(lhs, symbols);
                dim(rhs, symbols);
            }
            crate::exec::DimExpr::NamedExpr { expr, .. } => dim(expr, symbols),
            crate::exec::DimExpr::Const(_)
            | crate::exec::DimExpr::Fresh(_)
            | crate::exec::DimExpr::NamedFresh { .. }
            | crate::exec::DimExpr::Unknown => {}
        }
    }
    for value in &expr.dims {
        dim(value, symbols);
    }
    for constraint in &expr.constraints {
        match constraint {
            crate::exec::Constraint::Equal { lhs, rhs }
            | crate::exec::Constraint::BroadcastCompatible { lhs, rhs } => {
                dim(lhs, symbols);
                dim(rhs, symbols);
            }
            crate::exec::Constraint::LowerBound { value, .. }
            | crate::exec::Constraint::UpperBound { value, .. }
            | crate::exec::Constraint::Divisible { value, .. } => dim(value, symbols),
        }
    }
}
