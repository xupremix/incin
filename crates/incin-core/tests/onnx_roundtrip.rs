//! Structural round-trip properties for the ONNX writer/reader pair
//! (parser-fuzzing slice of #48).
//!
//! `export_to_onnx` followed by `OnnxImporter::import` is only a round trip
//! for graphs the file format can carry end to end, so the generator builds
//! ONNX-file-valid graphs rather than arbitrary [`Graph`]s:
//!
//! - every op is drawn from the subset where `operation_from_onnx`
//!   inverts `onnx_name` back to the same `OperationKind` once the rank
//!   rules for `MatMul` are satisfied (the legacy `Unsqueeze` kind is
//!   excluded: the importer intentionally mints `UnsqueezeExact`, which
//!   the forward table cannot export again);
//! - every node output has shape/dtype in the file, because
//!   `import_from_onnx` performs no shape inference and refuses a node
//!   output it cannot look up - the exporter's `value_info` section is
//!   what makes a multi-node chain survive;
//! - `inputs` and `initializers` stay disjoint, matching the wiring the
//!   importer produces when a name appears in both sections.
//!
//! The compared projection is deliberately keyed by the names the file
//! carries (the exporter names every tensor after its `ValueId`, and
//! `Value::name` is not part of the ONNX contract), and covers inputs,
//! outputs, initializers, node wiring, attributes, and every value's
//! shape/dtype - the structural contract the importer's documentation
//! promises, with no numeric evaluation and no shape re-derivation.

use std::collections::BTreeMap;

use incin_core::graph::{AttributeValue, Graph, Value, ValueId};
use incin_core::onnx::{OnnxImporter, export_to_onnx};
use incin_core::prelude::{DTypeDescriptor, DTypeId, OperationKind};
use proptest::prelude::*;
use tempfile::tempdir;

// --- dtype vocabulary ----------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum DClass {
    Float,
    Int,
    Bool,
}

fn class_of(dtype: DTypeId) -> DClass {
    match dtype {
        DTypeId::F32 | DTypeId::F64 | DTypeId::F16 | DTypeId::BF16 => DClass::Float,
        DTypeId::U8 | DTypeId::U32 | DTypeId::I64 => DClass::Int,
        DTypeId::Bool => DClass::Bool,
        // `dtype_to_onnx` refuses `Q8_0`, so no round-trip graph can carry
        // one; the arm exists only to keep the match exhaustive. `DTypeId`
        // is `non_exhaustive`, so any future variant lands here too - it
        // will not be in a generated graph unless the strategy learns it.
        _ => DClass::Int,
    }
}

/// Element width for initializer payloads. The importer copies `raw_data`
/// through without validating it against the declared shape, so this only
/// needs to be plausible, not computed from a storage descriptor.
fn elem_size(dtype: DTypeId) -> usize {
    match dtype {
        DTypeId::U8 | DTypeId::Bool => 1,
        DTypeId::F16 | DTypeId::BF16 => 2,
        DTypeId::F32 | DTypeId::U32 => 4,
        DTypeId::F64 | DTypeId::I64 => 8,
        // Unreachable for generated graphs (see `class_of`); zero-length
        // payloads are still well-formed protobuf, so a future dtype would
        // round-trip rather than panic.
        _ => 0,
    }
}

fn exportable_float() -> impl Strategy<Value = DTypeId> {
    prop_oneof![
        6 => Just(DTypeId::F32),
        1 => Just(DTypeId::F64),
        1 => Just(DTypeId::F16),
        1 => Just(DTypeId::BF16),
    ]
}

fn exportable_int() -> impl Strategy<Value = DTypeId> {
    prop_oneof![3 => Just(DTypeId::I64), 1 => Just(DTypeId::U8), 1 => Just(DTypeId::U32)]
}

// --- graph specification -------------------------------------------------

/// One chain step: an op whose shape rules the builder can satisfy from
/// whatever the pool already holds, plus a selection seed that decides
/// which compatible operand the step consumes (and, where the op needs
/// one, its attribute payload).
#[derive(Debug, Clone, Copy)]
struct Step {
    op: OperationKind,
    sel: u8,
}

#[derive(Debug)]
struct Spec {
    a: usize,
    b: usize,
    n: usize,
    first_float: DTypeId,
    second_float: DTypeId,
    an_int: DTypeId,
    initializers: Vec<(Vec<usize>, DTypeId)>,
    steps: Vec<Step>,
    /// Indices into the final value pool, reduced modulo its length, that
    /// become graph outputs.
    outputs: Vec<u8>,
}

fn small_shape(max_rank: usize) -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(1usize..=4, 0..=max_rank)
}

fn step_op() -> impl Strategy<Value = OperationKind> {
    prop_oneof![
        6 => prop::sample::select(vec![
            OperationKind::Relu,
            OperationKind::Exp,
            OperationKind::Neg,
            OperationKind::Sqrt,
            OperationKind::Log,
            OperationKind::Tanh,
            OperationKind::Sigmoid,
            OperationKind::LogSoftmax,
        ]),
        6 => prop::sample::select(vec![
            OperationKind::Add,
            OperationKind::Sub,
            OperationKind::Mul,
            OperationKind::Div,
            OperationKind::Maximum,
            OperationKind::Minimum,
        ]),
        3 => prop::sample::select(vec![
            OperationKind::CmpEq,
            OperationKind::CmpLt,
            OperationKind::CmpLe,
            OperationKind::CmpGt,
            OperationKind::CmpGe,
        ]),
        2 => prop::sample::select(vec![
            OperationKind::LogicalAnd,
            OperationKind::LogicalOr,
            OperationKind::LogicalNot,
        ]),
        3 => prop::sample::select(vec![
            OperationKind::MatMulExact,
            OperationKind::BatchedMatMul,
            OperationKind::Softmax,
        ]),
        2 => prop::sample::select(vec![
            OperationKind::Triu,
            OperationKind::Tril,
            OperationKind::TransposeExact,
            OperationKind::ConcatExact,
            OperationKind::WhereCond,
        ]),
    ]
}

fn spec_strategy() -> impl Strategy<Value = Spec> {
    (
        1usize..=4,
        1usize..=4,
        1usize..=4,
        exportable_float(),
        exportable_float(),
        exportable_int(),
        proptest::collection::vec(
            (
                small_shape(2),
                prop_oneof![3 => exportable_float(), 1 => exportable_int()],
            ),
            0..=2,
        ),
        proptest::collection::vec((step_op(), any::<u8>()), 0..=5),
        proptest::collection::vec(any::<u8>(), 1..=3),
    )
        .prop_map(
            |(a, b, n, first_float, second_float, an_int, initializers, steps, outputs)| Spec {
                a,
                b,
                n,
                first_float,
                second_float,
                an_int,
                initializers,
                steps: steps
                    .into_iter()
                    .map(|(op, sel)| Step { op, sel })
                    .collect(),
                outputs,
            },
        )
}

// --- builder -------------------------------------------------------------

#[derive(Clone)]
struct Entry {
    id: ValueId,
    shape: Vec<usize>,
    dtype: DTypeId,
}

fn push_value(
    graph: &mut Graph,
    pool: &mut Vec<Entry>,
    shape: Vec<usize>,
    dtype: DTypeId,
) -> ValueId {
    let id = graph.add_value(shape.clone(), dtype, None);
    pool.push(Entry { id, shape, dtype });
    id
}

fn pick(pool: &[Entry], sel: u8, usable: impl Fn(&Entry) -> bool) -> Option<Entry> {
    let candidates: Vec<&Entry> = pool.iter().filter(|entry| usable(entry)).collect();
    if candidates.is_empty() {
        return None;
    }
    Some(candidates[usize::from(sel) % candidates.len()].clone())
}

fn ensure_operand(
    graph: &mut Graph,
    pool: &mut Vec<Entry>,
    shape: Vec<usize>,
    dtype: DTypeId,
    sel: u8,
) -> ValueId {
    if let Some(existing) = pick(pool, sel, |entry| {
        entry.shape == shape && entry.dtype == dtype
    }) {
        return existing.id;
    }
    let numel = shape.iter().product::<usize>();
    let id = push_value(graph, pool, shape, dtype);
    graph
        .initializers
        .insert(id, vec![0u8; numel * elem_size(dtype)]);
    id
}

/// Builds the [`Graph`] a [`Spec`] describes. Every `pick` below has a
/// witness among the five fixed inputs (float `[a,b]`, float `[a,b]`,
/// float `[n,a,b]`, bool `[a,b]`, int `[a,b]`), so no step is ever
/// silently skipped and coverage of each op the strategy emits is real.
fn build(spec: &Spec) -> Graph {
    let mut graph = Graph::new();
    let mut pool = Vec::new();

    let inputs = [
        push_value(
            &mut graph,
            &mut pool,
            vec![spec.a, spec.b],
            spec.first_float,
        ),
        push_value(
            &mut graph,
            &mut pool,
            vec![spec.a, spec.b],
            spec.second_float,
        ),
        push_value(
            &mut graph,
            &mut pool,
            vec![spec.n, spec.a, spec.b],
            spec.first_float,
        ),
        push_value(&mut graph, &mut pool, vec![spec.a, spec.b], DTypeId::Bool),
        push_value(&mut graph, &mut pool, vec![spec.a, spec.b], spec.an_int),
    ];
    for &id in &inputs {
        graph.mark_input(id);
    }

    // Initializers are deliberately *not* graph inputs: the importer binds
    // a name listed in both sections as an initializer only, which would
    // drop it from `inputs` and break the structural comparison.
    for (shape, dtype) in &spec.initializers {
        let numel = shape.iter().product::<usize>();
        let id = push_value(&mut graph, &mut pool, shape.clone(), *dtype);
        graph
            .initializers
            .insert(id, vec![0u8; numel * elem_size(*dtype)]);
    }

    let numeric = |entry: &Entry| class_of(entry.dtype) != DClass::Bool;

    for step in &spec.steps {
        let Step { op, sel } = *step;
        let (node_inputs, out_shape, out_dtype, attributes) = match op {
            OperationKind::Softmax => {
                let source = pick(&pool, sel, |entry| {
                    class_of(entry.dtype) == DClass::Float && !entry.shape.is_empty()
                })
                .expect("the first float input is a permanent witness");
                let mut attributes = BTreeMap::new();
                // The ONNX default spelled out; valid for any rank >= 1.
                attributes.insert(String::from("axis"), AttributeValue::Int(-1));
                (
                    vec![source.id],
                    source.shape.clone(),
                    source.dtype,
                    attributes,
                )
            }
            OperationKind::Relu
            | OperationKind::Exp
            | OperationKind::Neg
            | OperationKind::Sqrt
            | OperationKind::Log
            | OperationKind::Tanh
            | OperationKind::Sigmoid
            | OperationKind::LogSoftmax => {
                let source = pick(&pool, sel, |entry| class_of(entry.dtype) == DClass::Float)
                    .expect("the first float input is a permanent witness");
                (
                    vec![source.id],
                    source.shape.clone(),
                    source.dtype,
                    BTreeMap::new(),
                )
            }
            OperationKind::LogicalNot => {
                let source = pick(&pool, sel, |entry| class_of(entry.dtype) == DClass::Bool)
                    .expect("the bool input is a permanent witness");
                (
                    vec![source.id],
                    source.shape.clone(),
                    source.dtype,
                    BTreeMap::new(),
                )
            }
            OperationKind::Add
            | OperationKind::Sub
            | OperationKind::Mul
            | OperationKind::Div
            | OperationKind::Maximum
            | OperationKind::Minimum => {
                let (lhs, rhs) = same_shape_pair(&pool, sel, numeric);
                (
                    vec![lhs.id, rhs.id],
                    lhs.shape.clone(),
                    lhs.dtype,
                    BTreeMap::new(),
                )
            }
            OperationKind::CmpEq
            | OperationKind::CmpLt
            | OperationKind::CmpLe
            | OperationKind::CmpGt
            | OperationKind::CmpGe => {
                let (lhs, rhs) = same_shape_pair(&pool, sel, numeric);
                (
                    vec![lhs.id, rhs.id],
                    lhs.shape.clone(),
                    DTypeId::Bool,
                    BTreeMap::new(),
                )
            }
            OperationKind::LogicalAnd | OperationKind::LogicalOr => {
                let (lhs, rhs) =
                    same_shape_pair(&pool, sel, |entry| class_of(entry.dtype) == DClass::Bool);
                (
                    vec![lhs.id, rhs.id],
                    lhs.shape.clone(),
                    lhs.dtype,
                    BTreeMap::new(),
                )
            }
            OperationKind::MatMulExact => {
                let lhs = pick(&pool, sel, |entry| {
                    class_of(entry.dtype) == DClass::Float && entry.shape.len() == 2
                })
                .expect("the [a, b] float input is a permanent witness");
                let rhs_shape = vec![lhs.shape[1], 1 + usize::from(sel % 4)];
                let out_shape = vec![lhs.shape[0], rhs_shape[1]];
                let rhs = ensure_operand(&mut graph, &mut pool, rhs_shape, lhs.dtype, sel);
                (vec![lhs.id, rhs], out_shape, lhs.dtype, BTreeMap::new())
            }
            OperationKind::BatchedMatMul => {
                let lhs = pick(&pool, sel, |entry| {
                    class_of(entry.dtype) == DClass::Float && entry.shape.len() == 3
                })
                .expect("the [n, a, b] float input is a permanent witness");
                let rhs_shape = vec![lhs.shape[0], lhs.shape[2], 1 + usize::from(sel % 4)];
                let out_shape = vec![lhs.shape[0], lhs.shape[1], rhs_shape[2]];
                let rhs = ensure_operand(&mut graph, &mut pool, rhs_shape, lhs.dtype, sel);
                (vec![lhs.id, rhs], out_shape, lhs.dtype, BTreeMap::new())
            }
            OperationKind::Triu | OperationKind::Tril => {
                let source = pick(&pool, sel, |entry| numeric(entry) && entry.shape.len() >= 2)
                    .expect("the [a, b] float input is a permanent witness");
                let mut attributes = BTreeMap::new();
                let upper = if op == OperationKind::Triu { 1 } else { 0 };
                attributes.insert(String::from("upper"), AttributeValue::Int(upper));
                (
                    vec![source.id],
                    source.shape.clone(),
                    source.dtype,
                    attributes,
                )
            }
            OperationKind::TransposeExact => {
                let source = pick(&pool, sel, |entry| !entry.shape.is_empty())
                    .expect("the [a, b] float input is a permanent witness");
                let rank = source.shape.len();
                let shift = usize::from(sel) % rank;
                let perm: Vec<i64> = (0..rank)
                    .map(|axis| ((axis + shift) % rank) as i64)
                    .collect();
                let out_shape: Vec<usize> = perm
                    .iter()
                    .map(|&axis| source.shape[axis as usize])
                    .collect();
                let mut attributes = BTreeMap::new();
                attributes.insert(String::from("perm"), AttributeValue::Ints(perm));
                (vec![source.id], out_shape, source.dtype, attributes)
            }
            OperationKind::ConcatExact => {
                let (lhs, rhs) = same_shape_pair(&pool, sel, |entry| !entry.shape.is_empty());
                let rank = lhs.shape.len();
                let axis = usize::from(sel) % rank;
                let mut out_shape = lhs.shape.clone();
                out_shape[axis] *= 2;
                let mut attributes = BTreeMap::new();
                attributes.insert(String::from("axis"), AttributeValue::Int(axis as i64));
                (vec![lhs.id, rhs.id], out_shape, lhs.dtype, attributes)
            }
            OperationKind::WhereCond => {
                let cond = pick(&pool, sel, |entry| class_of(entry.dtype) == DClass::Bool)
                    .expect("the bool input is a permanent witness");
                // ONNX `Where` broadcasts `cond` against `X`/`Y`, which
                // must share one type; preferring a same-shaped numeric
                // operand and falling back to the condition itself keeps
                // every generated node inside the spec's rules.
                let value = pick(&pool, sel, |entry| {
                    numeric(entry) && entry.shape == cond.shape
                })
                .unwrap_or_else(|| cond.clone());
                (
                    vec![cond.id, value.id, value.id],
                    value.shape.clone(),
                    value.dtype,
                    BTreeMap::new(),
                )
            }
            other => panic!(
                "step strategy emitted {}, which the builder does not model",
                other.name()
            ),
        };

        let out_id = push_value(&mut graph, &mut pool, out_shape, out_dtype);
        graph.add_node(op, node_inputs, vec![out_id], attributes);
    }

    for &output_pick in &spec.outputs {
        let entry = &pool[usize::from(output_pick) % pool.len()];
        graph.mark_output(entry.id);
    }

    graph
}

/// Two pool entries with identical shape and dtype, falling back to using
/// one entry as both operands (self-addition and friends are legal ONNX).
fn same_shape_pair(pool: &[Entry], sel: u8, usable: impl Fn(&Entry) -> bool) -> (Entry, Entry) {
    let lhs = pick(pool, sel, &usable).expect("the fixed inputs always witness a step op");
    let rhs = pick(pool, sel.wrapping_add(1), |entry| {
        usable(entry) && entry.id != lhs.id && entry.shape == lhs.shape && entry.dtype == lhs.dtype
    })
    .unwrap_or(lhs.clone());
    (lhs, rhs)
}

// --- structural projection ------------------------------------------------

#[derive(Debug, PartialEq)]
struct TensorView {
    name: String,
    shape: Vec<usize>,
    dtype: DTypeDescriptor,
}

#[derive(Debug, PartialEq)]
struct NodeView {
    op: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: BTreeMap<String, AttributeValue>,
}

#[derive(Debug, PartialEq)]
struct GraphView {
    inputs: Vec<TensorView>,
    outputs: Vec<TensorView>,
    initializers: BTreeMap<String, (Vec<usize>, DTypeDescriptor, Vec<u8>)>,
    nodes: Vec<NodeView>,
    values: BTreeMap<String, (Vec<usize>, DTypeDescriptor)>,
}

/// Names a value the way the file will carry it. Export keys every name by
/// `ValueId`; import hands those same names back as `Value::name`.
fn file_name(value: &Value, from_import: bool) -> String {
    if from_import {
        value
            .name
            .clone()
            .expect("the importer always binds values under a file name")
    } else {
        value.id.to_string()
    }
}

fn view(graph: &Graph, from_import: bool) -> GraphView {
    let name = |id: ValueId| -> String {
        let value = graph
            .values
            .get(&id)
            .expect("viewed graphs only reference bound values");
        file_name(value, from_import)
    };
    let tensor = |id: ValueId| -> TensorView {
        let value = &graph.values[&id];
        TensorView {
            name: file_name(value, from_import),
            shape: value.shape.clone(),
            dtype: value.dtype,
        }
    };

    GraphView {
        inputs: graph.inputs.iter().map(|&id| tensor(id)).collect(),
        outputs: graph.outputs.iter().map(|&id| tensor(id)).collect(),
        initializers: graph
            .initializers
            .iter()
            .map(|(id, bytes)| {
                let value = &graph.values[id];
                (
                    file_name(value, from_import),
                    (value.shape.clone(), value.dtype, bytes.clone()),
                )
            })
            .collect(),
        nodes: graph
            .nodes
            .iter()
            .map(|node| NodeView {
                op: format!("{:?}", node.operation),
                inputs: node.inputs.iter().map(|&id| name(id)).collect(),
                outputs: node.outputs.iter().map(|&id| name(id)).collect(),
                attributes: node.attributes.clone(),
            })
            .collect(),
        values: graph
            .values
            .values()
            .map(|value| {
                (
                    file_name(value, from_import),
                    (value.shape.clone(), value.dtype),
                )
            })
            .collect(),
    }
}

// --- properties ----------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        max_shrink_iters: 4096,
        ..ProptestConfig::default()
    })]

    /// The load-bearing property of issue #48's ONNX slice: a graph built
    /// from file-valid pieces survives `export -> import` with its inputs,
    /// initializers, nodes (identity, wiring, attributes), outputs, and
    /// every value's shape/dtype intact.
    #[test]
    fn export_then_import_preserves_graph_structure(spec in spec_strategy()) {
        let graph = build(&spec);
        let dir = tempdir().expect("temp dir");
        let path = dir.path().join("roundtrip.onnx");

        export_to_onnx(&graph, &path)
            .map_err(|error| TestCaseError::fail(format!("export failed: {error:#}")))?;
        let imported = OnnxImporter::new(&path)
            .import()
            .map_err(|error| TestCaseError::fail(format!("import failed: {error:#}")))?;

        prop_assert_eq!(view(&graph, false), view(&imported, true));
    }
}

// --- deterministic pins ---------------------------------------------------

/// The regression the `value_info` export section exists for: before it,
/// a two-node chain lost the shape/dtype of the tensor between the nodes
/// and the importer refused its own exporter's file.
#[test]
fn a_multi_node_chain_round_trips_through_intermediate_value_info() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 3], DTypeId::F32, None);
    let t = graph.add_value(vec![2, 3], DTypeId::F32, None);
    let y = graph.add_value(vec![2, 3], DTypeId::F32, None);
    graph.mark_input(x);
    graph.add_node(OperationKind::Relu, vec![x], vec![t], Default::default());
    graph.add_node(OperationKind::Neg, vec![t], vec![y], Default::default());
    graph.mark_output(y);

    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("chain.onnx");
    export_to_onnx(&graph, &path).expect("chain should export");
    let imported = OnnxImporter::new(&path)
        .import()
        .expect("chain should import");

    assert_eq!(view(&graph, false), view(&imported, true));
    assert_eq!(imported.nodes.len(), 2);
    let intermediate = imported
        .values
        .values()
        .find(|value| value.name.as_deref() == Some("1"))
        .expect("the value between the nodes is named after its id");
    assert_eq!(intermediate.shape, vec![2, 3]);
}

/// `LogSoftmax` was the one op whose forward-table entry
/// (`onnx_name => "LogSoftmax"`) `operation_from_onnx` had no inverse for,
/// so a file written from it was refused on the way back in. The pin makes
/// that pairing deterministic; in the generator it only appears when
/// `step_op` happens to pick it.
#[test]
fn log_softmax_round_trips_through_the_inverse_table() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 3], DTypeId::F32, None);
    let y = graph.add_value(vec![2, 3], DTypeId::F32, None);
    graph.mark_input(x);
    graph.add_node(
        OperationKind::LogSoftmax,
        vec![x],
        vec![y],
        Default::default(),
    );
    graph.mark_output(y);

    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("logsoftmax.onnx");
    export_to_onnx(&graph, &path).expect("graph should export");
    let imported = OnnxImporter::new(&path)
        .import()
        .expect("LogSoftmax should map back to itself");

    assert_eq!(imported.nodes.len(), 1);
    assert_eq!(view(&graph, false), view(&imported, true));
}

/// Every `AttributeValue` variant has to survive the file, not just the
/// `Int`/`Ints` pair the schema-shaped ops in the generator happen to
/// use. Attribute *schema conformance* (whether `Softmax` may carry a
/// `tag`) is the graph author's business; the exporter's contract is a
/// lossless pass-through of what the graph already holds.
#[test]
fn every_attribute_value_kind_survives_the_round_trip() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 2], DTypeId::F32, None);
    let y = graph.add_value(vec![2, 2], DTypeId::F32, None);
    graph.mark_input(x);
    let mut attributes = BTreeMap::new();
    attributes.insert(String::from("axis"), AttributeValue::Int(-1));
    attributes.insert(String::from("scale"), AttributeValue::Float(0.5));
    attributes.insert(
        String::from("tag"),
        AttributeValue::String(String::from("incin")),
    );
    attributes.insert(String::from("perm"), AttributeValue::Ints(vec![1, 0]));
    attributes.insert(
        String::from("weights"),
        AttributeValue::Floats(vec![0.25, 0.75]),
    );
    graph.add_node(OperationKind::Softmax, vec![x], vec![y], attributes);
    graph.mark_output(y);

    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("attrs.onnx");
    export_to_onnx(&graph, &path).expect("graph should export");
    let imported = OnnxImporter::new(&path)
        .import()
        .expect("graph should import");

    assert_eq!(
        imported.nodes[0].attributes, graph.nodes[0].attributes,
        "all five attribute kinds must read back unchanged"
    );
}

/// An initializer stays out of `inputs` across the file even when it feeds
/// a node, and its raw bytes are byte-for-byte the ones exported.
#[test]
fn initializer_bytes_survive_and_stay_out_of_the_input_list() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 3], DTypeId::F32, None);
    let w = graph.add_value(vec![2, 3], DTypeId::F32, None);
    let y = graph.add_value(vec![2, 3], DTypeId::F32, None);
    graph.mark_input(x);
    let payload: Vec<u8> = (0..24).map(|byte| byte as u8).collect();
    graph.initializers.insert(w, payload.clone());
    graph.add_node(OperationKind::Add, vec![x, w], vec![y], Default::default());
    graph.mark_output(y);

    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("init.onnx");
    export_to_onnx(&graph, &path).expect("graph should export");
    let imported = OnnxImporter::new(&path)
        .import()
        .expect("graph should import");

    assert_eq!(
        imported.inputs.len(),
        1,
        "the initializer must not reappear as an input"
    );
    assert_eq!(view(&graph, false), view(&imported, true));
    let carried = imported
        .initializers
        .values()
        .next()
        .expect("one initializer");
    assert_eq!(carried, &payload);
}
