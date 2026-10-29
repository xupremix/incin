//! Adversarial-input coverage for the ONNX reader and writer (parser
//! fuzzing slice of #48; `docs/security/threat-model.md` names malformed
//! ONNX as an adversarial input and defers its coverage to this issue).
//!
//! The contract under test is fail-closed. `import_from_onnx` returns
//! `Err` for every byte string short of a coherent model and never
//! panics - a successful `ModelProto::decode` is only the first half, and
//! the graph reconstruction that follows has to refuse dangling
//! references, missing shapes, unsupported ops, and negative dimensions
//! rather than index or widen its way past them. On the write side,
//! `export_to_onnx` refuses the structurally broken `Graph`s a
//! deserializer can produce (public fields, arbitrary wiring) instead of
//! panicking on the caller's behalf.
//!
//! Cases come in three shapes: a `proptest` sweep over capped random
//! bytes, deterministic mutations of a valid export (every truncating
//! prefix, every single-byte flip), and hand-encoded protobufs that each
//! isolate one refusal - the hand-rolled encoder exists because the
//! protobuf types are crate-private, and because it can claim lengths no
//! real exporter would ever write.

use incin_core::graph::{AttributeValue, Graph};
use incin_core::onnx::{OnnxImporter, export_to_onnx};
use incin_core::prelude::{DTypeId, OperationKind};
use proptest::prelude::*;
use tempfile::tempdir;

// --- harness -------------------------------------------------------------

/// One scratch directory per test case. Files are rewritten in place, so a
/// whole sweep of mutations shares a single path without racing itself.
struct Case {
    dir: tempfile::TempDir,
}

impl Case {
    fn new() -> Self {
        Self {
            dir: tempdir().expect("temp dir"),
        }
    }

    fn path(&self) -> std::path::PathBuf {
        self.dir.path().join("case.onnx")
    }

    /// Imports `bytes` through the same path a caller of the public API
    /// takes. The error type is flattened to `String` because `anyhow` is
    /// not a dev-dependency of this crate; messages must still survive so
    /// tests can pin *which* refusal fired.
    fn import(&self, bytes: &[u8]) -> Result<Graph, String> {
        let path = self.path();
        std::fs::write(&path, bytes).expect("case file should be writable");
        OnnxImporter::new(&path)
            .import()
            .map_err(|error| error.to_string())
    }

    fn export(&self, graph: &Graph) -> Result<Vec<u8>, String> {
        let path = self.path();
        export_to_onnx(graph, &path).map_err(|error| error.to_string())?;
        Ok(std::fs::read(&path).expect("exported file should be readable"))
    }
}

/// A graph touching every section the exporter writes: two inputs worth of
/// chain (so `value_info` is populated), an initializer, two nodes, one
/// output. Used as the substrate for truncation and mutation sweeps.
fn fixture_graph() -> Graph {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 3], DTypeId::F32, None);
    let w = graph.add_value(vec![2, 3], DTypeId::F32, None);
    let t = graph.add_value(vec![2, 3], DTypeId::F32, None);
    let y = graph.add_value(vec![2, 3], DTypeId::F32, None);
    graph.mark_input(x);
    graph.initializers.insert(w, vec![7u8; 24]);
    graph.add_node(OperationKind::Relu, vec![x], vec![t], Default::default());
    graph.add_node(OperationKind::Add, vec![t, w], vec![y], Default::default());
    graph.mark_output(y);
    graph
}

/// The invariants any successfully imported graph must hold, regardless of
/// what mutation produced it: no half-bound ids anywhere.
fn assert_sane(graph: &Graph) {
    for &id in graph.inputs.iter().chain(&graph.outputs) {
        assert!(
            graph.values.contains_key(&id),
            "unbound graph boundary value {id}"
        );
    }
    for id in graph.initializers.keys() {
        assert!(
            graph.values.contains_key(id),
            "unbound initializer value {id}"
        );
    }
    for node in &graph.nodes {
        for &id in node.inputs.iter().chain(&node.outputs) {
            assert!(
                graph.values.contains_key(&id),
                "node {:?} references unbound value {id}",
                node.operation
            );
        }
    }
}

// --- minimal protobuf encoding -------------------------------------------

fn pb_varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

fn pb_key(field: u32, wire: u8) -> Vec<u8> {
    pb_varint(u64::from((field << 3) | u32::from(wire)))
}

fn pb_varint_field(field: u32, value: u64) -> Vec<u8> {
    let mut out = pb_key(field, 0);
    out.extend(pb_varint(value));
    out
}

fn pb_bytes_field(field: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = pb_key(field, 2);
    out.extend(pb_varint(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

fn pb_string_field(field: u32, value: &str) -> Vec<u8> {
    pb_bytes_field(field, value.as_bytes())
}

/// `ValueInfoProto` with a tensor type whose dims are all concrete.
/// Negative dims are written as two's-complement `int64`, exactly as any
/// protobuf encoder would emit them.
fn pb_value_info(name: &str, elem_type: i64, dims: &[i64]) -> Vec<u8> {
    let mut shape = Vec::new();
    for &dim in dims {
        shape.extend(pb_bytes_field(1, &pb_varint_field(1, dim as u64)));
    }
    pb_value_info_shape(name, elem_type, &shape)
}

/// `ValueInfoProto` whose single dim is symbolic (`dim_param`).
fn pb_value_info_symbolic(name: &str, elem_type: i64, param: &str) -> Vec<u8> {
    let dim = pb_string_field(2, param);
    pb_value_info_shape(name, elem_type, &pb_bytes_field(1, &dim))
}

/// `ValueInfoProto` with a tensor type but no shape at all.
fn pb_value_info_unshaped(name: &str, elem_type: i64) -> Vec<u8> {
    let tensor = pb_varint_field(1, elem_type as u64);
    let mut out = pb_string_field(1, name);
    out.extend(pb_bytes_field(2, &pb_bytes_field(1, &tensor)));
    out
}

fn pb_value_info_shape(name: &str, elem_type: i64, shape_payload: &[u8]) -> Vec<u8> {
    let mut tensor = pb_varint_field(1, elem_type as u64);
    tensor.extend(pb_bytes_field(2, shape_payload));
    let mut out = pb_string_field(1, name);
    out.extend(pb_bytes_field(2, &pb_bytes_field(1, &tensor)));
    out
}

fn pb_node(op_type: &str, inputs: &[&str], outputs: &[&str], attributes: &[Vec<u8>]) -> Vec<u8> {
    let mut node = Vec::new();
    for input in inputs {
        node.extend(pb_string_field(1, input));
    }
    for output in outputs {
        node.extend(pb_string_field(2, output));
    }
    node.extend(pb_string_field(4, op_type));
    for attribute in attributes {
        node.extend(pb_bytes_field(5, attribute));
    }
    node
}

fn pb_attribute(name: &str, kind: i64, payload: &[u8]) -> Vec<u8> {
    let mut out = pb_string_field(1, name);
    out.extend(pb_varint_field(20, kind as u64));
    out.extend_from_slice(payload);
    out
}

/// An attribute with no `type` field at all - the legacy-IR shape this
/// reader deliberately refuses rather than resolving with a has-field
/// heuristic.
fn pb_attribute_untyped(name: &str, payload: &[u8]) -> Vec<u8> {
    let mut out = pb_string_field(1, name);
    out.extend_from_slice(payload);
    out
}

fn pb_tensor(
    name: Option<&str>,
    dims: &[i64],
    data_type: Option<i64>,
    raw: Option<&[u8]>,
) -> Vec<u8> {
    let mut out = Vec::new();
    for &dim in dims {
        out.extend(pb_varint_field(1, dim as u64));
    }
    if let Some(code) = data_type {
        out.extend(pb_varint_field(2, code as u64));
    }
    if let Some(name) = name {
        out.extend(pb_string_field(8, name));
    }
    if let Some(raw) = raw {
        out.extend(pb_bytes_field(9, raw));
    }
    out
}

fn pb_graph(
    nodes: &[Vec<u8>],
    initializers: &[Vec<u8>],
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
    value_info: &[Vec<u8>],
) -> Vec<u8> {
    let mut graph = Vec::new();
    for node in nodes {
        graph.extend(pb_bytes_field(1, node));
    }
    for initializer in initializers {
        graph.extend(pb_bytes_field(5, initializer));
    }
    for input in inputs {
        graph.extend(pb_bytes_field(11, input));
    }
    for output in outputs {
        graph.extend(pb_bytes_field(12, output));
    }
    for info in value_info {
        graph.extend(pb_bytes_field(13, info));
    }
    graph
}

fn wrap_model(graph: &[u8]) -> Vec<u8> {
    pb_bytes_field(7, graph)
}

// ONNX `tensor_proto.DataType` codes this reader knows or must refuse.
const ONNX_FLOAT: i64 = 1;
const ONNX_INT32: i64 = 6;
// ONNX `attribute_proto.AttributeType` codes.
const ATTR_GRAPH: i64 = 5;
const ATTR_STRING: i64 = 3;
const ATTR_INT: i64 = 2;

/// Each case isolates one refusal. The graph section is otherwise valid,
/// so a passing `Err` proves the named defect is what stopped the import
/// rather than some incidental damage earlier in the pipeline.
fn broken_models() -> Vec<(&'static str, Vec<u8>)> {
    let x = pb_value_info("x", ONNX_FLOAT, &[2, 3]);
    let y = pb_value_info("y", ONNX_FLOAT, &[2, 3]);
    let cases: Vec<(&'static str, Vec<u8>)> = vec![
        (
            "a model whose only field is a producer name",
            pb_string_field(2, "abc"),
        ),
        (
            "a node referencing an input nothing defined",
            wrap_model(&pb_graph(
                &[pb_node("Relu", &["ghost"], &["y"], &[])],
                &[],
                &[],
                &[],
                std::slice::from_ref(&y),
            )),
        ),
        (
            "a node output with no value_info anywhere",
            wrap_model(&pb_graph(
                &[pb_node("Relu", &["x"], &["y"], &[])],
                &[],
                std::slice::from_ref(&x),
                &[],
                &[],
            )),
        ),
        (
            "an op_type outside the forward mapping",
            wrap_model(&pb_graph(
                &[pb_node("Frobnicate", &["x"], &["y"], &[])],
                &[],
                std::slice::from_ref(&x),
                &[],
                std::slice::from_ref(&y),
            )),
        ),
        (
            "an attribute with no declared type",
            wrap_model(&pb_graph(
                &[pb_node(
                    "Relu",
                    &["x"],
                    &["y"],
                    &[pb_attribute_untyped("axis", &[])],
                )],
                &[],
                std::slice::from_ref(&x),
                &[],
                std::slice::from_ref(&y),
            )),
        ),
        (
            "an attribute declaring a subgraph",
            wrap_model(&pb_graph(
                &[pb_node(
                    "Relu",
                    &["x"],
                    &["y"],
                    &[pb_attribute("g", ATTR_GRAPH, &[])],
                )],
                &[],
                std::slice::from_ref(&x),
                &[],
                std::slice::from_ref(&y),
            )),
        ),
        (
            "an attribute declaring INT but carrying no value",
            wrap_model(&pb_graph(
                &[pb_node(
                    "Relu",
                    &["x"],
                    &["y"],
                    &[pb_attribute("k", ATTR_INT, &[])],
                )],
                &[],
                std::slice::from_ref(&x),
                &[],
                std::slice::from_ref(&y),
            )),
        ),
        (
            "an attribute string that is not UTF-8",
            wrap_model(&pb_graph(
                &[pb_node(
                    "Relu",
                    &["x"],
                    &["y"],
                    &[pb_attribute("s", ATTR_STRING, &pb_bytes_field(4, &[0xff]))],
                )],
                &[],
                std::slice::from_ref(&x),
                &[],
                std::slice::from_ref(&y),
            )),
        ),
        (
            "a graph input with no shape",
            wrap_model(&pb_graph(
                &[],
                &[],
                &[pb_value_info_unshaped("x", ONNX_FLOAT)],
                &[],
                &[],
            )),
        ),
        (
            "a graph input whose dim is symbolic",
            wrap_model(&pb_graph(
                &[],
                &[],
                &[pb_value_info_symbolic("x", ONNX_FLOAT, "N")],
                &[],
                &[],
            )),
        ),
        (
            "a graph input with a negative dimension",
            wrap_model(&pb_graph(
                &[],
                &[],
                &[pb_value_info("x", ONNX_FLOAT, &[-1, 3])],
                &[],
                &[],
            )),
        ),
        (
            "an initializer with a negative dimension",
            wrap_model(&pb_graph(
                &[],
                &[pb_tensor(Some("w"), &[-1], Some(ONNX_FLOAT), Some(&[]))],
                &[],
                &[],
                &[],
            )),
        ),
        (
            "an initializer with no raw_data",
            wrap_model(&pb_graph(
                &[],
                &[pb_tensor(Some("w"), &[2], Some(ONNX_FLOAT), None)],
                &[],
                &[],
                &[],
            )),
        ),
        (
            "an initializer with an unsupported element type",
            wrap_model(&pb_graph(
                &[],
                &[pb_tensor(Some("w"), &[2], Some(ONNX_INT32), Some(&[]))],
                &[],
                &[],
                &[],
            )),
        ),
        (
            "an initializer with no name",
            wrap_model(&pb_graph(
                &[],
                &[pb_tensor(None, &[2], Some(ONNX_FLOAT), Some(&[]))],
                &[],
                &[],
                &[],
            )),
        ),
        (
            "a graph output nothing ever produced",
            wrap_model(&pb_graph(
                &[pb_node("Relu", &["x"], &["y"], &[])],
                &[],
                std::slice::from_ref(&x),
                &[pb_value_info("z", ONNX_FLOAT, &[2, 3])],
                std::slice::from_ref(&y),
            )),
        ),
        (
            "an op_type that is not valid UTF-8",
            wrap_model(&pb_graph(
                &[{
                    // `op_type` is field 4, wire 2: [0x22, len, payload...].
                    // Patch the first payload byte of "Relu" so prost's
                    // string decoder rejects it mid-message.
                    let mut node = pb_node("Relu", &["x"], &["y"], &[]);
                    let payload_start = node
                        .windows(2)
                        .position(|window| window == [0x22, 0x04])
                        .expect("op_type field present");
                    node[payload_start + 2] = 0xff;
                    node
                }],
                &[],
                &[x],
                &[],
                &[y],
            )),
        ),
    ];
    cases
}

// --- deterministic refusals ----------------------------------------------

#[test]
fn an_empty_file_is_refused() {
    let case = Case::new();
    let error = case.import(&[]).expect_err("empty files carry no graph");
    assert!(error.contains("no graph"), "unexpected refusal: {error}");
}

/// Splits a well-formed `ModelProto` buffer into its top-level fields as
/// `(tag, start, end)` ranges. The exporter writes only wire types 0
/// (varint) and 2 (length-delimited), and so does every fixture derived
/// from it, which is the only reason this walker may assume those two.
fn model_fields(bytes: &[u8]) -> Vec<(u32, usize, usize)> {
    let mut fields = Vec::new();
    let mut pos = 0usize;
    let read_varint = |pos: &mut usize| -> u64 {
        let mut shift = 0u32;
        let mut value = 0u64;
        loop {
            assert!(*pos < bytes.len(), "truncated varint at {pos}");
            let byte = bytes[*pos];
            *pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return value;
            }
            shift += 7;
            assert!(shift < 64, "varint too long");
        }
    };
    while pos < bytes.len() {
        let start = pos;
        let key = read_varint(&mut pos);
        let tag = (key >> 3) as u32;
        match key & 7 {
            0 => {
                read_varint(&mut pos);
            }
            2 => {
                let len = read_varint(&mut pos) as usize;
                pos += len;
                assert!(pos <= bytes.len(), "truncated length-delimited field");
            }
            other => panic!("fixture model uses unexpected wire type {other}"),
        }
        fields.push((tag, start, pos));
    }
    fields
}

/// A valid export's `graph` (tag 7) is followed only by `opset_import`
/// (tag 8): prost encodes top-level fields in tag order, and the importer
/// never reads the opset section. So the precise fail-closed boundary is
/// the end of the graph field - every truncation that cuts into or short
/// of the graph must be refused, the graph-complete prefix (missing only
/// the trailing opset metadata) and the intact file may import, and a cut
/// inside the trailing field itself must be refused too.
#[test]
fn every_truncation_short_of_the_complete_graph_is_refused() {
    let case = Case::new();
    let bytes = case
        .export(&fixture_graph())
        .expect("fixture should export");
    assert!(
        bytes.len() > 64,
        "fixture is too small for truncation to be meaningful"
    );

    let fields = model_fields(&bytes);
    let tags: Vec<u32> = fields.iter().map(|&(tag, _, _)| tag).collect();
    assert_eq!(
        tags,
        vec![1, 2, 7, 8],
        "exporter writes ir_version, producer_name, graph, opset_import in tag order"
    );
    let graph_end = fields
        .iter()
        .find(|&&(tag, _, _)| tag == 7)
        .map(|&(_, _, end)| end)
        .expect("export carries a graph field");

    for length in 0..graph_end {
        assert!(
            case.import(&bytes[..length]).is_err(),
            "prefix of length {length} (short of graph end {graph_end}) must be refused"
        );
    }
    assert!(
        case.import(&bytes[..graph_end]).is_ok(),
        "the graph-complete prefix, missing only the trailing opset, must import"
    );
    for length in graph_end + 1..bytes.len() {
        assert!(
            case.import(&bytes[..length]).is_err(),
            "prefix of length {length} cuts into the trailing field and must be refused"
        );
    }
    assert!(case.import(&bytes).is_ok(), "the intact export must import");
}

/// Every single byte of a valid export is flipped. Most flips break the
/// structure and must produce `Err`; the rest land in payload bytes whose
/// mutated value is still well-formed (a producer-name letter, another
/// positive dimension, raw initializer data the reader does not
/// interpret). Those must produce a *coherent* graph - the property is
/// fail-closed-or-valid, never a panic and never a half-bound `Graph`.
#[test]
fn single_byte_corruption_of_a_valid_export_never_panics() {
    let case = Case::new();
    let bytes = case
        .export(&fixture_graph())
        .expect("fixture should export");
    let mut refused = 0usize;
    let mut accepted = 0usize;
    for index in 0..bytes.len() {
        let mut mutated = bytes.clone();
        mutated[index] = !mutated[index];
        match case.import(&mutated) {
            Ok(graph) => {
                assert_sane(&graph);
                accepted += 1;
            }
            Err(_) => refused += 1,
        }
    }
    // The fixture is structurally dense; a sweep in which most mutations
    // "succeed" would mean the reader stopped actually reading.
    assert!(
        refused > accepted,
        "only {refused} of {} byte flips were refused",
        bytes.len()
    );
}

/// Lengths no real file carries: a varint claiming more bytes than the
/// buffer holds must error out of `ModelProto::decode` before anything
/// tries to allocate or index the claimed span.
#[test]
fn huge_claimed_lengths_fail_closed() {
    let case = Case::new();
    let cases: &[(&str, &[u8])] = &[
        (
            "graph claims ~7e16 bytes",
            &[0x3a, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x00],
        ),
        (
            "graph claims 2^32 bytes",
            &[0x3a, 0x80, 0x80, 0x80, 0x80, 0x10],
        ),
        (
            "node inside graph claims more than the graph holds",
            &[
                0x3a, 0x0a, 0x0a, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x00, 0x00, 0x00,
            ],
        ),
        (
            "ir_version encoded as an oversized length-delimited field",
            &[0x0a, 0xff, 0xff, 0xff, 0xff, 0x0f],
        ),
    ];
    for (name, bytes) in cases {
        assert!(case.import(bytes).is_err(), "{name} must be refused");
    }
}

/// Attribute values nest graphs inside graphs inside graphs. `prost`
/// caps decode recursion at 100 frames; anything deeper must surface as
/// `Err`, not a stack overflow, and a shallow-but-opaque nest must fail
/// during reconstruction instead.
#[test]
fn deeply_nested_attribute_graphs_fail_closed() {
    let case = Case::new();

    let wrap_once = |inner: &[u8]| {
        let attribute = pb_bytes_field(6, inner); // AttributeProto.g
        let node = pb_bytes_field(5, &attribute); // NodeProto.attribute
        pb_bytes_field(1, &node) // GraphProto.node
    };

    let mut beyond_limit = Vec::new();
    for _ in 0..2_000 {
        beyond_limit = wrap_once(&beyond_limit);
    }
    assert!(
        case.import(&pb_bytes_field(7, &beyond_limit)).is_err(),
        "nesting past the decoder recursion limit must be refused"
    );

    let mut shallow = Vec::new();
    for _ in 0..50 {
        shallow = wrap_once(&shallow);
    }
    assert!(
        case.import(&pb_bytes_field(7, &shallow)).is_err(),
        "a decodable nest still fails reconstruction: its nodes carry no op_type"
    );
}

/// One `Err` per defect. Each graph is otherwise well-formed, so the
/// refusal proves the reader checked the named condition instead of
/// tripping over unrelated damage.
#[test]
fn structurally_broken_models_fail_closed() {
    let case = Case::new();
    for (name, bytes) in broken_models() {
        assert!(case.import(&bytes).is_err(), "{name} must be refused");
    }
}

/// The negative-initializer-dimension refusal is the fix for a specific
/// fail-open: `dim as usize` turned `-1` into `usize::MAX`, minting a
/// shape every downstream reader would inherit as fact.
#[test]
fn a_negative_initializer_dimension_names_its_refusal() {
    let case = Case::new();
    let bytes = wrap_model(&pb_graph(
        &[],
        &[pb_tensor(
            Some("w"),
            &[-1, 4],
            Some(ONNX_FLOAT),
            Some(&[0; 16]),
        )],
        &[],
        &[],
        &[],
    ));
    let error = case
        .import(&bytes)
        .expect_err("negative dims must be refused");
    assert!(
        error.contains("negative dimension"),
        "refusal should name the defect, got: {error}"
    );
}

// --- export-side refusals -------------------------------------------------

/// Before the writer looked values up instead of indexing, a `Graph`
/// whose `inputs` named an id `values` never held - reachable straight
/// through `Deserialize`, whose input fields are arbitrary - panicked
/// inside `export_to_onnx`.
#[test]
fn export_refuses_dangling_value_references() {
    let case = Case::new();

    let mut input_dangling = Graph::new();
    input_dangling.inputs.push(7);
    let error = case
        .export(&input_dangling)
        .expect_err("a dangling input must be refused");
    assert!(
        error.contains("graph input 7"),
        "unexpected refusal: {error}"
    );

    let mut initializer_dangling = Graph::new();
    initializer_dangling.initializers.insert(3, vec![0u8; 4]);
    let error = case
        .export(&initializer_dangling)
        .expect_err("a dangling initializer must be refused");
    assert!(
        error.contains("initializer 3"),
        "unexpected refusal: {error}"
    );

    let mut node_output_dangling = Graph::new();
    let x = node_output_dangling.add_value(vec![2], DTypeId::F32, None);
    node_output_dangling.mark_input(x);
    node_output_dangling.add_node(OperationKind::Relu, vec![x], vec![9], Default::default());
    let error = case
        .export(&node_output_dangling)
        .expect_err("a dangling node output must be refused");
    assert!(
        error.contains("node output 9"),
        "unexpected refusal: {error}"
    );

    let mut output_dangling = Graph::new();
    output_dangling.outputs.push(5);
    let error = case
        .export(&output_dangling)
        .expect_err("a dangling graph output must be refused");
    assert!(
        error.contains("graph output 5"),
        "unexpected refusal: {error}"
    );
}

/// `usize as i64` wraps above `i64::MAX`; the file would then claim a
/// negative dimension that reads back as a different, plausible extent.
#[test]
fn export_refuses_dimensions_outside_the_onnx_range() {
    let case = Case::new();
    let mut graph = Graph::new();
    let value = graph.add_value(vec![usize::MAX, 4], DTypeId::F32, None);
    graph.mark_input(value);
    let error = case
        .export(&graph)
        .expect_err("unrepresentable dims must be refused");
    assert!(
        error.contains("does not fit the ONNX int64 dimension type"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn export_refuses_operations_with_no_onnx_projection() {
    let case = Case::new();
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2], DTypeId::F32, None);
    let y = graph.add_value(vec![2], DTypeId::F32, None);
    graph.mark_input(x);
    // `Storage` is a real catalog op with deliberately no ONNX name.
    graph.add_node(OperationKind::Storage, vec![x], vec![y], Default::default());
    graph.mark_output(y);
    let error = case
        .export(&graph)
        .expect_err("unmappable op must be refused");
    assert!(
        error.contains("no ONNX mapping"),
        "unexpected refusal: {error}"
    );
}

// --- properties ----------------------------------------------------------

/// Degenerate-by-construction graphs: random dims (including `usize::MAX`),
/// every built-in dtype, ids that may not exist, ops with no ONNX
/// projection, attributes with `NaN` floats. Export may succeed or refuse;
/// it may not panic, and whatever it writes must be readable bytes.
fn degenerate_graph_strategy() -> impl Strategy<Value = Graph> {
    let value_strategy = (proptest::collection::vec(any::<usize>(), 0..=3), 0u8..=8);
    let id_strategy = any::<usize>();
    let attribute_strategy = (
        prop_oneof![Just("axis"), Just("k"), Just("perm"), Just("tag")],
        prop_oneof![
            any::<i64>().prop_map(AttributeValue::Int),
            any::<f32>().prop_map(AttributeValue::Float),
            proptest::collection::vec(any::<u8>(), 0..=8).prop_map(|bytes| {
                AttributeValue::String(String::from_utf8_lossy(&bytes).into_owned())
            }),
            proptest::collection::vec(any::<i64>(), 0..=4).prop_map(AttributeValue::Ints),
            proptest::collection::vec(any::<f32>(), 0..=4).prop_map(AttributeValue::Floats),
        ],
    );
    let node_strategy = (
        prop_oneof![
            Just(OperationKind::Add),
            Just(OperationKind::Relu),
            Just(OperationKind::Storage),
            Just(OperationKind::Fill),
            Just(OperationKind::UnsqueezeExact),
            Just(OperationKind::LogSoftmax),
            Just(OperationKind::MatMulExact),
            Just(OperationKind::CmpEq),
            Just(OperationKind::Triu),
            Just(OperationKind::PixelShuffle),
        ],
        proptest::collection::vec(any::<usize>(), 0..=2),
        proptest::collection::vec(any::<usize>(), 1..=2),
        proptest::collection::vec(attribute_strategy, 0..=3),
    );

    (
        proptest::collection::vec(value_strategy, 0..=4),
        proptest::collection::vec(id_strategy, 0..=3),
        proptest::collection::vec(id_strategy, 0..=3),
        proptest::collection::vec(
            (id_strategy, proptest::collection::vec(any::<u8>(), 0..=8)),
            0..=3,
        ),
        proptest::collection::vec(node_strategy, 0..=3),
    )
        .prop_map(|(values, inputs, outputs, initializers, nodes)| {
            let mut graph = Graph::new();
            let mut bound = Vec::new();
            for (shape, dtype_byte) in values {
                let dtype = match dtype_byte {
                    0 => DTypeId::F32,
                    1 => DTypeId::F64,
                    2 => DTypeId::F16,
                    3 => DTypeId::BF16,
                    4 => DTypeId::U8,
                    5 => DTypeId::U32,
                    6 => DTypeId::I64,
                    7 => DTypeId::Bool,
                    _ => DTypeId::Q8_0,
                };
                bound.push(graph.add_value(shape, dtype, None));
            }
            // Reduce arbitrary ids into "usually bound, sometimes not":
            // `+2` guarantees ids past the end stay reachable for any pool
            // size, including the empty one.
            let map_id = |raw: usize| -> usize {
                if bound.is_empty() {
                    raw % 2
                } else {
                    raw % (bound.len() + 2)
                }
            };
            for raw in inputs {
                graph.inputs.push(map_id(raw));
            }
            for raw in outputs {
                graph.outputs.push(map_id(raw));
            }
            for (raw, bytes) in initializers {
                graph.initializers.insert(map_id(raw), bytes);
            }
            for (operation, node_inputs, node_outputs, attributes) in nodes {
                let inputs = node_inputs.into_iter().map(map_id).collect();
                let outputs = node_outputs.into_iter().map(map_id).collect();
                let attributes = attributes
                    .into_iter()
                    .map(|(name, value)| (name.to_string(), value))
                    .collect();
                graph.add_node(operation, inputs, outputs, attributes);
            }
            graph
        })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        max_shrink_iters: 4096,
        ..ProptestConfig::default()
    })]

    /// Arbitrary bytes through the whole public parse path. The only
    /// accepted `Ok` is a genuinely coherent model - random bytes can in
    /// principle spell an empty-but-valid `GraphProto`, which the reader
    /// is correct to return - and whatever comes back must be fully bound.
    #[test]
    fn arbitrary_bytes_never_panic_on_import(
        bytes in proptest::collection::vec(any::<u8>(), 0..=1024),
    ) {
        let case = Case::new();
        if let Ok(graph) = case.import(&bytes) {
            assert_sane(&graph);
        }
    }

    /// Export is total over `Graph`: every wiring the type allows, from
    /// dangling ids to unmappable ops to wrapped-too-large dims, resolves
    /// to `Ok` or `Err` - never a panic on the caller's behalf.
    #[test]
    fn export_is_total_over_degenerate_graphs(graph in degenerate_graph_strategy()) {
        let case = Case::new();
        if let Ok(bytes) = case.export(&graph) {
            assert!(!bytes.is_empty(), "a successful export writes bytes");
        }
    }
}
