use incin_backends::dist::{
    CollectiveBackend, GroupId, ReferenceBuffer, ReferenceTransport, StreamId,
};
use incin_core::exec::ReduceOp;
use incin_core::prelude::Q8_0;

fn rejected(transport: &ReferenceTransport, inputs: &[ReferenceBuffer<Q8_0>]) {
    // Block-quantized values have no elementwise reduction semantics. A static
    // Q8_0 collective is therefore absent rather than a runtime refusal.
    let _ = transport.all_reduce(
        GroupId::new(0, 2).unwrap(),
        inputs,
        ReduceOp::Sum,
        StreamId::default(),
    );
}

fn main() {}
