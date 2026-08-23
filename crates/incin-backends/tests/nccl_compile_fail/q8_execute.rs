//! Integration coverage for `execute` on the documented public surface.
use incin_backends::dist::{NcclBuffer, NcclTransport};
use incin_core::prelude::Q8_0;

fn execute(transport: &mut NcclTransport, input: &NcclBuffer<Q8_0>) {
    let _ = transport.execute(input);
}

fn main() {}
