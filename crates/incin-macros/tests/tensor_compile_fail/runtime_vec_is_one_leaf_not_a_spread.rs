use incin::prelude::*;

fn main() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let _ = tensor![data]; // shape must come from macro syntax, not a runtime Vec's length
}
