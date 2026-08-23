use incin::prelude::*;

fn main() {
    // Placement is the allocation target's job now, not a macro clause.
    let _ = tensor![1.0, 2.0; backend: DefaultBackend];
}
