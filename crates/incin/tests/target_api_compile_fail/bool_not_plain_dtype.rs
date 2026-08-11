use incin::prelude::*;

fn requires_plain<T: PlainDType>() {}

fn main() {
    requires_plain::<bool>();
}
