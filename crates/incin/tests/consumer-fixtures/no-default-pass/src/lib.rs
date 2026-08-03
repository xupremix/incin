use incin::{Dyn, Error, Result, Shape};

pub fn stable_types<S: Shape>(value: Result<S>) -> Result<S> {
    let _runtime_marker = Dyn(());
    value.map_err(|error: Error| error)
}
