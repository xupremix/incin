use incin::{Dyn, Error, Result, Shape};
use core::marker::PhantomData;

pub fn stable_types<S: Shape>(value: Result<S>) -> Result<S> {
    let _runtime_marker: PhantomData<Dyn> = PhantomData;
    value.map_err(|error: Error| error)
}
