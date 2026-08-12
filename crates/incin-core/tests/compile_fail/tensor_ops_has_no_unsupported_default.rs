//! `EXE-009`: the same guarantee as
//! `module_ops_has_no_unsupported_default`, held against the largest
//! operation family.
//!
//! `TensorOps` carried forty-nine unsupported default bodies — more than the
//! other families combined. Every backend but the CPU one inherited the same
//! thirty-three-operation hole, and two of those backends only refused because
//! of the default: `TracingBackend` and `DispatchBackend` both delegate to a
//! real inner backend that implements the operation. A trait default cannot
//! tell a missing kernel apart from a wrapper that forgot to forward, so
//! neither gap was visible until the defaults came off.

use incin_core::backend_authoring::Backend;
use incin_core::tensor::backend::TensorOps;

struct Incomplete<B>(core::marker::PhantomData<B>);

impl<B: Backend> TensorOps<B> for Incomplete<B> {}

fn main() {}
