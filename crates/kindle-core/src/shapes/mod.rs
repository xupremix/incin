pub mod arithmetic;
pub mod const_dim;
pub mod dim;
pub mod hlist;
pub mod named;
pub mod shape;
pub mod shape_ops;
pub mod spatial;

pub use dim::*;
pub use shape::*;
pub use spatial::*;
pub use shape_ops::*;

pub mod prelude {
    pub use super::arithmetic::*;
    pub use super::const_dim::*;
    pub use super::dim::*;
    pub use super::hlist::*;
    pub use super::named::*;
    pub use super::shape::*;
}
