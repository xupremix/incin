pub mod arithmetic;
pub mod broadcast;

pub mod dim;
pub mod named;
pub mod shape;
pub mod shape_ops;
pub mod spatial;
pub mod reshape;
pub mod concat;
pub mod stack;
pub mod idx;

pub use arithmetic::*;
pub use broadcast::BroadcastShape;
pub use dim::*;
pub use shape::*;
pub use shape_ops::*;
pub use spatial::*;
pub use reshape::*;
pub use idx::*;

pub mod prelude {
    pub use super::arithmetic::*;
    pub use super::broadcast::*;

    pub use super::dim::*;
    pub use super::named::*;
    pub use super::shape::*;
}
