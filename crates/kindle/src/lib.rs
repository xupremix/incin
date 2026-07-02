// #![cfg_attr(not(feature = "std"), no_std)]
// #![cfg_attr(
//     feature = "nightly",
//     feature(generic_const_exprs),
//     allow(incomplete_features)
// )]

pub use kindle_core::*;

pub mod macros {
    pub use kindle_macros::s;
}

pub mod prelude {
    pub use kindle_core::prelude::*;
    pub use kindle_macros::*;
}
