// #![cfg_attr(not(feature = "std"), no_std)]
// #![cfg_attr(
//     feature = "nightly",
//     feature(generic_const_exprs),
//     allow(incomplete_features)
// )]

pub use kindle_core::*;

pub use kindle_macros::{module, forward};

pub mod macros {
    pub use kindle_macros::{impl_arg_into, s, idx};
}

pub mod prelude {
    pub use kindle_core::prelude::*;
    pub use kindle_macros::*;
}
