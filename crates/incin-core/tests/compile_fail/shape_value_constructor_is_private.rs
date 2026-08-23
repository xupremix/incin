//! Integration coverage for `main` on the documented public surface.
extern crate incin_core as incin;

use incin::prelude::{ShapeBuf, ShapeValue};
use incin_macros::s;

fn main() {
    let _ = ShapeValue::<s![2, 3]>::from_validated(ShapeBuf::from_slice(&[9, 9]));
}
