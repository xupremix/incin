//! Integration coverage for `main` on the documented public surface.
extern crate incin_core as incin;
use incin_macros::s;

/// Bad shape.
type BadShape = s![10, "foo"];

fn main() {}
