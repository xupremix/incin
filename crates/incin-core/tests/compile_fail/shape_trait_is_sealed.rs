use incin_core::prelude::{Shape, ShapeBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Forged;

impl Shape for Forged {
    type Arg = ShapeBuf;

    fn resolve(_: Self::Arg) -> Result<ShapeBuf, incin_core::prelude::ShapeError> {
        Ok(ShapeBuf::from_slice(&[]))
    }

    fn validate_dims(_: &[usize]) -> Result<(), incin_core::prelude::ShapeError> {
        Ok(())
    }
}

fn main() {}
