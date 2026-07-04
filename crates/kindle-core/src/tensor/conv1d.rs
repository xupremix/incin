use crate::prelude::*;

/// Trait to compute output shape of Conv1d operations.
pub trait Conv1dShape<Weight: Shape> {
    type Output: Shape;
}

/// Dynamic Conv1d Shape Resolution
impl Conv1dShape<Dyn> for Dyn {
    type Output = Dyn;
}

// Minimal macro for static shape resolution
#[macro_export]
macro_rules! impl_conv1d_shape {
    (
        In = [$n:ident, $c_in:ident, $l_in:ident],
        Weight = [$c_out:ident, $c_in_w:ident, $k:ident],
        Out = [$n_out:ident, $c_out_o:ident, $l_out:ident],
        Stride = $stride:literal,
        Padding = $padding:literal,
        Dilation = $dilation:literal
    ) => {
        impl $crate::tensor::Conv1dShape<s![$c_out, $c_in_w, $k]> for s![$n, $c_in, $l_in] {
            type Output = s![$n_out, $c_out_o, $l_out];
        }
    };
}
