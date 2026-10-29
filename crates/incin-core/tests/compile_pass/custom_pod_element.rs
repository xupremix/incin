//! A downstream POD newtype satisfies the open `TensorElement` bound (D-110,
//! issue #96): the proof obligation is the POD bounds, not membership in a
//! sealed list.
use incin_core::prelude::TensorElement;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PositElem([u8; 3]);

// SAFETY: transparent over `[u8; 3]`; every byte pattern is a valid value.
unsafe impl bytemuck::NoUninit for PositElem {}
// SAFETY: same layout argument; zeroed memory is a valid value.
unsafe impl bytemuck::Zeroable for PositElem {}

fn assert_element<T: TensorElement>() {}

fn main() {
    assert_element::<PositElem>();
    assert_element::<f32>();
}
