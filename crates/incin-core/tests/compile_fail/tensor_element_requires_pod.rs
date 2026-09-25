use incin_core::prelude::TensorElement;

// Copy + Debug, but no `bytemuck` proof obligations: the open bound still
// refuses types that are not plain old data (D-110 kept every bound except
// the seal).
#[derive(Clone, Copy, Debug)]
struct NotPod(u8, bool);

fn assert_element<T: TensorElement>() {}

fn main() {
    assert_element::<NotPod>();
}
