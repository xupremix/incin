use super::Nil;

/// The zero-dimensional shape marker used for scalar tensors.
pub type Scalar = Nil;

/// Folds per-axis static extents into a static element count.
///
/// `None` from any axis, or an overflowing product, makes the whole answer
/// `None`. The multiplication is checked because a wrapped element count
/// could undersize an allocation or a kernel constant.
pub const fn fold_static_numel(extents: &[Option<usize>]) -> Option<usize> {
    let mut product: usize = 1;
    let mut index = 0;
    while index < extents.len() {
        match extents[index] {
            Some(extent) => match product.checked_mul(extent) {
                Some(next) => product = next,
                None => return None,
            },
            None => return None,
        }
        index += 1;
    }
    Some(product)
}
