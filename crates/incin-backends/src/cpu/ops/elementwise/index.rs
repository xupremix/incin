/// Increment a row-major multi-index in place (odometer-style), matching
/// `storage.rs`/`tape.rs`'s own iteration order.
pub(crate) fn increment_index(idx: &mut [usize], shape: &[usize]) {
    for i in (0..idx.len()).rev() {
        idx[i] += 1;
        if idx[i] < shape[i] {
            return;
        }
        idx[i] = 0;
    }
}

pub(crate) fn flat_to_nd(mut flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut nd = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        nd[i] = flat_idx % shape[i];
        flat_idx /= shape[i];
    }
    nd
}
