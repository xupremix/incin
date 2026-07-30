use incin_macros::einsum;

#[test]
fn test_einsum_macro_basic() {
    let (subscript, a, b) = einsum!("ij,jk->ik"; 1, 2);
    assert_eq!(subscript, "ij,jk->ik");
    assert_eq!(a, 1);
    assert_eq!(b, 2);
}

#[test]
fn test_einsum_macro_single_operand() {
    let (subscript, x) = einsum!("ii->i"; 10);
    assert_eq!(subscript, "ii->i");
    assert_eq!(x, 10);
}
