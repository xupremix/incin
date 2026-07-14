// diagnostic_test.rs
// Run with: rustc --edition=2021 diagnostic_test.rs

pub mod my_typenum {
    // A hypothetical simplified typenum that uses direct struct names
    pub struct _2;
    pub struct _3;
    pub struct _5;

    pub trait Add<Rhs> { type Output; }
    impl Add<_3> for _2 { type Output = _5; }
}

pub mod nested_typenum {
    // Simulating what typenum actually does under the hood
    pub struct UTerm;
    pub struct UInt<U, B> { _marker: core::marker::PhantomData<(U, B)> }
    pub struct B0;
    pub struct B1;

    pub type U2 = UInt<UInt<UTerm, B1>, B0>;
    pub type U3 = UInt<UInt<UTerm, B1>, B1>;
    pub type U5 = UInt<UInt<UInt<UTerm, B1>, B0>, B1>;

    pub trait Add<Rhs> { type Output; }
    impl Add<U3> for U2 { type Output = U5; }
}

#[diagnostic::on_unimplemented(
    message = "Custom diagnostic: Cannot combine shape `{Self}` with `{Other}`",
    label = "Shape combination failed",
    note = "This is a test of how rustc renders the types in the {Self} and {Other} placeholders"
)]
pub trait Combine<Other> {}

fn require_combine<A, B>() where A: Combine<B> {}

fn main() {
    // 1. Using our custom flat structs
    require_combine::<my_typenum::_5, my_typenum::_3>();

    // 2. Using our custom flat structs via a projection
    require_combine::<<my_typenum::_2 as my_typenum::Add<my_typenum::_3>>::Output, my_typenum::_3>();

    // 3. Using nested typenum structs (direct alias)
    require_combine::<nested_typenum::U5, nested_typenum::U3>();

    // 4. Using nested typenum structs (via projection)
    require_combine::<<nested_typenum::U2 as nested_typenum::Add<nested_typenum::U3>>::Output, nested_typenum::U3>();
}
