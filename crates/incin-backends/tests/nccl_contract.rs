//! Compile-time half of the NCCL dtype contract.

#![cfg(feature = "distributed-nccl")]

#[test]
fn unsupported_static_nccl_dtypes_are_compile_errors() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/nccl_compile_fail/*.rs");
}
