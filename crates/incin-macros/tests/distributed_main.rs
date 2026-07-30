//! `UX-007`: `#[distributed_main]` attribute macro test.

use incin_macros::distributed_main;

#[distributed_main]
fn entry_point() {
    println!("Distributed main entry point executing");
}

#[test]
fn test_distributed_main_execution() {
    entry_point();
}
