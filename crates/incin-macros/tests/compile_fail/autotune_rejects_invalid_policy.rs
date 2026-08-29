use incin_macros::autotune;

#[autotune(
    key = "valid_key",
    params = [(16, 16)],
    policy = invalid_policy_name
)]
fn sample_func() {}

fn main() {}
