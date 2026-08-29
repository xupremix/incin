use incin_macros::autotune;

#[autotune(
    key = "",
    params = [(16, 16)],
    policy = heuristic
)]
fn sample_func() {}

fn main() {}
