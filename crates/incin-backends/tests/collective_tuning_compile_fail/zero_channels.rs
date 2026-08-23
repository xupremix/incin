use incin_backends::dist::{CollectiveTuningCandidate, Ring, Simple};
use incin_core::typenum::{U0, U16};

fn zero_channels() {
    let _ = CollectiveTuningCandidate::new_static::<Ring, Simple, U0, U16>(
        0, 0, 0, true,
    );
}

fn main() {}
