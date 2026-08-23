//! Integration coverage for `requires_warmup` on the documented public surface.
use incin_backends::tuning::service::{
    CoordinatedWarmupTuning, DisabledTuning, TuningService,
};

fn requires_warmup(_: TuningService<CoordinatedWarmupTuning>) {}

fn main() {
    requires_warmup(TuningService::<DisabledTuning>::disabled());
}
