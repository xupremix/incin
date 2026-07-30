use incin_backends::tuning::signature::{KernelSignature, RankClass};

fn accepts_signature(_: KernelSignature) {}

fn main() {
    accepts_signature(RankClass::Vector);
}
