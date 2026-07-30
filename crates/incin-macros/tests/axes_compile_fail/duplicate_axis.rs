use incin_macros::axes;

struct Batch;
struct Channels;

fn main() {
    type Invalid = axes![Batch, Channels, Batch];
}
