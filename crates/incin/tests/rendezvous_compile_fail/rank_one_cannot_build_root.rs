use incin::experimental::distributed::{RunId, StaticRendezvousConfig};
use incin::typenum::U1;

fn main() {
    let _ = StaticRendezvousConfig::<U1>::root(
        RunId::new("wrong-role").unwrap(),
        "127.0.0.1:12345".parse().unwrap(),
        0,
        std::time::Duration::from_secs(1),
    );
}
