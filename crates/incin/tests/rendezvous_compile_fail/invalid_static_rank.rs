//! Integration coverage for `main` on the documented public surface.
use incin::experimental::distributed::{RunId, TwoRankDataParallel, TwoRankLaunchPlan};
use incin::typenum::U2;

fn main() {
    let plan = TwoRankLaunchPlan::<TwoRankDataParallel>::new_static(
        RunId::new("invalid-rank").unwrap(),
        "127.0.0.1:12345".parse().unwrap(),
        [0, 0],
        std::time::Duration::from_secs(1),
    )
    .unwrap();
    let _ = plan.rank_static::<U2>();
}
