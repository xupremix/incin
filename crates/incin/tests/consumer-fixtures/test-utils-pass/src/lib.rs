// `test-utils` gates deterministic fault injection against the real CPU
// backend. It no longer carries a stand-in backend: a consumer that needs a
// backend in a test uses a real one.
use incin::test_utils::{AssignFailureGuard, fail_assign_on};

pub fn fault_injection_is_reachable() -> AssignFailureGuard {
    fail_assign_on(1)
}
