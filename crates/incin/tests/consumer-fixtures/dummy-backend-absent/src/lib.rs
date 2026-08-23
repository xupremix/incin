//! Integration coverage for `leaked` on the documented public surface.
// The shape-only stand-in backend is retired. Enabling `test-utils` - the
// feature that used to carry it - must not bring it back, otherwise a
// downstream test could still assert against a backend that computes nothing.
use incin::test_utils::DummyBackend;

pub fn leaked(_: DummyBackend<incin::Cpu>) {}
