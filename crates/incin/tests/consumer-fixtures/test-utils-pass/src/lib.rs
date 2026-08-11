use incin::Cpu;
use incin::test_utils::DummyBackend;

pub fn test_backend_type() -> core::marker::PhantomData<DummyBackend<Cpu>> {
    core::marker::PhantomData
}
