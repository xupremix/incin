use incin::Cpu;
use incin::test_utils::DummyBackend;

pub fn test_backend_type() -> core::marker::PhantomData<DummyBackend<f32, Cpu>> {
    core::marker::PhantomData
}
