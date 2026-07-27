//! `best_device!()` resolves to the most capable enabled backend.
//!
//! Run under each feature set:
//!
//! ```text
//! cargo test -p incin-core --test auto_device
//! cargo test -p incin-core --test auto_device --features wgpu
//! cargo test -p incin-core --test auto_device --features cuda
//! cargo test -p incin-core --test auto_device --features cuda,wgpu
//! ```

use incin_core::best_device;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
use incin_core::prelude::Cpu;
use incin_core::prelude::{BestDevice, BestDeviceAt, ConstDevice, Device, DeviceKind};

/// The family this build should resolve to, stated independently of the
/// aliases under test so the test cannot agree with a wrong answer by
/// construction.
const EXPECTED: DeviceKind = if cfg!(feature = "cuda") {
    DeviceKind::Cuda
} else if cfg!(feature = "wgpu") {
    DeviceKind::Wgpu
} else {
    DeviceKind::Cpu
};

#[test]
fn best_device_resolves_to_the_most_capable_enabled_family() {
    let field = <BestDevice as Device>::init(());
    let id = <BestDevice as Device>::to_incin(&field).expect("logical selector resolves");
    assert_eq!(id.kind(), EXPECTED);
}

#[test]
fn cuda_outranks_wgpu_when_both_are_enabled() {
    // The ordering claim, not just "some GPU was picked".
    if cfg!(all(feature = "cuda", feature = "wgpu")) {
        let field = <BestDevice as Device>::init(());
        let id = <BestDevice as Device>::to_incin(&field).unwrap();
        assert_eq!(id.kind(), DeviceKind::Cuda, "wgpu outranked cuda");
    }
}

#[test]
fn the_macro_and_the_alias_name_the_same_type() {
    // If these ever diverge, the macro is lying about what it expands to.
    fn same<T>(_: T) {}
    let _: best_device!() = <BestDevice as Default>::default();
    same::<best_device!()>(BestDevice::default());
}

#[test]
fn an_ordinal_selects_that_device_on_a_gpu_build() {
    type Second = best_device!(typenum::U1);
    let field = <Second as Device>::init(());
    let id = <Second as Device>::to_incin(&field).expect("logical selector resolves");

    assert_eq!(id.kind(), EXPECTED);
    if EXPECTED == DeviceKind::Cpu {
        // The CPU has no ordinal, so it is accepted and ignored rather than
        // failing to compile -- code written for a GPU build still builds here.
        assert_eq!(id.ordinal(), 0);
    } else {
        assert_eq!(id.ordinal(), 1);
    }
}

#[test]
fn best_device_is_a_fully_static_selector() {
    // `ConstDevice` is the marker for "no constructor argument needed", which
    // is what makes `Tensor::zeros(())` work with this type.
    fn assert_const_device<D: ConstDevice>() {}
    assert_const_device::<BestDevice>();
    assert_const_device::<BestDeviceAt<typenum::U0>>();
}

// A type-level assertion cannot hide behind `cfg!`, which is an ordinary
// runtime bool -- both arms are still type-checked, so the CPU arm fails to
// compile on a GPU build. These need real `#[cfg]` attributes.

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
#[test]
fn a_cpu_only_build_resolves_to_cpu() {
    fn assert_is_cpu(_: Cpu) {}
    assert_is_cpu(BestDevice::default());
}

#[cfg(feature = "cuda")]
#[test]
fn a_cuda_build_resolves_to_cuda_at_the_requested_ordinal() {
    fn assert_is_cuda<N: typenum::Unsigned>(_: incin_core::prelude::CudaN<N>) {}
    assert_is_cuda(BestDevice::default());
    assert_is_cuda(<BestDeviceAt<typenum::U3>>::default());
}

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
#[test]
fn a_wgpu_only_build_resolves_to_wgpu() {
    fn assert_is_wgpu<N: typenum::Unsigned>(_: incin_core::prelude::WgpuN<N>) {}
    assert_is_wgpu(BestDevice::default());
    assert_is_wgpu(<BestDeviceAt<typenum::U3>>::default());
}
