//! Runtime detection of the best device this machine can actually run on.
//!
//! This is the half of automatic device selection that inspects real hardware.
//! It runs on the machine that executes the binary, which is the only machine
//! whose hardware is relevant — see `incin_core::tensor::auto_device` for why
//! the compile-time half deliberately does not probe.
//!
//! The order is CUDA → WGPU → CPU: native GPU first, then the portable GPU
//! backend, then the always-available fallback. A family that this build did
//! not enable is skipped without being probed, so `detect_device` never links
//! against a driver the build excluded.

use incin_core::tensor::device::{DeviceId, DeviceKind};

/// The best device available on this machine right now.
///
/// Tries each enabled backend family in order and returns the first that
/// reports usable hardware. [`DeviceId::cpu`] is the terminal fallback, so this
/// function is total — it always returns a device.
///
/// Detection is performed on every call and nothing is cached. A caller that
/// selects a device once per process should hold on to the result; a caller
/// that wants to react to hardware appearing or disappearing can call it again.
///
/// ```rust
/// # extern crate incin_backends as incin;
/// use incin_backends::prelude::*;
/// use incin_core::tensor::prelude::*;
///
/// let device = incin_backends::detect_device();
/// let t = Tensor::<Dyn, IncinBackend<Dyn>>::zeros(([2, 3], device));
/// ```
#[must_use]
pub fn detect_device() -> DeviceId {
    detect_device_in(PREFERENCE)
}

/// The families [`detect_device`] tries, most capable first.
pub const PREFERENCE: &[DeviceKind] = &[
    DeviceKind::Cuda,
    DeviceKind::Metal,
    DeviceKind::Wgpu,
    DeviceKind::Cpu,
];

/// [`detect_device`] over a caller-chosen preference order.
///
/// Useful for pinning a policy — `detect_device_in(&[DeviceKind::Wgpu,
/// DeviceKind::Cpu])` refuses CUDA even where it is present. If no listed
/// family is available the CPU is still returned, because returning "no device"
/// would leave the caller with nothing runnable.
#[must_use]
pub fn detect_device_in(preference: &[DeviceKind]) -> DeviceId {
    for kind in preference {
        if let Some(device) = probe(*kind) {
            return device;
        }
    }
    DeviceId::cpu()
}

/// Whether a specific family is usable, and at which ordinal.
///
/// Returns `None` both when the family is not compiled in and when it is
/// compiled in but no hardware answers. Those are different reasons and a
/// caller that needs to tell them apart should ask
/// [`is_compiled_in`] as well.
#[must_use]
pub fn probe(kind: DeviceKind) -> Option<DeviceId> {
    match kind {
        DeviceKind::Cpu => cfg!(feature = "cpu").then(DeviceId::cpu),
        DeviceKind::Cuda => probe_cuda(),
        DeviceKind::Wgpu => probe_wgpu(),
        DeviceKind::Metal => probe_metal(),
        _ => None,
    }
}

/// Whether this build contains the given backend family at all.
///
/// Independent of whether hardware is present: `is_compiled_in(Cuda)` can be
/// `true` on a machine with no GPU.
#[must_use]
// With every backend feature enabled Clippy can fold this to `matches!`, but
// each arm intentionally retains its own feature gate for powerset builds.
#[allow(clippy::match_like_matches_macro)]
pub const fn is_compiled_in(kind: DeviceKind) -> bool {
    match kind {
        DeviceKind::Cpu => cfg!(feature = "cpu"),
        DeviceKind::Cuda => cfg!(feature = "cuda"),
        DeviceKind::Wgpu => cfg!(feature = "wgpu"),
        DeviceKind::Metal => cfg!(feature = "metal"),
        _ => false,
    }
}

#[cfg(feature = "metal")]
fn probe_metal() -> Option<DeviceId> {
    // Probes for Metal availability on Apple Silicon / macOS.
    if crate::metal::is_unified_memory() {
        Some(DeviceId::metal(0))
    } else {
        None
    }
}

#[cfg(not(feature = "metal"))]
fn probe_metal() -> Option<DeviceId> {
    None
}

#[cfg(feature = "cuda")]
fn probe_cuda() -> Option<DeviceId> {
    // Constructing a context is the actual availability question: a driver can
    // be installed, and `nvidia-smi` can list a device, while the context still
    // fails to initialize (wrong driver version, exhausted device, no
    // permission inside a container). Only the thing we are about to do proves
    // it can be done.
    //
    // Through the cache, so the probe both answers the question and leaves the
    // context retained. Creating one here and dropping it would release the
    // primary context, making the very next allocation pay the 131 ms
    // re-initialization that detection just proved was possible.
    crate::cuda::gpu::cuda_cache::try_get_cuda_device(0)
        .ok()
        .map(|_| DeviceId::cuda(0))
}

#[cfg(not(feature = "cuda"))]
fn probe_cuda() -> Option<DeviceId> {
    None
}

/// The one `wgpu::Instance` this crate's probe uses.
///
/// Shared rather than created per call because creating and dropping instances
/// concurrently takes the process down: two threads each probing twice is a
/// reproducible `SIGSEGV` inside adapter enumeration on a Mesa/llvmpipe stack,
/// found by `UX-014`'s `cargo incin doctor` — which was the first caller to
/// probe the same family more than once in a process.
///
/// This does not cache *detection*, which the module contract says is
/// performed on every call: [`wgpu::Instance::request_adapter`] still runs per
/// probe, so hardware appearing or disappearing is still observed. Only the
/// instance is long-lived, which is what `wgpu::device::get_device_state`
/// already does with the instance the backend runs on.
#[cfg(feature = "wgpu")]
static WGPU_INSTANCE: std::sync::OnceLock<wgpu::Instance> = std::sync::OnceLock::new();

#[cfg(feature = "wgpu")]
fn probe_wgpu() -> Option<DeviceId> {
    let instance = WGPU_INSTANCE.get_or_init(|| {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        })
    });
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        // A software adapter would be reported as a GPU while performing
        // worse than the CPU backend, so it does not count as available.
        force_fallback_adapter: false,
    }))
    .map(|_| DeviceId::wgpu(0))
}

#[cfg(not(feature = "wgpu"))]
fn probe_wgpu() -> Option<DeviceId> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_always_yields_a_runnable_device() {
        // Total by construction: the CPU terminates the chain.
        let device = detect_device();
        assert!(PREFERENCE.contains(&device.kind()), "{device:?}");
    }

    #[test]
    fn an_empty_preference_still_yields_the_cpu() {
        assert_eq!(detect_device_in(&[]).kind(), DeviceKind::Cpu);
    }

    #[test]
    fn preference_order_is_honored() {
        // Restricting to CPU must return the CPU even where a GPU exists,
        // which is what makes this usable as a policy override.
        assert_eq!(detect_device_in(&[DeviceKind::Cpu]).kind(), DeviceKind::Cpu);
    }

    #[test]
    fn a_family_that_is_not_compiled_in_is_never_detected() {
        for kind in [DeviceKind::Cuda, DeviceKind::Wgpu, DeviceKind::Cpu] {
            if !is_compiled_in(kind) {
                assert!(
                    probe(kind).is_none(),
                    "{kind:?} probed while not compiled in"
                );
            }
        }
    }

    /// Probing the same family repeatedly from more than one thread must not
    /// take the process down.
    ///
    /// This is a regression test for a real `SIGSEGV`, not a hypothetical.
    /// `probe_wgpu` used to build a `wgpu::Instance` per call and drop it; two
    /// threads each probing twice crashed inside adapter enumeration, three
    /// times out of three. `UX-014`'s `cargo incin doctor` found it by being
    /// the first caller to probe a family twice in one process.
    ///
    /// Deliberately not `wgpu`-specific: `probe` is documented as callable
    /// repeatedly, and that has to hold for whatever families the build has.
    /// On a CPU-only build this passes trivially, which is the correct amount
    /// of work for a build with no driver to crash in.
    #[test]
    fn probing_repeatedly_from_several_threads_is_survivable() {
        let threads: std::vec::Vec<_> = (0..2)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..2 {
                        for kind in [DeviceKind::Cpu, DeviceKind::Cuda, DeviceKind::Wgpu] {
                            let _ = probe(kind);
                        }
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("a probing thread must not panic");
        }
    }

    #[test]
    fn detection_agrees_with_itself() {
        // Nothing is cached, so two calls exercise the probe twice. They must
        // still agree, or a caller could get a different device per call.
        assert_eq!(detect_device(), detect_device());
    }
}
