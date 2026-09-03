//! The `K` parameter of `Tensor<S, B, K, G, P>`: choosing an element dtype,
//! converting between dtypes, and asking a backend what it can actually run.
//!
//! Most incin code leaves `K` at its `f32` default and never writes it. This
//! example writes it deliberately, because the interesting part of the dtype
//! axis is not that a tensor can hold `f16`, it is that *holding* a dtype and
//! *computing* in it are two different questions with two different answers.
//!
//! CPU today allocates all eight built-in dtypes but computes in far fewer:
//! `matmul` is `f32` only, and elementwise arithmetic is float only. Those are
//! declared facts, not accidents. `docs/capabilities.md` is generated from the
//! same tables this example queries, so the table, the refusal message, and
//! the kernel cannot disagree with each other.
//!
//! Run with: `cargo run -p incin --example dtypes --features cpu`

// These examples exist to spell out tensor types in full, so the type
// parameters are visible at the point of use rather than hidden behind an
// alias. That is exactly what `type_complexity` asks you to stop doing.
#![allow(clippy::type_complexity)]

use incin::prelude::*;
use incin_core::exec::{Capabilities, CapabilityQuery, LayoutClass, MathMode, SupportLevel};
use incin_core::shapes::OperationKind;
use incin_core::tensor::device::DeviceKind;
use incin_core::tensor::dtype::DTypeId;

type B = DefaultBackend;

fn main() -> incin::Result<()> {
    section("1. Every built-in dtype allocates");
    allocation()?;

    section("2. Converting along the K axis");
    conversion()?;

    section("3. Asking before executing");
    ask_first();

    section("4. What a refusal looks like");
    refusal()?;

    Ok(())
}

/// `K` is a type parameter, so the dtype is fixed when the tensor type is
/// named. There is no runtime `dtype=` argument to get wrong, and no way to
/// end up holding a tensor whose element type differs from the one the
/// surrounding code was compiled against.
fn allocation() -> incin::Result<()> {
    // Floats.
    let _f32: Tensor<s![2, 2], B, f32> = Tensor::zeros(())?;
    let _f64: Tensor<s![2, 2], B, f64> = Tensor::zeros(())?;
    let _f16: Tensor<s![2, 2], B, f16> = Tensor::zeros(())?;
    let _bf16: Tensor<s![2, 2], B, bf16> = Tensor::zeros(())?;

    // Integers, for indices and labels rather than arithmetic.
    let _i64: Tensor<s![2, 2], B, i64> = Tensor::zeros(())?;
    let _u32: Tensor<s![2, 2], B, u32> = Tensor::zeros(())?;
    let _u8: Tensor<s![2, 2], B, u8> = Tensor::zeros(())?;

    // `bool`, which masks are made of.
    let _bool: Tensor<s![2, 2], B, bool> = Tensor::zeros(())?;

    println!("  f32 f64 f16 bf16 i64 u32 u8 bool all allocated");

    // `K` can also be `Dyn`, for a dtype chosen at run time. The tag then
    // travels with the value instead of the type, exactly as `Dyn` does for
    // shapes, and with the same trade: the program accepts a dtype it was not
    // compiled against, and the compiler stops being able to check it.
    let runtime: Tensor<Dyn, B, Dyn> = Tensor::ones((vec![2, 2], DTypeId::F64.descriptor()))?;
    println!(
        "  and one chosen at run time: {}",
        runtime.dtype().key().name()
    );

    Ok(())
}

/// `to_dtype` moves a tensor along the `K` axis and leaves the other four
/// parameters alone: the shape, backend, gradient mode, and placement of the
/// result are the ones it started with. The target dtype is a turbofish, so a
/// conversion to a dtype the build does not have is a compile error rather
/// than a runtime string lookup.
fn conversion() -> incin::Result<()> {
    let source: Tensor<s![2, 2], B, f32> = Tensor::ones(())?;

    let half = source.to_dtype::<f16>()?;
    let brain = source.to_dtype::<bf16>()?;
    let wide = source.to_dtype::<f64>()?;
    let index = source.to_dtype::<i64>()?;

    // And back again. Narrowing and widening are both just conversions; the
    // precision that a round trip through `f16` loses is the caller's problem
    // to reason about, not something incin hides.
    let round_trip = half.to_dtype::<f32>()?;

    println!("  f32 -> f16, bf16, f64, i64 and back to f32");
    println!(
        "  shapes preserved: {:?} {:?} {:?} {:?} {:?}",
        half.dims(),
        brain.dims(),
        wide.dims(),
        index.dims(),
        round_trip.dims()
    );
    Ok(())
}

/// The capability registry answers "can this backend do this?" without
/// allocating a tensor or launching a kernel. This is the same query
/// `cargo incin doctor` runs, and the same table `docs/capabilities.md` is
/// generated from, so an answer here is the answer execution will give.
fn ask_first() {
    let probes = [
        (OperationKind::MatMul, DTypeId::F32),
        (OperationKind::MatMul, DTypeId::F16),
        (OperationKind::MatMul, DTypeId::F64),
        (OperationKind::Add, DTypeId::F32),
        (OperationKind::Add, DTypeId::BF16),
        (OperationKind::Add, DTypeId::I64),
    ];

    for (operation, dtype) in probes {
        let query = CapabilityQuery {
            operation: incin_core::exec::OperationIdentity::Builtin(operation),
            dtype: dtype.descriptor(),
            layout: LayoutClass::Contiguous,
            rank: 2,
            training: false,
            math_mode: MathMode::Precise,
        };
        let verdict = match incin_backends::capability::registry(DeviceKind::Cpu).support(&query) {
            SupportLevel::Native => "native".to_string(),
            SupportLevel::Composed => "composed".to_string(),
            SupportLevel::Fallback => "fallback".to_string(),
            SupportLevel::Unsupported(why) => format!("unsupported ({why})"),
        };
        println!("  {operation:?} in {}: {verdict}", dtype.name());
    }
}

/// A dtype the backend does not compute in is refused, not approximated and
/// not silently promoted. The error names the backend, the dtype, and the
/// operation, which is enough to find the row in `docs/capabilities.md` that
/// says so.
///
/// Note the fallible call. The `+` operator returns a `Tensor` rather than a
/// `Result` and panics on refusal, so code that wants to *handle* an
/// unsupported dtype takes the method form.
fn refusal() -> incin::Result<()> {
    let lhs: Tensor<s![2, 2], B, f16> = Tensor::ones(())?;
    let rhs: Tensor<s![2, 2], B, f16> = Tensor::ones(())?;

    match lhs.matmul(&rhs) {
        Ok(_) => println!("  f16 matmul succeeded, so this backend grew a kernel"),
        Err(error) => println!("  f16 matmul refused: {error}"),
    }

    // The same operation in the dtype the row does advertise.
    let lhs: Tensor<s![2, 2], B, f32> = Tensor::ones(())?;
    let product = lhs.matmul(&lhs)?;
    println!("  f32 matmul returned {:?}", product.dims());

    Ok(())
}

fn section(title: &str) {
    println!("\n{title}");
    println!("{}", "-".repeat(title.len()));
}
