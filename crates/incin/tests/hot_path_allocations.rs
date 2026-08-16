#![cfg(feature = "cpu")]

//! PRF-001: how many heap allocations one eager operation costs.
//!
//! Latency is what a benchmark measures and allocation count is what explains
//! it, so this pins the second directly. A fixed-rank operation should not be
//! paying for variable-length metadata structures, and before this row a
//! `[64, 8] + [64, 1]` add spent twenty-one allocations, of which exactly one
//! held the result.
//!
//! Two kinds of assertion appear below and they carry different weight. The
//! aligned and broadcast paths have separate ceilings because the broadcast
//! path validates and constructs its right-aligned output shape, while the
//! aligned path can reuse the existing shape. The absolute ceilings are
//! regression gates carrying the count measured on x86-64, and they are
//! ceilings rather than equalities only because kernel selection is a
//! per-target decision. Each prints what it actually counted, so a run that
//! comes in under its ceiling says so rather than passing quietly.
//!
//! Counting is per-thread, for two reasons. The test harness runs these in
//! parallel, and a process-wide counter would report whatever the other tests
//! happened to be allocating at the time. It also confines the measurement to
//! the calling thread, so what is counted is the metadata an operation builds
//! rather than anything a worker pool does; the operations are sized small
//! enough that the CPU backend does not split them anyway.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use incin::prelude::*;

type B = incin::DefaultBackend;

struct Counting;

thread_local! {
    /// Const-initialized so that reading it cannot itself allocate, which is a
    /// requirement rather than a preference inside a global allocator.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

/// Add one to this thread's counter, if this thread is counting.
///
/// `try_with` rather than `with`: a thread tearing down its locals can still
/// allocate, and panicking from inside the global allocator at that point
/// would abort the process.
fn record() {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|counter| counter.set(counter.get() + 1));
    }
}

// SAFETY: every method forwards to `System` unchanged. The counter is
// incremented before the forwarded call, and nothing here reads or writes the
// allocation itself.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Count the allocations one call to `body` performs on this thread.
///
/// `body` runs once uncounted first. That warm-up is not cosmetic: the CPU
/// backend's thread pool and its thread-local tape each allocate once on first
/// use, and counting those would measure lazy initialization rather than the
/// steady-state cost of an operation.
fn allocations_of<R>(mut body: impl FnMut() -> R) -> usize {
    drop(body());

    ALLOCATIONS.with(|counter| counter.set(0));
    COUNTING.with(|flag| flag.set(true));
    let result = body();
    COUNTING.with(|flag| flag.set(false));
    let counted = ALLOCATIONS.with(Cell::get);
    drop(result);

    counted
}

/// The count measured on x86-64 for a rank-2 binary elementwise operation.
///
/// One allocation holds the output data and one holds the `Arc` around it; the
/// tape takes two, for its input-id list and its boxed backward closure. The
/// rest are rank-2 dimension and stride vectors handed between the tensor
/// frontend, the backend's shape accessor, descriptor validation, and the
/// storage constructor.
const BINARY_ALLOCATIONS: usize = 27;

/// The same count for a rank-2 unary elementwise operation, which records one
/// input on the tape rather than two.
const UNARY_ALLOCATIONS: usize = 20;

#[test]
fn a_binary_elementwise_operation_stays_within_its_measured_allocation_count() {
    let lhs = Tensor::<Dyn, B>::ones(vec![64, 8]).unwrap();
    let rhs = Tensor::<Dyn, B>::ones(vec![64, 8]).unwrap();

    let counted = allocations_of(|| lhs.try_add(&rhs).unwrap());

    assert!(
        counted <= BINARY_ALLOCATIONS,
        "a rank-2 binary elementwise op allocated {counted} times, above the \
         recorded {BINARY_ALLOCATIONS}"
    );
    println!("binary elementwise allocations: {counted}");
}

#[test]
fn broadcasting_an_operand_stays_within_its_measured_ceiling() {
    let lhs = Tensor::<Dyn, B>::ones(vec![64, 8]).unwrap();
    let aligned = Tensor::<Dyn, B>::ones(vec![64, 8]).unwrap();
    let broadcast = Tensor::<Dyn, B>::ones(vec![64, 1]).unwrap();

    let aligned_count = allocations_of(|| lhs.try_add(&aligned).unwrap());
    let broadcast_count = allocations_of(|| lhs.try_add(&broadcast).unwrap());

    // Broadcasting has a small, bounded metadata cost for constructing the
    // right-aligned output shape. The backend iteration plan remains
    // rank-independent, so this stays within the measured ceiling.
    assert_eq!(
        aligned_count, 24,
        "aligned operation allocation count changed to {aligned_count}"
    );
    assert!(
        broadcast_count <= BINARY_ALLOCATIONS,
        "broadcast operation allocated {broadcast_count} times, above the \
         recorded {BINARY_ALLOCATIONS}"
    );
    println!("aligned {aligned_count}, broadcasting {broadcast_count}");
}

#[test]
fn a_unary_operation_stays_within_its_measured_allocation_count() {
    let input = Tensor::<Dyn, B>::ones(vec![64, 8]).unwrap();

    let counted = allocations_of(|| input.relu().unwrap());

    assert!(
        counted <= UNARY_ALLOCATIONS,
        "a rank-2 unary elementwise op allocated {counted} times, above the \
         recorded {UNARY_ALLOCATIONS}"
    );
    println!("unary elementwise allocations: {counted}");
}

#[test]
fn a_higher_rank_operand_does_not_cost_more_metadata_than_a_rank_two_one() {
    // The plan holds its dimensions and strides inline up to INLINE_RANK, which
    // is 8. Rank 6 is therefore well inside the inline capacity but far enough
    // from rank 2 to catch a spill: this fails if INLINE_RANK is ever lowered
    // under the ranks the typed frontend can express, which is the one way the
    // inline buffers could stop being free without any of them being removed.
    // It does not catch a plan that went back to a vector, because that cost a
    // fixed six allocations at every rank.
    let lhs = Tensor::<Dyn, B>::ones(vec![2, 2, 2, 2, 2, 2]).unwrap();
    let rhs = Tensor::<Dyn, B>::ones(vec![2, 2, 2, 2, 2, 1]).unwrap();
    let rank_two_lhs = Tensor::<Dyn, B>::ones(vec![8, 8]).unwrap();
    let rank_two_rhs = Tensor::<Dyn, B>::ones(vec![8, 1]).unwrap();

    let rank_six = allocations_of(|| lhs.try_add(&rhs).unwrap());
    let rank_two = allocations_of(|| rank_two_lhs.try_add(&rank_two_rhs).unwrap());

    assert_eq!(
        rank_six, rank_two,
        "a rank-6 broadcasting add cost {rank_six} allocations against \
         {rank_two} for the rank-2 one"
    );
    println!("rank 2: {rank_two}, rank 6: {rank_six}");
}
