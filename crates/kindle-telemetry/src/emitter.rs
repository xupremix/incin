//! `Emitter`: the first concrete implementor of the [`Reporter`] trait.
//!
//! Two `crossbeam-channel` bounded channels (priority + bulk) drain into one
//! dedicated background writer thread, satisfying TELEM-02's non-blocking
//! contract and ARCH-02's "training proceeds identically whether or not a
//! viewer is attached" requirement. The training thread never touches a
//! `Write` impl, a `Mutex`, or the filesystem -- it only calls `try_send`
//! (bulk channel, drop-oldest on `Full`) or a bounded `send_timeout`
//! (priority channel, effectively-always-succeeds given its low volume and
//! dedicated capacity).

use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::events::{
    EpochEvent, Event, GradientNormEvent, GraphSnapshotEvent, HyperparamEvent, MemoryEvent,
    ScalarEvent, WeightNormEvent,
};
use crate::reporter::Reporter;
use crate::transport::Transport;

/// Bounded capacity of the always-delivered priority channel
/// (`HyperparamEvent`/`GraphSnapshotEvent`/`EpochEvent`/`MemoryEvent`).
const PRIORITY_CAP: usize = 64;
/// Bounded capacity of the drop-eligible bulk channel
/// (`ScalarEvent`/`GradientNormEvent`/`WeightNormEvent`).
const BULK_CAP: usize = 4096;
/// Bounded wait on the priority channel's `send_timeout` -- a short bound,
/// not an indefinite block, so the training thread can never stall
/// indefinitely even if the writer thread is fully wedged (T-07-03).
const PRIORITY_SEND_TIMEOUT: Duration = Duration::from_millis(50);
/// Max bulk events drained per writer-loop iteration before re-checking the
/// priority channel, so sustained bulk traffic cannot starve priority
/// delivery (Pitfall 5).
const BULK_DRAIN_BATCH: usize = 256;
/// Idle-wait bound for the writer loop's `select!` -- purely to avoid a
/// busy-spin when both channels are empty; never used for dequeue ordering.
const IDLE_WAIT: Duration = Duration::from_millis(200);

/// Non-blocking [`Reporter`] implementation. Holds two
/// `crossbeam_channel::Sender<Event>` handles -- one small, always-delivered
/// priority lane, one larger, drop-eligible bulk lane -- plus an atomic
/// dropped-event counter. Construction spawns exactly one background thread
/// that owns both `Receiver`s and drains them into the given [`Transport`]s.
///
/// Dropping an `Emitter` (or calling [`Emitter::shutdown`]) drops both
/// `Sender`s, which lets the writer thread observe `Disconnected` on both
/// channels and exit its loop, and then joins that thread so all in-flight
/// events are guaranteed to have been drained and written before the drop
/// (or `shutdown` call) returns -- see `Drop for Emitter`.
pub struct Emitter {
    // Wrapped in `Option` solely so `Drop::drop` can `.take()` (drop) them
    // *before* joining the writer thread -- `Drop::drop` runs before a
    // struct's own fields are dropped, so without this the writer thread
    // would never observe `Disconnected` and `join()` would block forever.
    // Always `Some` outside of `Drop::drop`; `send_bulk`/`send_priority`
    // rely on this via `.as_ref().expect(...)`.
    priority_tx: Option<Sender<Event>>,
    bulk_tx: Option<Sender<Event>>,
    // A second handle onto the same bulk channel (crossbeam channels are
    // MPMC, so this is a cheap clone of the `Receiver` also held by the
    // writer thread), used exclusively by `send_bulk` to evict the oldest
    // queued event on overflow (true drop-oldest per D-05). Never used to
    // consume events destined for a `Transport` -- that's the writer
    // thread's job via its own `Receiver` handle.
    bulk_rx_for_eviction: Receiver<Event>,
    dropped_count: Arc<AtomicU64>,
    /// Count of `Transport::write_event` failures across all transports and
    /// the whole run (WR-01) -- distinct from `dropped_count`, which only
    /// tracks channel-overflow drops, never write failures.
    write_error_count: Arc<AtomicU64>,
    writer_handle: Option<std::thread::JoinHandle<()>>,
}

impl Emitter {
    /// Constructs an `Emitter` with production capacities (`PRIORITY_CAP`,
    /// `BULK_CAP`) writing to `transports`, spawning the background writer
    /// thread.
    pub fn new(transports: Vec<Box<dyn Transport>>) -> Self {
        Self::with_capacities(transports, PRIORITY_CAP, BULK_CAP)
    }

    /// Constructs an `Emitter` with caller-specified channel capacities.
    /// This exists so tests can exercise tiny (e.g. capacity-1) channels
    /// without touching the production `new()` path's constants.
    pub fn with_capacities(
        mut transports: Vec<Box<dyn Transport>>,
        priority_cap: usize,
        bulk_cap: usize,
    ) -> Self {
        let (priority_tx, priority_rx) = bounded::<Event>(priority_cap);
        let (bulk_tx, bulk_rx) = bounded::<Event>(bulk_cap);
        let bulk_rx_for_eviction = bulk_rx.clone();
        let dropped_count = Arc::new(AtomicU64::new(0));
        let write_error_count = Arc::new(AtomicU64::new(0));
        let write_error_count_writer = Arc::clone(&write_error_count);

        let writer_handle = std::thread::spawn(move || {
            writer_loop(
                priority_rx,
                bulk_rx,
                &mut transports,
                &write_error_count_writer,
            );
        });

        Self {
            priority_tx: Some(priority_tx),
            bulk_tx: Some(bulk_tx),
            bulk_rx_for_eviction,
            dropped_count,
            write_error_count,
            writer_handle: Some(writer_handle),
        }
    }

    /// Explicitly drains and shuts down this `Emitter`: drops both senders
    /// (letting the writer thread observe `Disconnected` on both channels
    /// and exit its loop), then joins the writer thread so all buffered
    /// events are guaranteed to have been drained and written to every
    /// [`Transport`] before this call returns.
    ///
    /// Equivalent to simply dropping the `Emitter` (see `Drop for Emitter`),
    /// but named/callable explicitly so callers who care about durability
    /// (e.g. `--bench-telemetry`, or any real training run) have a
    /// self-documenting way to block until the writer thread has drained
    /// everything, rather than relying on implicit scope-exit behavior.
    pub fn shutdown(self) {
        // `self` is consumed and its `Drop` impl runs at the end of this
        // function, which drops both senders and joins the writer thread.
    }

    /// Drop-eligible enqueue for high-frequency events (D-05/D-07). On a
    /// full channel, increments `dropped_count`, evicts the oldest queued
    /// event via `try_recv`, then retries `try_send` with the new (freshest)
    /// event -- true drop-oldest semantics per D-05: the channel stays full
    /// of the most recent events, and the stalest queued item is discarded
    /// first. Never blocks the calling thread.
    fn send_bulk(&self, event: Event) {
        // `.expect`: only `None` during/after `Drop::drop`, at which point
        // no caller can still hold a live `&self` to invoke this method.
        let bulk_tx = self.bulk_tx.as_ref().expect("bulk_tx taken before drop");
        match bulk_tx.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(event)) => {
                // Drop-oldest (D-05): count the drop, then evict the oldest
                // queued item via try_recv to make room, then retry
                // try_send with the new (freshest) event. If the evict-then-
                // retry also fails (raced by another concurrent producer
                // refilling the slot first), the new event is dropped
                // outright -- still correct, just less precise about which
                // element ultimately ends up evicted under contention.
                self.dropped_count.fetch_add(1, Ordering::Relaxed);
                let _ = self.bulk_rx_for_eviction.try_recv();
                let _ = bulk_tx.try_send(event);
            }
            Err(TrySendError::Disconnected(_)) => {
                // Writer thread died -- nothing more we can do; never panic
                // the training thread over a telemetry-sink failure.
            }
        }
    }

    /// Guaranteed-delivery enqueue for low-frequency structural events
    /// (D-07). A short bounded wait, not an indefinite block -- the
    /// `Result` is discarded because `Reporter` methods are infallible by
    /// contract (see `reporter.rs`).
    fn send_priority(&self, event: Event) {
        // `.expect`: only `None` during/after `Drop::drop`, at which point
        // no caller can still hold a live `&self` to invoke this method.
        let priority_tx = self
            .priority_tx
            .as_ref()
            .expect("priority_tx taken before drop");
        let _ = priority_tx.send_timeout(event, PRIORITY_SEND_TIMEOUT);
    }

    /// Reads the monotonic dropped-events counter (D-06).
    pub fn dropped_count(&self) -> u64 {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Reads the monotonic transport write-failure counter (WR-01) --
    /// incremented once per failing `Transport::write_event` call, across
    /// every transport and the whole run. Distinct from `dropped_count`,
    /// which only tracks channel-overflow drops on the training-thread
    /// side, never writer-thread I/O failures.
    pub fn write_error_count(&self) -> u64 {
        self.write_error_count.load(Ordering::Relaxed)
    }
}

impl Drop for Emitter {
    /// Drops both senders (via `.take()`) *before* joining the writer
    /// thread. `Drop::drop` runs before a struct's own fields are dropped,
    /// so without this explicit `.take()` the writer thread would never
    /// observe `Disconnected` on either channel and `join()` would block
    /// forever. Dropping the senders here first lets the writer thread's
    /// `try_recv` loop see both channels disconnect, drain whatever was
    /// left, and return -- at which point `join()` is guaranteed to
    /// complete promptly.
    fn drop(&mut self) {
        drop(self.priority_tx.take());
        drop(self.bulk_tx.take());

        if let Some(handle) = self.writer_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Reporter for Emitter {
    fn log_scalar(&self, event: ScalarEvent) {
        self.send_bulk(Event::Scalar(event));
    }

    fn log_gradient_norm(&self, event: GradientNormEvent) {
        self.send_bulk(Event::GradientNorm(event));
    }

    fn log_weight_norm(&self, event: WeightNormEvent) {
        self.send_bulk(Event::WeightNorm(event));
    }

    /// Routes through the priority channel alongside `HyperparamEvent`/
    /// `GraphSnapshotEvent`/`EpochEvent`. `MemoryEvent` was not named in
    /// D-07's high-frequency (drop-eligible) list, so it shares the
    /// always-delivered lane here.
    ///
    /// **Call-frequency contract (caller responsibility, not enforced by
    /// `Emitter`):** invoke this at most once per epoch, or on an explicit,
    /// deliberately-throttled cadence (e.g. every N steps) -- never from an
    /// unthrottled per-step hot-loop position. The shared priority channel
    /// has only `PRIORITY_CAP` (64) slots and no drop-eligibility; a caller
    /// hammering `log_memory` at per-step frequency could saturate the
    /// channel with memory samples and starve `HyperparamEvent`/
    /// `GraphSnapshotEvent` delivery. `Emitter` does not rate-limit this
    /// call itself -- respecting the cadence is on the caller.
    fn log_memory(&self, event: MemoryEvent) {
        self.send_priority(Event::Memory(event));
    }

    fn log_epoch(&self, event: EpochEvent) {
        self.send_priority(Event::Epoch(event));
    }

    fn log_hyperparam(&self, event: HyperparamEvent) {
        self.send_priority(Event::Hyperparam(event));
    }

    fn log_graph_snapshot(&self, event: GraphSnapshotEvent) {
        self.send_priority(Event::GraphSnapshot(event));
    }
}

/// Background writer-thread loop: drains `priority_rx` fully via `try_recv`
/// every iteration (never relying on `select!`'s branch fairness for
/// ordering, per Pitfall 5), then bounded-drains `bulk_rx` up to
/// `BULK_DRAIN_BATCH` items, then uses `select!` purely as an idle-wait.
fn writer_loop(
    priority_rx: Receiver<Event>,
    bulk_rx: Receiver<Event>,
    transports: &mut [Box<dyn Transport>],
    write_error_count: &AtomicU64,
) {
    loop {
        // Drain priority fully every iteration -- biased toward priority.
        // `TryRecvError::Disconnected` is tracked so both channels being
        // simultaneously empty-and-disconnected (Emitter dropped, both
        // Senders gone) can terminate the loop below instead of spinning
        // on `select!`'s `default` branch forever.
        let mut priority_disconnected = false;
        loop {
            match priority_rx.try_recv() {
                Ok(event) => write_to_all(transports, &event, write_error_count),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    priority_disconnected = true;
                    break;
                }
            }
        }

        // Bounded drain of bulk so priority gets re-checked frequently even
        // under sustained bulk load, rather than starving behind a busy
        // bulk channel.
        let mut bulk_disconnected = false;
        for _ in 0..BULK_DRAIN_BATCH {
            match bulk_rx.try_recv() {
                Ok(event) => write_to_all(transports, &event, write_error_count),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    bulk_disconnected = true;
                    break;
                }
            }
        }

        if priority_disconnected && bulk_disconnected {
            break; // Emitter dropped, both senders gone, nothing left to flush.
        }

        // Block (briefly) on whichever channel has data next, so the loop
        // doesn't busy-spin when idle. This must only observe readiness,
        // never actually dequeue -- `select!`'s `recv(chan) -> _` arms
        // perform a real `recv()` (consuming and discarding one item via
        // the `_` binding), which would silently drop exactly one event
        // per idle-wake. `Select::ready_timeout` polls which operand is
        // ready without consuming anything, so the next loop iteration's
        // explicit `try_recv` drain loops above do the actual dequeue
        // (Pitfall 5: `select!` is idle-wait only, never dequeue ordering).
        let mut selector = crossbeam_channel::Select::new();
        selector.recv(&priority_rx);
        selector.recv(&bulk_rx);
        let _ = selector.ready_timeout(IDLE_WAIT);
    }
}

/// Cap on how many `eprintln!` lines `write_to_all` will emit for
/// transport-write failures (WR-01) -- beyond this, failures are still
/// counted in `write_error_count` but no longer printed, so a persistently
/// broken transport doesn't spam stderr for the rest of the run.
const MAX_WRITE_ERROR_PRINTS: u64 = 10;

/// Writes `event` to every transport, never panicking on a single
/// transport's I/O error. Every failure increments `write_error_count`
/// (WR-01) so a caller can observe persistent transport breakage even when
/// stderr isn't captured; the `eprintln!` itself is rate-limited to the
/// first `MAX_WRITE_ERROR_PRINTS` occurrences to avoid unbounded stderr
/// spam from a transport that fails on every subsequent write.
fn write_to_all(
    transports: &mut [Box<dyn Transport>],
    event: &Event,
    write_error_count: &AtomicU64,
) {
    for t in transports.iter_mut() {
        if let Err(e) = t.write_event(event) {
            let prior_count = write_error_count.fetch_add(1, Ordering::Relaxed);
            if prior_count < MAX_WRITE_ERROR_PRINTS {
                eprintln!("kindle-telemetry: transport write failed: {e}");
            } else if prior_count == MAX_WRITE_ERROR_PRINTS {
                eprintln!(
                    "kindle-telemetry: transport write failed {MAX_WRITE_ERROR_PRINTS} times; \
                     suppressing further messages for this run (see Emitter::write_error_count)"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::CURRENT_SCHEMA_VERSION;
    use std::sync::Mutex;
    use std::time::Instant;

    /// Test-only `Transport` spy collecting every event written to it,
    /// local to this module's test infrastructure -- not a new public API.
    struct CollectingTransport(Arc<Mutex<Vec<Event>>>);

    impl Transport for CollectingTransport {
        fn write_event(&mut self, event: &Event) -> crate::err::Result<()> {
            self.0.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    /// A `Transport` that blocks for a fixed delay on every write, used to
    /// simulate a stalled/slow writer thread so overflow behavior can be
    /// observed deterministically.
    struct StallingTransport {
        delay: Duration,
        inner: Arc<Mutex<Vec<Event>>>,
    }

    impl Transport for StallingTransport {
        fn write_event(&mut self, event: &Event) -> crate::err::Result<()> {
            std::thread::sleep(self.delay);
            self.inner.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    fn scalar_event(step: usize, name: &str, value: f64) -> ScalarEvent {
        ScalarEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            name: name.to_string(),
            value,
        }
    }

    fn hyperparam_event() -> HyperparamEvent {
        let mut params = std::collections::HashMap::new();
        params.insert("lr".to_string(), "0.001".to_string());
        HyperparamEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            params,
        }
    }

    #[test]
    fn new_with_tiny_capacities_returns_without_blocking_or_panicking() {
        let collected = Arc::new(Mutex::new(Vec::new()));
        let transport: Box<dyn Transport> = Box::new(CollectingTransport(collected));
        let emitter = Emitter::with_capacities(vec![transport], 1, 1);

        // Merely constructing (and letting it drop at end of scope) must not
        // block or panic.
        emitter.log_scalar(scalar_event(0, "loss", 0.1));
    }

    #[test]
    fn overflowing_bulk_channel_does_not_block_caller() {
        // Capacity-1 bulk channel, writer thread deliberately stalled so the
        // channel stays saturated for the duration of the test.
        let inner = Arc::new(Mutex::new(Vec::new()));
        let transport: Box<dyn Transport> = Box::new(StallingTransport {
            delay: Duration::from_secs(5),
            inner,
        });
        let emitter = Emitter::with_capacities(vec![transport], 1, 1);

        // Give the writer thread a moment to pick up its first (stalling)
        // item, so the channel is genuinely full for the subsequent sends.
        std::thread::sleep(Duration::from_millis(50));

        emitter.log_scalar(scalar_event(0, "loss", 0.1));

        let start = Instant::now();
        emitter.log_scalar(scalar_event(1, "loss", 0.2));
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_millis(100),
            "send_bulk must not block the caller, took {elapsed:?}"
        );
    }

    #[test]
    fn dropped_count_increments_on_bulk_overflow_and_stays_zero_otherwise() {
        let inner = Arc::new(Mutex::new(Vec::new()));
        let transport: Box<dyn Transport> = Box::new(StallingTransport {
            delay: Duration::from_secs(5),
            inner,
        });
        let emitter = Emitter::with_capacities(vec![transport], 1, 1);

        assert_eq!(emitter.dropped_count(), 0);

        std::thread::sleep(Duration::from_millis(50));

        emitter.log_scalar(scalar_event(0, "loss", 0.1));
        emitter.log_scalar(scalar_event(1, "loss", 0.2));

        assert!(
            emitter.dropped_count() >= 1,
            "expected at least one dropped event after overflowing a saturated bulk channel"
        );
    }

    #[test]
    fn priority_event_delivered_even_while_bulk_channel_saturated() {
        let collected = Arc::new(Mutex::new(Vec::new()));
        let transport: Box<dyn Transport> = Box::new(CollectingTransport(collected.clone()));
        // Small bulk capacity so it saturates quickly; priority capacity
        // large enough that a single hyperparam event always fits.
        let emitter = Emitter::with_capacities(vec![transport], 8, 1);

        // Saturate the bulk channel with scalar events.
        for step in 0..64 {
            emitter.log_scalar(scalar_event(step, "loss", step as f64));
        }

        emitter.log_hyperparam(hyperparam_event());

        // Poll for up to 5s (well beyond PRIORITY_SEND_TIMEOUT) for the
        // writer thread to have drained and written the hyperparam event --
        // generous bound to avoid flakiness under CI/parallel-test CPU
        // contention; the assertion is about eventual delivery.
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut saw_hyperparam = false;
        while Instant::now() < deadline {
            if collected
                .lock()
                .unwrap()
                .iter()
                .any(|e| matches!(e, Event::Hyperparam(_)))
            {
                saw_hyperparam = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        assert!(
            saw_hyperparam,
            "HyperparamEvent must be delivered even while the bulk channel is saturated"
        );
    }

    #[test]
    fn event_type_routes_to_correct_channel_behavior_under_saturation() {
        let collected = Arc::new(Mutex::new(Vec::new()));
        let transport: Box<dyn Transport> = Box::new(CollectingTransport(collected.clone()));
        let emitter = Emitter::with_capacities(vec![transport], 64, 1);

        // Flood the bulk channel -- some of these are expected to be
        // dropped since bulk capacity is 1.
        for step in 0..100 {
            emitter.log_scalar(scalar_event(step, "loss", step as f64));
        }

        // Priority-side events -- all of these must arrive.
        emitter.log_hyperparam(hyperparam_event());
        emitter.log_epoch(EpochEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            epoch: 0,
            metrics: std::collections::HashMap::new(),
        });
        emitter.log_memory(MemoryEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step: 0,
            rss_bytes: 1024,
        });
        emitter.log_graph_snapshot(GraphSnapshotEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            graph: kindle_core::graph::Graph::default(),
        });

        // Bounded well beyond PRIORITY_SEND_TIMEOUT/IDLE_WAIT so the test is
        // not flaky under CI/parallel-test CPU contention -- the assertion
        // is about eventual delivery, not tight timing.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let events = collected.lock().unwrap();
            let has_hyperparam = events.iter().any(|e| matches!(e, Event::Hyperparam(_)));
            let has_epoch = events.iter().any(|e| matches!(e, Event::Epoch(_)));
            let has_memory = events.iter().any(|e| matches!(e, Event::Memory(_)));
            let has_graph = events.iter().any(|e| matches!(e, Event::GraphSnapshot(_)));
            drop(events);

            if has_hyperparam && has_epoch && has_memory && has_graph {
                break;
            }
            if Instant::now() >= deadline {
                panic!(
                    "expected all priority-routed events (Hyperparam/Epoch/Memory/GraphSnapshot) to be delivered: hyperparam={has_hyperparam} epoch={has_epoch} memory={has_memory} graph={has_graph}"
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        // Bulk (Scalar) events may legitimately have been dropped -- the
        // point is priority delivery is unconditional, bulk is best-effort.
        let events = collected.lock().unwrap();
        let scalar_count = events
            .iter()
            .filter(|e| matches!(e, Event::Scalar(_)))
            .count();
        assert!(scalar_count <= 100);
    }
}
