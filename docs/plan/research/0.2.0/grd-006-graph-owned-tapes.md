# GRD-006 remainder — graph-owned tapes

Finding: the unification half is done and recorded (ledger 2026-07-30: all
three backends use `TapeNode<S>`, the core walk, `TapeStorage`, one global
`TensorId`). What remains is what the code comments still name GRD-006 in
the future tense: four thread-local tapes (`cpu/tape.rs`, `cuda/tape.rs`,
`wgpu/tape.rs`, `metal/tape.rs`) own the graphs, and saved-tensor lifetime
is whatever the thread-local holds. Consequences, all structural: one graph
per thread per backend (two interleaved training loops share and can pollute
one tape; `drain_reachable` mitigates, the structure invites); graphs cannot
cross threads, be returned as values, inspected, or handed to another
backend; activations live until the next `backward` drains them, so there is
no handle for early drop, offload, or recompute; every backend reimplements
ownership (TLS + push + drain + walk) for one conceptual feature, and the
recently added `tape_record` shims are visibility patches over exactly this
wart. The recipe-arity zip-truncation is parked here too (ledger: "belongs
with GRD-006, which is already changing what a node holds") — the walk now
refuses mismatches, but the node still carries an unchecked `Vec`.

Recommendation: make the graph an explicit value in three stages. (1) Carry
a graph handle in `ExecutionContext`; `RecordingBackend::record_custom` and
the `tape_record` shims resolve it instead of touching a static, keeping
their signatures. (2) Move saved-storage ownership into the graph
(co-owned handles; dropping the graph releases saved tensors
deterministically), delete the four statics, and reshape `TapeNode` to a
checked gradient structure, consuming the parked arity item. (3) Keep
drain-by-value semantics re-expressed per graph (D-06 survives: a second
`backward` on a drained graph still returns only the seed), with one now
meaningful addition — a retained graph both backward calls can walk, which
is currently unrepresentable. Non-goals: second-order rules, in-place
mutation, cross-backend execution itself (this only makes the graph a value
those can be built on).

Risk: the drain changeover touches every backend's backward entry at once;
saved storages need `Send` bounds the closures already carry, but audit it;
the `record` shims deprecate rather than vanish in one step or external
crates break mid-migration; test-thread isolation assumptions change shape
(one tape per thread becomes one graph per handle — tests that relied on
ambient separation need explicit graphs). Needs no hardware: all of it is
provable on CPU plus the WGPU adapter.

Unblocks: retained graphs and second `backward`, cross-backend `backward`,
activation checkpointing (a recompute boundary needs a graph value to hold
the boundary), GRD-007 liveness (cannot analyze saved tensors it cannot
see), foreign backends without the TLS tax.
