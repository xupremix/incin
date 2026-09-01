compiled graphs: everything but fusion, and CMP-005 is missing from the ledger

Finding: `crates/incin-core/src/compiled/` is a complete capture-plan-execute
pipeline with one hole, and the hole is documented and deliberate.

What works: `capture.rs` records a `CapturedGraph`; `plan.rs` (32 KB) compiles it
to an immutable `CompiledPlan` under `CompileOptions` with `ShapeGuard`
verification, a `DynamicShapePolicy` and a `FusionPolicy`; `alloc.rs` computes
liveness intervals and a `MemoryPlan` with buffer slot reuse; `fold.rs` does
constant folding and weight prepacking; `artifact.rs` defines a versioned
`CompiledArtifact` with a magic number; `manifest.rs` records a
`ReproducibilityManifest`; `tuning.rs` provides a `BoundedPlanTuner`. Execution
is real: `crates/incin-backends/src/cpu/compiled.rs` (432 lines) is a reference
evaluator that walks a plan node by node through the ordinary CPU executors and
returns `CpuStorage` outputs, and it reports per-operation eligibility through
`compiled_support()`.

What does not work: fusion. `compiled/fusion.rs` finds candidates with a
pointwise-only `can_fuse` over 14 operation kinds, and `FusionPass::apply`
returns the graph unchanged for an empty candidate set and
`Err("compiled fusion has no executable fused descriptor lowering")` for a
non-empty one. Its own doc comment says the candidate search "does not prove that
an intermediate value has no other consumers". So the pass is inspection, and it
fails closed rather than pretending.

That is the correct behaviour for the state it is in, but the task that would
change it is untracked. `codebase-truth-audit.md` lists **CMP-005, "Replace fake
fusion with fusion groups", priority P1**, depending on CMP-001 and CMP-003, and
requiring that a fused unit retain the ordered operation sequence, that
single-use producer outputs be proven rather than assumed, and that fused output
and gradients be compared against unfused execution. There is no
`docs/plan/tasks/CMP-005.md`. CMP-001, 002, 003, 004 and 006 all have task files
and are all checked off. CMP-005 is the only number in the range with neither a
task file nor a GitHub issue, and 33 open issues mention neither fusion nor a
compiler.

The whole module also sits behind `incin::experimental::compiled`, and nothing in
`Trainer` or `fit` touches it. Its consumers are eight test files in
`crates/incin-core/tests/` and two consumer fixtures. So the compiled path is
verified in isolation and unreachable from the training loop, which is a separate
gap from fusion and should not be conflated with it.

Recommendation: file CMP-005 as a GitHub issue rather than reviving the ledger,
which is closed. Scope it to the smallest honest unit: prove exclusive
consumption of the intermediate value, emit a fused pointwise chain as a single
`KernelDefinition` (which is exactly what `codegen/ir.rs` already builds, so this
is where the two stranded bodies of work meet), and gate acceptance on comparing
fused against unfused output *and* gradients. Anything larger reproduces the
reason the current pass fails closed.

Risk: fusing across a saved-for-backward value silently changes what the tape can
replay. The audit calls this out and the current pass avoids it only by refusing
everything.

Depends on: codegen-adoption (shares `ir.rs` as the fused-kernel representation).
