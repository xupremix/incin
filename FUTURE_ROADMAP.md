# Future Roadmap — Post-Native-Backend Ideas

Captured from a design-discussion thread on 2026-07-09/10. None of this is
scheduled — it's a backlog of validated ideas for milestones *after* the
native backend (see `NATIVE_BACKEND_PLAN.md`) and distributed training prove
themselves. Do not pull any of this forward into the current milestone.

## Observability / Visualization ("TensorBoard, but native")

**Why it's cheap later:** the autograd tape (native backend) and, later, the
type-level `Placement`/`Sharding` data (distributed milestone) already carry
everything a visualizer needs — this is mostly surfacing existing structure,
not new instrumentation.

- **Zero-config, single-binary.** `Trainer::new().watch()` spins up a live
  dashboard in the same process — no separate `tensorboard --logdir`
  server, no writing summary files to disk first.
- **TUI option, not just a browser.** An `htop`-style terminal dashboard for
  live loss/gradient-norm/throughput over SSH, no port-forwarding — most
  training happens on a remote box, and that's the actual TensorBoard
  friction point.
- **Live gradient-flow visualization tied to the real tape.** Color-code
  layers by live gradient magnitude as training runs, so a vanishing/
  exploding layer is visible in real time instead of discovered after the
  fact in a static histogram tab.
- **Shape-aware, hoverable graph.** Since every tensor's shape is known at
  compile time, show real shapes flowing through the graph interactively —
  an interactive, type-checked architecture diagram instead of disconnected
  scalar tabs.

### Distributed extension (once the Placement/Sharding milestone exists)

- **Device-mesh view** — a topology diagram of the device grid, with each
  tensor's shards highlighted on the devices that actually hold them, live.
- **Communication as visible traffic** — an all-reduce/all-gather firing
  (gradient sync, tensor-parallel matmul needing a gather) shown as an
  animated edge between devices with real bytes/timing, so you can *see*
  whether training is network-bound vs. compute-bound instead of guessing
  from wall-clock numbers.
- **Per-device gradient/loss views side by side** — catch a straggler
  device or a desync bug (one replica's loss diverging) visually instead of
  digging through per-rank logs.

**Sequencing:** firmly *after* the native backend milestone (and, for the
distributed extension, after the distributed/Placement milestone) — fun to
build, easy to over-invest in before the underlying autograd/op coverage is
solid. Treat as "a differentiator to build once there's something real to
visualize," not a distraction now.

## Common-problem-fixing features (product pitches, ranked by leverage)

These are the sharpest because they're largely *free* — the machinery they
need (tape, gradient checks) already has to exist for the native backend
itself; making it user-facing is the only new work.

1. **"Where did this NaN come from?"** — walk the tape checking for the
   first non-finite value and point directly at the offending op/layer,
   turning a multi-hour bisection into an instant answer.
2. **Gradient-check as a one-line public API.** The native backend's own
   test suite already does finite-difference gradient checking to validate
   its ops (see Phase 5 / NATBACK-10 parity harness). Exposing this as
   `#[derive(GradCheck)]` or a `verify_gradients()` call would let any user
   confirm a custom layer's backward pass matches numerical differentiation
   — PyTorch has `torch.autograd.gradcheck` but it's obscure/manual; making
   it first-class and discoverable is the differentiator.
3. **Tensor layout bugs (NCHW vs. NHWC) caught at compile time.** A very
   common silent-wrong-answer bug in vision code (feed a channels-last
   tensor to a channels-first conv — it "works," wrong numbers, no crash).
   Encoding layout as part of the shape type would catch a bug class
   PyTorch structurally cannot.
4. **Memory profiling mapped to actual named tensors**, not just a raw byte
   total — attribute OOM/usage to specific tensors/layers instead of an
   undifferentiated count.
5. **Reproducibility as a structural guarantee**, not a checklist — seed
   threading built into the type/builder system so "same seed → bit-
   identical result" is just true, instead of separately managing
   numpy/torch/cuda/hash-seed.

## Positioning notes (from the same discussion, for future PROJECT.md updates)

- **Not a frontier-lab training replacement.** Switching cost for a
  multi-thousand-accelerator training run dwarfs any ergonomics win a new
  library could offer — don't pitch it that way.
- **Realistic audience:** individual researchers, small-to-mid teams,
  single-node/small-multi-node training, embedded/edge/WASM deployment —
  places where compile-time safety and a no-Python-runtime deployment story
  are real, felt advantages.
- **The highest-leverage single investment for adoption:** diagnostic error
  message quality (`#[diagnostic::on_unimplemented]`-style tutoring
  messages for shape/device/dtype mismatches). Rust's generic error spam is
  the biggest first-five-minutes turn-off for newcomers; if kindle's errors
  read like a tutor instead, that's the actual hook — more so than any
  individual feature above.
- **"Prototype and production are the same binary"** — what you trained is
  what you deploy, compiled, no ONNX/TorchScript export dance and no
  Python runtime in the shipping artifact. Strongest pitch to people
  building an actual product on top of this, not just researchers.

---
*Captured 2026-07-10 from a design-discussion thread following the native
backend milestone's initialization. See `NATIVE_BACKEND_PLAN.md` for the
current milestone and `.planning/PROJECT.md` for locked scope/decisions.*
