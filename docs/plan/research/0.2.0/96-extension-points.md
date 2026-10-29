# #96 Extension points — SOTA survey: custom dtypes and device identity

Research lane S3 for issue #96. Survey only: how other frameworks let
downstream authors add dtypes/devices while keeping their guarantees, mapped
onto incin's five subsystems and the `TensorElement` seal. No recommendation;
the seal and device-identity decisions belong to the orchestrator at
integration. Complements `96-custom-dtypes-devices.md` (in-tree finding),
which this lane did not modify.

Drift note (verified against source on this branch, `feat/custom-autograd-dtype`):
the issue text predates several landings. `DeviceKind` is no longer a fully
closed enum — it is `#[non_exhaustive]` with a `Custom(u64)` variant and
`DeviceId::custom(namespace, ord)` (`crates/incin-core/src/tensor/device.rs`).
`CapabilityQuery.dtype` is already a `DTypeDescriptor`
(`crates/incin-core/src/exec/capability.rs:57`). The accelerator inherent
helpers are already `pub(crate)` (`docs/book/src/whats_not_finished.md`).
The `BuiltinDType` doc comment's closed-vocabulary list now names: distributed
static planners plus plan digests, `CollectiveTuningProblem::new_static`,
the backend kernel-table fast-path lookup, and the `safetensors` export
(`crates/incin-core/src/tensor/dtype/traits.rs`). The SOTA below is mapped
onto that current list, under the issue's five-subsystem headings.

## 1. Custom-dtype SOTA

Reading key for the table: "element access without patching" means a
downstream crate ships a dtype with a usable element type and executes ops on
it, touching no framework source.

| Framework | Downstream dtype + element access, no patch? | What stops abuse (safety / correctness) | Where a closed set is still keyed, and how it is documented |
|---|---|---|---|
| PyTorch | New `torch.dtype` identity: no. `ScalarType` is a closed C++ enum (`c10/core/ScalarType.h`, `AT_FORALL_SCALAR_TYPES_*` macros); a genuinely new dtype is a core patch (the hard path). What most custom-dtype authors actually do instead: `__torch_dispatch__` tensor subclasses that re-encode values in *existing* dtypes (quantized, sparse, masked, lazy), plus `torch.library` custom ops with explicit FakeTensor/meta kernels for shape/dtype/device inference. | Subclass authors must handle every dispatched op or explicitly fall through; unhandled ops raise rather than silently compute. Custom ops require a meta kernel before `torch.compile` accepts them; `opcheck` lints registration misuse. | Closed `ScalarType` is documented as the dtype inventory in `tensor_attributes`; per-op dtype support is enforced by `AT_DISPATCH_*` macros at each kernel site — an unlisted dtype is a loud dispatch failure, not a coercion. |
| JAX + `ml_dtypes` | Yes, by design, but split across two packages. New storage dtypes (fp8 variants, int4/int2, fp6, bfloat16) ship in the separate `ml_dtypes` package as NumPy-registered dtypes, not in JAX core; JAX consumes them via `jax.extend` (custom primitives need `def_abstract_eval` plus a registered MLIR lowering rule to run under `jit`). Extended (non-array) dtypes such as PRNG keys subclass `jax.dtypes.extended`. | Closed-switch subsystems refuse loudly: `grad`/`jacfwd` raise `TypeError` on extended dtypes (`jax/_src/api.py`); promotion and canonicalization reject non-JAX dtypes. A primitive without a lowering rule traces but fails at compile time. `jax.extend` carries no cross-release compatibility guarantee (stated in its docs). | The valid array-dtype set (`jax.extend.core.array_types`) is enumerated; transforms document which dtype classes they accept. Lowering coverage is the author's registered rule per primitive — the framework validates presence, not completeness. |
| TVM (Bring Your Own Datatypes) | Yes: `tvm.target.datatype.register(name, type_code)` claims an unused type code, so the compiler can *parse* `custom[posit]16`; execution needs per-`(op, target, dtype, bits)` lowering functions via `register_op`. | Nothing in the framework checks lowering coverage: a parsed-but-unlowered program fails at compile time. Correctness of the lowering (usually a call into an external library over bit-identical integers) is entirely the author's, reviewed by nothing. | Type codes are a finite registry with hard-coded meanings for built-ins; custom codes live in the documented-unused range. Coverage is per-site registration, visible only by inspection. |
| MLIR | Yes: the type system is open — dialects define custom types with no fixed list (`MLIR LangRef`: "types may have application-specific semantics"); only the builtin dialect's types are shared. | Verifiers: ops declare type constraints via traits/interfaces, and invalid IR is rejected at construction/verification. A custom type that no pass understands simply matches no pattern and lowers nowhere — failure, not miscompilation. | The builtin dialect is the closed shared vocabulary; everything else is dialect-scoped. Interop points (e.g. StableHLO) re-close the set at the boundary by specifying exactly which types cross it. |
| Candle (Rust) | No. `DType` is a closed enum (`U8, U32, I64, BF16, F16, F32, F64`; docs.rs `candle_core`) and `Device` is likewise closed — both require a patch. The cautionary example: closed-closed is simple and total, and it forecloses out-of-tree dtypes *and* devices. | Exhaustive `match` everywhere: the compiler itself is the refusal mechanism. | The enum definitions are the documentation; adding a variant is a breaking, tree-wide change. |
| Burn (Rust) | Backends yes, dtypes no. Everything is generic over the open `Backend` trait ("Custom Backend Extension" is a headline feature; third-party backends implement it out-of-tree), but kernel code queries `supports_dtype` / `dtype_usage` against the closed `burn::tensor::DType` enum — a backend advertises capability rows over a fixed dtype set. | The trait forces each backend to declare dtype support per device; unsupported pairs are refused at dispatch. Dtype-set evolution is still a framework release. | Closed `DType` enum plus per-backend capability rows; openness lives one layer up (choice of backend), not in the dtype vocabulary. |
| dfdx (Rust) | Element types yes, within bounds: `Dtype`/`Unit` are open traits (plus `SafeZeros`, `NotMixedPrecision` markers and the `AMP<W>` wrapper), so a POD newtype can participate without patching. | Trait bounds are the gate: arithmetic requires `Dtype`, zero-init requires `SafeZeros`. Anything not implementing the bound does not compile at the call site. | No central closed dtype registry; closedness (if any) lives in per-backend kernel impls. |
| tinygrad (Python) | Technically constructible (`DType` is a class with instances in `tinygrad/dtype.py`, not an enum), closed in practice: codegen, dispatch, and the `fmt`/interop helpers switch on known instances, so an unknown instance parses but executes nowhere. Partially verified — instance-construction path confirmed from source, backend coverage not exhaustively audited in this lane. | Unknown dtypes fail at lowering/codegen rather than being coerced; NumPy/torch interop maps unknown dtypes to `None`/fallback explicitly. | The `dtypes.*` instance list is the de-facto closed set; no formal extension registry was found. |

### Pattern extraction

Three patterns recur:

1. **Seal-the-element** (Candle; PyTorch's `ScalarType`; incin's `TensorElement` today): the set of element types is fixed by the framework. Fail-closed by construction — there is no unknown element to mishandle — at the price of closing the surface entirely.
2. **Open-with-bounds** (dfdx traits; incin's `DType`/`ConstDType` + the proposed unsealed bound): anyone may add a type satisfying public safety bounds. Fail-closed *only if* every consumer branches on identity instead of unwrapping a closed mapping. The counterexample pattern is a silent coercion default: this tree has one live instance at `crates/incin-backends/src/dist/nccl/transport.rs:236,277,438` (`builtin_id().unwrap_or(DTypeId::F32)`), currently unreachable for custom dtypes only because `BuiltinDType` bounds upstream refuse them first. Any bound-widening must eliminate or gate that fallback or the guarantee inverts.
3. **Registry-with-capability-rows** (Burn `supports_dtype`; TVM per-op lowering registration; MLIR dialect types + verifier traits; JAX lowering rules per primitive): admission is a query against rows the author (or backend) registers; unknown keys match no row and the miss path is a typed refusal. This is the pattern that preserves "unknown things refuse loudly" while staying open — *provided* the miss path cannot be bypassed and rows are allow-lists, not denylists.

## 2. The seal: Keep vs Unseal-with-bounds

The exact bound named in the issue and already enforced on current implementors
(`crates/incin-core/src/tensor/dtype/traits.rs:114-124`):

```rust
bytemuck::NoUninit + bytemuck::Zeroable + Copy + Debug + Send + Sync + 'static
// (+ the private `sealed::TensorElementSealed`, which is the policy component)
```

`f16`/`bf16` (foreign `half`-crate types admitted via `impl_plain_builtin_dtype!`)
prove the safety half is already satisfied by non-primitive types; `Q8_0`
(block-quantized, deliberately *without* an element) proves the framework
already distinguishes "has a Rust scalar" from "is a dtype". Unsealing would
remove only the `TensorElementSealed` gate, leaving the POD bounds.

What each of the five subsystems does with an unsealed-but-still-non-builtin
element type (i.e. seal lifts, `BuiltinDType` gates stay):

| Subsystem | Works unchanged | Still gated by `BuiltinDType` | New refusal paths that appear |
|---|---|---|---|
| Distributed plans (static planners in `dist/plan/planner.rs`, `dist/plan/collective.rs`, `dist/data_parallel.rs`, `dist/pipeline.rs`, `dist/fsdp.rs`, `dist/tensor_parallel.rs`; `CollectiveTuningProblem::new_static`) | Nothing new executes: bounds are `K: ConstDType + BuiltinDType + …DType`, so a custom dtype is still refused at compile time, identically under Keep and Unseal. | All of it: plan digests fingerprint by `DTypeId`, the tuning cache keys by `DTypeId`. Descriptor-keyed digests are the future-phase work either way. | None from the seal alone. If bounds later widen to `ConstDType`, digests must serialize `DTypeKey` (namespace/name/version) and mixed-version peers must refuse key mismatch — a wire-compat surface that does not exist today. |
| Capability registry | Fully: `CapabilityQuery.dtype` is already a `DTypeDescriptor`, and the allow-lists in `incin-backends/src/capability/constants.rs` are descriptor lists. An unknown descriptor matches no row, yielding `Unsupported` with a typed reason — the registry-with-capability-rows pattern, already fail-closed. A custom backend can advertise its own descriptor rows today. | No `BuiltinDType` bound was found in the registry path; gating is by row membership, not by trait. | None: misses are already typed refusals. Unsealing changes nothing here except letting a custom dtype also satisfy `PlainDType`-gated *callers* that build queries. |
| Operation catalog | Attribute layer: works — creation/data/full/quantization attributes all carry `DTypeDescriptor` (`exec/catalog/meta.rs`), and shape inference keys off descriptors. | Under audit in this lane: whether `DTypeRule` (`exec/catalog/table.rs:26`) or per-op exactness bounds (e.g. `exec/precision.rs` `Exact<K>`) close over `DTypeId` exhaustively was not line-audited here; the companion file `96-custom-dtypes-devices.md` records the catalog as descriptor-driven. Treat exact rule coverage as the orchestrator's verification item. | If any rule matches closed ID sets at runtime, a custom descriptor falls to the rule's else-arm — typed refusal if the arm is `Unsupported`, silent narrowing if it is not. Each else-arm wants one inspection. |
| Serialization | Postcard/state envelope: works — snapshots carry `DTypeDescriptor` (`nn/state.rs:976`) and round-trip custom dtypes. `safetensors` export: unchanged refusal — `safetensors_dtype` matches `builtin_id()` and returns a named error for anything else (`tensor-core serialize.rs:17-33`); the `DTypeKey` version field already exists for future descriptor-in-metadata export. | The `safetensors` table, explicitly and loudly. | None new; the existing error names the dtype. Unsealing does not touch the export table. |
| Backend kernel tables | Unchanged: fast-path dispatch resolves `builtin_id()` and returns typed errors (`incin-backends/src/kernel/types.rs:87` via `ok_or_else`; `cpu/mod.rs:295` `UnsupportedDType`; per-backend `match`es on `Some(DTypeId::…)`) — miss-is-refusal, already fail-closed. A custom backend ships its own kernels keyed by its own descriptors. | The built-in tables, by design (fast path for built-ins, not admission control for customs). | One hazard, pre-existing: the NCCL transport's `unwrap_or(DTypeId::F32)` coercion (section 1). Harmless while `BuiltinDType` bounds hold upstream; must be converted to a typed refusal before any bound widening reaches it. |

Keep vs Unseal, symmetric trade-offs (decision reserved):

| | Keep the seal (`SEC-005` as policy) | Unseal with the POD bound |
|---|---|---|
| What it buys | Zero new surface: `PlainDType`, `from_slice`-style typed constructors, and the target-layer typed paths stay built-in-only by construction. No audit of element-consuming call sites needed. FP8 follows the `Q8_0` precedent (opaque/block representation, no element) per #94. | Custom scalar dtypes (the posit24 sketch: a 3-byte POD newtype) get element access — slicing, typed construction, host interop — without re-proving safety per type; the bound *is* the proof obligation, satisfiable by any POD newtype including foreign-crate types, exactly as `half` proved. FP8 follows the `bf16` precedent per #94. |
| What it costs | The policy/safety conflation persists: the bound admits the types, the seal refuses them, so the refusal reads as arbitrary to downstream authors and must be re-justified each time (FP8 now, the next format later). The posit24 sketch remains inexpressible as an element dtype. | Every consumer of `TensorElement`/`PlainDType` becomes reachable by foreign types: each must be audited to confirm it branches on descriptor/encoding rather than assuming a built-in layout. The `sealed` module's purpose shifts from "only incin implements this" to documentation. `PROPOSALS.md` must record the decision either way (issue acceptance criterion). |
| Failure mode if wrong | Over-closure: pressure to punch per-dtype holes (one-off exceptions are worse than a principled bound). | Under-audit: a call site that assumes `Elem` is one of nine known layouts miscompiles or misreads a foreign POD. The mitigation is the audit itself, plus keeping `BuiltinDType` gates in the four closed subsystems during the transition. |

## 3. Device identity SOTA

| Framework | Third-party device without touching the device enum? | How the frozen vocabulary survives |
|---|---|---|
| PyTorch `torch.privateuse1` | Yes, with limits: out-of-tree backends register kernels to the `PrivateUse1` dispatch key, implement the hooks interface (`at::PrivateUse1HooksInterface`), and rename the device once per process (`rename_privateuse1_backend("npu")`), after which `"npu"` works as an ordinary device string. Maintained backends (CUDA, MPS, XLA-via-plugin) remain first-class `DeviceType` variants. | Two tiers: the `DeviceType` enum stays frozen for maintained backends; exactly one out-of-tree slot exists per process, and unhandled ops on it fail at dispatch. Limitation worth noting: a *single* `PrivateUse1` slot means two third-party backends cannot coexist in one process. |
| JAX PJRT plugins | Yes, and it is the *only* path: new hardware (Apple Metal via `jax-metallib`, ROCm, Intel GPU) ships as a PJRT C-API plugin package plus a `register_backend_factory` / `JAX_PLATFORMS` selection — no JAX-core device-enum change, because device identity is a backend string plus plugin-discovered topology, not a closed enum. | There is no frozen device vocabulary to break: `jax.devices(backend=…)` enumerates whatever the registered plugin reports. The stability contract lives in the PJRT C ABI and StableHLO, not in a device list. Compilation failures for unsupported ops surface from the plugin's XLA lowering. |

Map to incin. `DeviceKind::Custom(u64)` + `DeviceId::custom(namespace, ord)` already exists, so the "closed enum blocks third-party devices" premise is partly landed; what the issue sketch additionally proposes (`DeviceKind::external(DeviceKey)` with a structured namespace/name/version key, and a `Device::kind()` accessor — neither exists today; the `Device` trait resolves identity via `to_incin` into `DeviceId`) is richness, not openness. On the #6 reconciliation: the maintainer's "ROCm gets a first-class variant" is consistent with the PyTorch two-tier evidence — maintained backends get enum variants with exhaustive matching and per-backend kernel tables, third-party backends get the open namespace. The JAX evidence pulls the other way (even maintained hardware lives behind the plugin interface), but incin has already chosen first-class ROCm, so the live question is only whether the third-party tier is `Custom(u64)` or a structured key — not whether a third-party tier exists.

Custom-namespace vs External(DeviceKey) vs upstream-only:

| | `Custom(u64)` status quo | `External(DeviceKey)` (issue sketch) | Upstream-only (closed enum) |
|---|---|---|---|
| Mechanics | Opaque `u64` namespace chosen by the external backend; carried through metadata/serialization uninterpreted; `name()` reports `"custom"` for all of them. Multiple third-party backends coexist (unlike PyTorch's single `PrivateUse1` slot). | Structured key paralleling `DTypeKey` (namespace + name + version); distinct backends distinguishable in diagnostics, plan digests, and topology fingerprints; version bumps give the same refuse-on-mismatch story as dtype versions. Requires migrating existing `Custom(u64)` keys. | Revert to fully closed: every backend upstreams a variant. Exhaustive matching everywhere; device selection/placement/`to_device` need no open-identity handling. |
| Frozen-vocabulary implication | Vocabulary frozen in *shape* (five variants incl. `Custom`, `#[non_exhaustive]`), open in *membership*. `FROZEN_FOUNDATIONS.md` speaks to this point only by omission: its frozen rows cover the operation declaration, the descriptor vocabulary, the dispatch path, the execution contract, and the capability declaration — no row covers `DeviceKind`, and the experimental-boundaries table reserves "backend resource/session abstractions" as still-moving ("remain intentionally small until a concrete backend requires a larger resource model"). Verbatim, that is the only sentence in the file that touches backend identity, and it permits growth on demand rather than freezing the set. | Same shape-frozen property, richer membership: digests and fingerprints hash a structured key instead of an opaque integer, so equality semantics stay well-defined while debuggability improves. Still no frozen-foundation conflict, for the same reason: device identity is not a listed foundation. | Vocabulary frozen in both shape and membership: strongest exhaustiveness, and every new accelerator — maintained or proprietary — becomes a core change, i.e. the #6 ROCm pattern repeated for all hardware, including unreleased/proprietary targets that cannot upstream. Directly contradicts the issue's stated purpose ("makes an unreleased or proprietary accelerator impossible to support out of tree"). |
| Interop notes | Distributed plans record free-form architecture strings (`DeviceIdentity::architecture`, `dist/mesh.rs:262`); an opaque `u64` reaches digests only as an integer, so two backends sharing little else must still avoid key collision by convention. | Versioned keys compose with the plan-digest story: a backend that changes its topology encoding bumps its version and old plans refuse. Collision resistance comes from namespaced names rather than bare integers. | No interop questions arise, because no out-of-tree backend exists to interoperate with. |

## Decision (orchestrator, 2026-09-25 — maintainer delegated both calls)

**Seal: UNSEAL with the POD bound.** Remove only the private
`TensorElementSealed` gate, keeping `bytemuck::NoUninit + Zeroable +
Copy + Debug + Send + Sync + 'static` as the proof obligation any POD
newtype (including foreign-crate types, `half` precedent) satisfies.
Rationale: the bound is policy, not safety; the survey shows every
fail-closed framework either opens with bounds (dfdx) or rows (Burn/TVM/
MLIR/JAX) while only closed-closed Candle forecloses entirely — and
incin's registry, kernel tables, and safetensors paths already refuse
unknown descriptors loudly, so the open surface stays fail-closed.
Preconditions before the bound widens (all recorded here so the
implementation lane cannot skip them): (a) the NCCL
`builtin_id().unwrap_or(DTypeId::F32)` error-field coercions
(`dist/nccl/transport.rs:236,277,438`) become honest unknown-descriptor
naming — today they only mislabel refusals, after widening they would
mislabel execution; (b) audit every `TensorElement`/`PlainDType`
consumer for built-in-layout assumptions (catalog exact-rule else-arms
especially — typed-`Unsupported` required, silent narrowing forbidden);
(c) `BuiltinDType` gates stay in the four closed subsystems during the
transition; (d) the ruling is recorded in `PROPOSALS.md` (issue
acceptance criterion). Consequence recorded for #94: FP8 follows the
`bf16` precedent (real element type), not the `Q8_0` one.

**Device identity: two tiers.** Maintained backends get first-class
`DeviceKind` variants (ROCm per maintainer decision on #6; the frozen
vocabulary is shape-frozen, and `FROZEN_FOUNDATIONS.md` lists no
`DeviceKind` row, so a variant addition is growth on demand, not a
foundation break). Third-party backends get a structured
`External(DeviceKey)` (namespace + name + version, paralleling the
#93-established `DTypeKey` pattern) instead of the opaque
`Custom(u64)`: versioned keys compose with plan digests (refuse on
mismatch), diagnostics name names instead of integers, and multiple
third-party backends coexist (unlike PyTorch's single `PrivateUse1`
slot). Migrating the existing `Custom(u64)` is small (one constructor
+ one test usage in-tree). The JAX plugin-model (C ABI surface) is
explicitly not adopted: no such surface exists, and the namespace key
covers the unreleased/proprietary-accelerator case the issue demands.
Inherent-helper removal timing stays open as the issue allows.

## Sources

- Issue #96 body (`gh api repos/xupremix/incin/issues/96 --jq '.body'`); `crates/incin-core/src/tensor/dtype/traits.rs`, `registry.rs`, `builtin.rs`; `crates/incin-core/src/tensor/device.rs`; `docs/book/src/backend_authoring.md`; `docs/book/src/whats_not_finished.md`; `docs/FROZEN_FOUNDATIONS.md`.
- PyTorch: `torch.privateuse1` tutorial (docs.pytorch.org/tutorials/advanced/privateuseone.html); `rename_privateuse1_backend` docs; `__torch_dispatch__` rationale (dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557); `c10/core/ScalarType.h`; custom-ops C++ tutorial (FakeTensor/meta kernels, `opcheck`).
- JAX: `jax-ml/ml_dtypes` repo; `jax/_src/dtypes.py` (`extended`, `prng_key`); `jax/_src/api.py` (extended-dtype refusals); JEP 15856 (`jax.extend`); `jax.extend.backend` (`register_backend_factory`); OpenXLA PJRT integration guide; `jax-metallib` (Metal plugin precedent); JAX `about.html` (pluggable-backend statement).
- TVM: "Bring Your Own Datatypes" (tvm.apache.org/2020/09/26); MLIR LangRef type-system section.
- Rust frameworks: `candle_core::DType` (docs.rs, 0.6.0: closed enum); `burn::tensor::backend::Backend` + `supports_dtype`/`dtype_usage` (burn.dev/docs); `dfdx::dtypes` (`Dtype`/`Unit`/`SafeZeros` open traits, docs.rs); `tinygrad/tinygrad/dtype.py` (class-based `DType`, instance list).
