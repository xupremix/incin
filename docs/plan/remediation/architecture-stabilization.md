# Architecture stabilization gate

This is the active remediation plan after the repository-wide architecture and
correctness audit. Feature work that expands the execution, backend, public API,
or persistence surfaces remains gated until the proof gates below pass.

The binding contracts remain `docs/FROZEN_FOUNDATIONS.md`,
`docs/API_DESIGN.md`, `docs/ERROR_CONTRACT.md`, and
`docs/INVARIANT_TYPES.md`. This plan records implementation order and does not
override them.

## Decisions

1. The canonical operation catalog, descriptors, capability query, and
   `Execute<O>` path remain the only stable backend execution architecture.
2. `SupportLevel::Native` is always admissible. `Composed` requires
   `FallbackPolicy::AllowComposition` or `AllowTransfer`. `Fallback` requires
   `AllowTransfer`. The default permits same-device composition and denies
   transfer fallback.
3. Stable execution policy exposes only decisions with production enforcement.
   Allocator selection and a general determinism promise leave the stable
   execution context until an executor can honor them.
4. Recoverable Rust tensor operators return the typed `Result` promised by the
   error contract.
5. `TensorElement` is genuinely sealed. Custom logical `DType` descriptors
   remain supported through the deliberate dtype extension contract.
6. Model checkpoints receive an explicit versioned envelope. Loading validates
   all state before commit and preserves each parameter's existing placement.
7. `DummyBackend` is retired. Shape-only tests use proof and descriptor logic;
   numerical behavior uses a real backend; any remaining fixture advertises
   only operations it actually implements.
8. Public backend operation helpers that bypass canonical execution become
   crate-private or are removed. No compatibility feature recreates the dual
   API.
9. Public API, hidden-item, unsafe, documentation, book, and release gates are
   strengthened to test the claimed behavior rather than inventories or proxy
   state.

## Dependency order

| Phase | Work | Completion evidence |
| --- | --- | --- |
| R0 | Restore facade and workspace compilation after API moves | Trainer test, backend architecture regression, and workspace all-target check pass |
| R1 | Enforce support policy and remove unenforced policy axes | Dispatch spy tests cover native, composed, transfer fallback, custom, shaped, and metadata-free paths |
| R2 | Repair operator results and dtype sealing | Operator error-injection tests and downstream compile-pass/compile-fail fixtures pass |
| R3 | Version checkpoints and remove ignored placement arguments | Compatibility, corruption, rollback, limits, and placement tests pass |
| R4 | Retire `DummyBackend` and migrate its tests | No public or test-utils export remains; no blanket `Execute<O>` test implementation remains |
| R5 | Contract legacy backend APIs | Canonical completeness proof passes and downstream fixtures cannot call legacy operation methods |
| R6 | Strengthen API and governance gates | Approved public profiles and semantic unsafe/hidden checks pass |
| R7 | Harden the book and release process | Real browser interactions pass; releases remain draft until every expected checksummed artifact exists |
| R8 | Final stabilization gate | Formatting, focused crates, supported CPU feature matrix, doctests, documentation, packaging, and governance checks pass |

## Breaking changes to document

- Default fallback changes from denying all fallback to allowing same-device
  composition while still denying transfer.
- Unenforced allocator and general determinism settings leave the stable
  execution context.
- Arithmetic operator associated outputs become `Result`.
- The ignored device argument is removed from model loading; relocation remains
  a separate explicit operation.
- External `TensorElement` implementations and direct calls to backend-internal
  operation methods are rejected.
- `DummyBackend` and its test-utils export are removed.

Every phase updates user, backend-author, migration, and generated documentation
for the public behavior it changes before that phase is considered complete.
