# Frozen foundations

The parts of the tree that carry the most weight, are finished, and are not
supposed to be rewritten casually. Every entry names the mechanism that keeps
it true rather than asking anyone to remember it.

Read this before proposing a change to any file listed here. A change to a
frozen foundation is not forbidden, but it is a change to a contract several
consumers depend on, and it costs far more than it looks like it costs.

`crates/incin-core/tests/frozen_foundations.rs` fails if any path named below
stops existing, so this file cannot rot into a list of deleted files.

## What is frozen

| Foundation | Where | Why it is finished | What keeps it true |
|---|---|---|---|
| The operation declaration | `crates/incin-core/src/operation_catalog.rs` | One authoritative declaration of every operation the library has. 174 rows, each naming an identity, a semantic profile, an attribute type, an operand arity and historical family metadata | Every consumer is generated from it; a row cannot be added to one consumer only |
| The descriptor vocabulary | `crates/incin-core/src/exec/catalog/` | `op::X` markers, `Descriptor<O>`, typed attributes, `OperationCatalogEntry` and its classification enums are all expanded from the declaration above, split by concern per `docs/CONVENTIONS.md` (`table.rs` for the row construction and `OPERATION_CATALOG` itself, `descriptor.rs` for `Descriptor<O>` and `CanonicalOperation`, `attributes.rs` for the per-operation `AttributeContract` impls, and so on) | `incin_operation_catalog!(define_catalog)`; the sealed `CanonicalOperation` trait keeps new identities inside the crate |
| The dispatch path | `crates/incin-core/src/exec/dispatch.rs` | `execute::<O, B>` is the single route from an operation to a kernel: validate the descriptor against real storage metadata, query the exact capability row per operand, then dispatch | `B: Execute<O>` is a compile-time bound, and there is no default method to fall through |
| The execution contract | `crates/incin-core/src/exec/request.rs`, `crates/incin-core/src/tensor/backend/execute.rs` | `Execute<O>` carries an associated `Output`, so an operation returning a pair is expressible without a special case. `TensorHandle` carries checked metadata, so an executor never re-derives it | Type-checked at every call site |
| The capability declaration | `crates/incin-backends/src/capability/declarations.rs` | Groups are rule *shapes*, not operation families. Migrating an operation is one more name in an existing list, not a new group and a new arm in every consumer | One declaration feeds the capability rows and the canonical `Execute<O>` completeness proof |
| The completeness proof | `crates/incin-backends/src/cpu/canonical/` | A capability row advertised without an `Execute<op::X>` behind it does not compile | `assert_every_advertised_row_executes!`, driven by the same declaration that generates the rows |
| The generated evidence | `docs/capabilities.md`, `docs/OPERATION_SEMANTICS.md`, `audit-evidence/FND-005/cpu-migration-status.md` | Every number in them is derived from the Rust source rather than written by hand | A test fails when the committed file and the regenerated one differ; `docs/README.md` lists the regeneration command for each |

> **Adoption, as distinct from design.** The stable tensor methods route
> backend-executable operations through `dispatch::execute` and its shaped
> variants. The old core operation-family traits have been removed from
> production source. Backend-local helpers, fused special execution sites,
> tracing adapters, and compatibility tests use ordinary functions or explicit
> execution sites. They are not a backend-authoring API or a second stable
> tensor path.

The shape of that table is the point. Each row is a decision made once and then
made unrepeatable, so the cost of migrating operation number 118 is the same as
the cost of migrating number 5 was, rather than growing with the count.

## What remains intentionally experimental

These are current boundaries, not unfinished descriptor migration work. They
may change in a later focused design effort, but they do not introduce a
second stable execution path.

| Surface | Why it still moves |
|---|---|
| Experimental graph, compiled, distributed, import, and tooling APIs | Their dedicated documentation defines the supported subset and limitations; they are not part of the stable tensor execution contract |
| True in-place mutation and aliasing semantics | Ownership, views, allocation identity, and autograd versioning require a separate focused design |
| Richer training contexts and custom automatic differentiation extensions | Train-mode evolution, custom VJP/JVP/batching, and higher-order AD remain future architecture |
| Backend resource/session abstractions | Backend identity and capability contracts remain intentionally small until a concrete backend requires a larger resource model |
| Ragged, sparse, and fully mature distributed runtime support | These are future capabilities, not gaps in the canonical descriptor path |

## Historical migration notes

The following records the dependency order used during the foundation
migration. It is historical context, not a list of unfinished HND-004b work.
The current execution contract is the descriptor and `Execute<O>` path above.

Each step is blocked by the one above it, and the reason is stated rather than
implied.

The canonical CPU migration is complete for all 164 backend-executable
operations. Ten catalog entries remain at explicit non-backend execution
sites and are not counted as missing kernel executors.

The dtype-set blocker is closed. `embedding` and `cross_entropy_loss` were the
only two operations whose operands differ in dtype by construction, and both
are migrated using the union technique described in the table above. A row
stating `INDEX_AND_F32_DTYPES`, the descriptor's own per-operand contract
refusing the wrong pair before any capability query, and `f32_only` in the
executor enforcing whichever operand the union cannot pin down alone. No
`CapabilityRule` struct change was needed or made.

1. **Let a descriptor carry a payload and a weight set.** `rnn` and `lstm` have
   an operand arity that admits their states and not their matrices. The data
   constructors now carry their payload through `DataAttributes`; recurrent
   descriptors still cannot name their complete request.
2. **Add a distribution registry**, mapping a name and a parameter buffer back
   to a sampler, which is the whole of what `sample` needs.
3. **Widen `Execute` to the sites it cannot reach**, or split them off into a
   contract that can. Sixteen operations are not pending migrations at all,
   and counting them as such overstates the remaining work by roughly 30%.
4. **Keep legacy operation-family adapters absent from the authoring surface**
   and bound each remaining compatibility call site by the capability it
   actually uses. This is the step that ends the dual architecture. It is no
   longer blocked by migration coverage: the remaining executable gaps are
   explicit and the stable data constructors use canonical dispatch.
5. **Completed: delete the broad family rows and grouped adapters.** The
   compatibility adapter and migration-incomplete guard were removed when the
   descriptor/`Execute<O>` path became canonical. This historical step is
   recorded here to explain the dependency order, not as remaining work.
