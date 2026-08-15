# Frozen foundations

The parts of the tree that carry the most weight, are finished, and are not
supposed to be rewritten again. Everything in this file is load-bearing for the
remaining FND-005 work, and every entry names the mechanism that keeps it true
rather than asking anyone to remember it.

Read this before proposing a change to any file listed here. A change to a
frozen foundation is not forbidden, but it is a change to a contract several
consumers depend on, and it costs far more than it looks like it costs.

`crates/incin-core/tests/frozen_foundations.rs` fails if any path named below
stops existing, so this file cannot rot into a list of deleted files.

## What is frozen

| Foundation | Where | Why it is finished | What keeps it true |
|---|---|---|---|
| The operation declaration | `crates/incin-core/src/operation_catalog.rs` | One authoritative declaration of every operation the library has. 174 rows, each naming an identity, a semantic profile, an attribute type, an operand arity and a legacy source | Every consumer is generated from it; a row cannot be added to one consumer only |
| The descriptor vocabulary | `crates/incin-core/src/exec/catalog.rs` | `op::X` markers, `Descriptor<O>`, typed attributes, `OperationCatalogEntry` and its classification enums are all expanded from the declaration above | `incin_operation_catalog!(define_catalog)`; the sealed `CanonicalOperation` trait keeps new identities inside the crate |
| The dispatch path | `crates/incin-core/src/exec/dispatch.rs` | `execute::<O, B>` is the single route from an operation to a kernel: validate the descriptor against real storage metadata, query the exact capability row per operand, then dispatch | `B: Execute<O>` is a compile-time bound, and there is no default method to fall through |
| The execution contract | `crates/incin-core/src/exec/request.rs`, `crates/incin-core/src/tensor/backend/execute.rs` | `Execute<O>` carries an associated `Output`, so an operation returning a pair is expressible without a special case. `TensorHandle` carries checked metadata, so an executor never re-derives it | Type-checked at every call site |
| The capability declaration | `crates/incin-backends/src/capability.rs` | Groups are rule *shapes*, not operation families. Migrating an operation is one more name in an existing list, not a new group and a new arm in every consumer | One declaration feeds the capability rows and the canonical executors |
| The completeness proof | `crates/incin-backends/src/cpu/canonical.rs` | A capability row advertised without an `Execute<op::X>` behind it does not compile | `assert_every_advertised_row_executes!`, driven by the same declaration that generates the rows |
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

## What is deliberately not frozen

These are the surfaces the remaining work changes. Nothing here should be
treated as settled.

| Surface | Why it still moves |
|---|---|
| The per-operation executor bodies in `cpu/canonical.rs` | 158 of 158 backend-executable operations migrated. Sixteen catalog entries remain at execution sites that `Execute` cannot carry and are tracked separately |
| The legacy core operation-family traits | Removed from production source. Backend-local ordinary helpers and explicit execution sites remain where an operation cannot be represented by `Execute<O>` |
| The broad family capability rows | `Pointwise`, `Reduction`, `Reshape`, `MatMul`, `Conv2d`, `Pool2d`, `Storage`, `Fill`, `Random`, `Normalization`, `Broadcast` are deleted once nothing resolves through them |
| `CapabilityRule`'s single dtype set | It describes an operation, but `dispatch::execute` applies it to each operand in turn. An operation whose operands differ in dtype by construction cannot state the tight per-operand pair directly, and no longer needs to: `INDEX_AND_F32_DTYPES` states the *union* the row can honestly claim, the same trick `descriptor_min_rank` already used for rank, and the descriptor's own per-operand contract (already `TypedContract`/hand-cased in `validate`, not something this added) rejects the wrong combination before any capability query runs. Both operations that needed it — `embedding` and `cross_entropy_loss` — are migrated on that technique, so no struct change to `CapabilityRule` was needed or made |
| `CapabilityRule`'s single rank range | Same cause, same fix already in place: the range states the minimum over *all* operands, which is what `descriptor_min_rank` has always done and what `INDEX_AND_F32_DTYPES`'s rows now also do for rank |
| `Execute`'s reachable sites | Sixteen operations have an `ExecutionSite` the trait cannot carry: they mutate through an operand, produce storage on another backend, or act on autograd state. `ExecutionSite::blocking_reason` states which |

## Next steps, in dependency order

Each step is blocked by the one above it, and the reason is stated rather than
implied.

The canonical CPU migration is complete for all 158 backend-executable
operations. Sixteen catalog entries remain at explicit non-backend execution
sites and are not counted as missing kernel executors.

The dtype-set blocker is closed. `embedding` and `cross_entropy_loss` were the
only two operations whose operands differ in dtype by construction, and both
are migrated on the union technique described in the table above — a row
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
5. **Delete the broad family rows and the grouped `Execute<MatMulSpec>`
   adapters**, then delete the compatibility adapter in `cpu::canonical` and the
   `the_migration_is_recorded_as_incomplete` test, which is written to fail once
   the catalog is fully migrated so that the completion claim has to be a
   deliberate edit.
