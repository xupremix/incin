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
| The dispatch path | `crates/incin-core/src/exec/dispatch.rs` | `execute::<O, B>` is the *intended* single route from an operation to a kernel: validate the descriptor against real storage metadata, query the exact capability row per operand, then dispatch. It is finished; what is not finished is its adoption — see the note below | `B: Execute<Descriptor<O>>` is a compile-time bound, and there is no default method to fall through |
| The execution contract | `crates/incin-core/src/exec/request.rs`, `crates/incin-core/src/tensor/backend.rs` | `Execute<O>` carries an associated `Output`, so an operation returning a pair is expressible without a special case. `TensorHandle` carries checked metadata, so an executor never re-derives it | Type-checked at every call site |
| The capability declaration | `crates/incin-backends/src/capability.rs` | Groups are rule *shapes*, not operation families. Migrating an operation is one more name in an existing list, not a new group and a new arm in every consumer | One declaration feeds the capability rows, the legacy executors and the canonical executors |
| The completeness proof | `crates/incin-backends/src/cpu/canonical.rs` | A capability row advertised without an `Execute<Descriptor<op::X>>` behind it does not compile | `assert_every_advertised_row_executes!`, driven by the same declaration that generates the rows |
| The generated evidence | `docs/capabilities.md`, `docs/OPERATION_SEMANTICS.md`, `audit-evidence/FND-005/cpu-migration-status.md` | Every number in them is derived from the Rust source rather than written by hand | A test fails when the committed file and the regenerated one differ; `docs/README.md` lists the regeneration command for each |

> **Adoption, as distinct from design.** Calling this "the single production
> route" overstated it for a long time, and the wording above is corrected.
> Until the `target-api` prototype landed, **every** call site of
> `dispatch::execute` in the tree was test code; the stable tensor surface
> reached kernels through the nine operation-family supertraits instead, which
> `audit-evidence/FND-005/cpu-migration-status.md` has always said plainly.
> `incin_backends::target::TargetExt::zeros_canonical` is now the first
> non-test caller. Everything else still goes through the family traits.
> `docs/plan/UX-ARCHITECTURE-HANDOFF.md` has the counts, the reproduction
> command, and the remaining steps.

The shape of that table is the point. Each row is a decision made once and then
made unrepeatable, so the cost of migrating operation number 118 is the same as
the cost of migrating number 5 was, rather than growing with the count.

## What is deliberately not frozen

These are the surfaces the remaining work changes. Nothing here should be
treated as settled.

| Surface | Why it still moves |
|---|---|
| The per-operation executor bodies in `cpu/canonical.rs` | 158 of 161 migrated. The remaining 3 executable operations are recorded in `cpu-migration-status.md`; non-executable catalog entries are tracked separately |
| The nine operation-family supertraits on `Backend` | Removing them is FND-005's completion condition. It is source-breaking for every backend |
| The broad family capability rows | `Pointwise`, `Reduction`, `Reshape`, `MatMul`, `Conv2d`, `Pool2d`, `Storage`, `Fill`, `Random`, `Normalization`, `Broadcast` are deleted once nothing resolves through them |
| `CapabilityRule`'s single dtype set | It describes an operation, but `dispatch::execute` applies it to each operand in turn. An operation whose operands differ in dtype by construction cannot state the tight per-operand pair directly, and no longer needs to: `INDEX_AND_F32_DTYPES` states the *union* the row can honestly claim, the same trick `descriptor_min_rank` already used for rank, and the descriptor's own per-operand contract (already `TypedContract`/hand-cased in `validate`, not something this added) rejects the wrong combination before any capability query runs. Both operations that needed it — `embedding` and `cross_entropy_loss` — are migrated on that technique, so no struct change to `CapabilityRule` was needed or made |
| `CapabilityRule`'s single rank range | Same cause, same fix already in place: the range states the minimum over *all* operands, which is what `descriptor_min_rank` has always done and what `INDEX_AND_F32_DTYPES`'s rows now also do for rank |
| `Execute`'s reachable sites | Thirteen operations have an `ExecutionSite` the trait cannot carry: they mutate through an operand, produce storage on another backend, or act on autograd state. `ExecutionSite::blocking_reason` states which |

## Next steps, in dependency order

Each step is blocked by the one above it, and the reason is stated rather than
implied.

The migration is no longer the bottleneck. 158 of the 161 backend-executable
operations have an executor, and the remaining 3 are explicitly tracked rather
than omitted from the catalog.

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
   contract that can. Thirteen operations are not pending migrations at all,
   and counting them as such overstates the remaining work by roughly 30%.
4. **Remove the nine supertraits from `Backend`** and bound each stable tensor
   method by the capability it actually uses. This is the step that ends the
   dual architecture. It is no longer blocked by migration coverage: the
   remaining executable gaps are explicit and the stable data constructors use
   canonical dispatch.
5. **Delete the broad family rows and the grouped `Execute<MatMulSpec>`
   adapters**, then delete the compatibility adapter in `cpu::canonical` and the
   `the_migration_is_recorded_as_incomplete` test, which is written to fail once
   the catalog is fully migrated so that the completion claim has to be a
   deliberate edit.
