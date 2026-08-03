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
| The dispatch path | `crates/incin-core/src/exec/dispatch.rs` | `execute::<O, B>` is the single production route from an operation to a kernel: validate the descriptor against real storage metadata, query the exact capability row per operand, then dispatch | `B: Execute<Descriptor<O>>` is a compile-time bound, and there is no default method to fall through |
| The execution contract | `crates/incin-core/src/exec/request.rs`, `crates/incin-core/src/tensor/backend.rs` | `Execute<O>` carries an associated `Output`, so an operation returning a pair is expressible without a special case. `TensorHandle` carries checked metadata, so an executor never re-derives it | Type-checked at every call site |
| The capability declaration | `crates/incin-backends/src/capability.rs` | Groups are rule *shapes*, not operation families. Migrating an operation is one more name in an existing list, not a new group and a new arm in every consumer | One declaration feeds the capability rows, the legacy executors and the canonical executors |
| The completeness proof | `crates/incin-backends/src/cpu/canonical.rs` | A capability row advertised without an `Execute<Descriptor<op::X>>` behind it does not compile | `assert_every_advertised_row_executes!`, driven by the same declaration that generates the rows |
| The generated evidence | `docs/capabilities.md`, `docs/OPERATION_SEMANTICS.md`, `audit-evidence/FND-005/cpu-migration-status.md` | Every number in them is derived from the Rust source rather than written by hand | A test fails when the committed file and the regenerated one differ; `docs/README.md` lists the regeneration command for each |

The shape of that table is the point. Each row is a decision made once and then
made unrepeatable, so the cost of migrating operation number 118 is the same as
the cost of migrating number 5 was, rather than growing with the count.

## What is deliberately not frozen

These are the surfaces the remaining work changes. Nothing here should be
treated as settled.

| Surface | Why it still moves |
|---|---|
| The per-operation executor bodies in `cpu/canonical.rs` | 7 backend-executable operations still have no canonical executor. Each is additive |
| The nine operation-family supertraits on `Backend` | Removing them is FND-005's completion condition. It is source-breaking for every backend |
| The broad family capability rows | `Pointwise`, `Reduction`, `Reshape`, `MatMul`, `Conv2d`, `Pool2d`, `Storage`, `Fill`, `Random`, `Normalization`, `Broadcast` are deleted once nothing resolves through them |
| `CapabilityRule`'s single dtype set | It describes an operation, but `dispatch::execute` applies it to each operand in turn. An operation whose operands differ in dtype by construction cannot be stated. This is what blocks `embedding` |
| `CapabilityRule`'s single rank range | Same cause. The range has to be the minimum over *all* operands, so it cannot constrain the primary one. See `descriptor_min_rank` |
| `Execute`'s reachable sites | Thirteen operations have an `ExecutionSite` the trait cannot carry: they mutate through an operand, produce storage on another backend, or act on autograd state. `ExecutionSite::blocking_reason` states which |

## Next steps, in dependency order

Each step is blocked by the one above it, and the reason is stated rather than
implied.

1. **Migrate the remaining 7 backend-executable operations.** Additive, and the
   count is generated, so progress cannot be overstated. The order that costs
   least is by rule shape: an operation joining an existing capability group is
   one name in a list, and one that needs a new group is a new arm in every
   consumer of the declaration.
2. **Give `CapabilityRule` per-operand dtype and rank sets.** This unblocks
   `embedding` and lets the convolution rows constrain their activation again
   instead of stating the minimum their bias forces. It is a change to a frozen
   foundation and should be done once, deliberately, rather than worked around
   per operation.
3. **Widen `Execute` to the sites it cannot reach**, or split them off into a
   contract that can. Until then, thirteen operations are not pending
   migrations, and counting them as such misstates the remaining work by 30%.
4. **Remove the nine supertraits from `Backend`** and bound each stable tensor
   method by the capability it actually uses. This is the step that ends the
   dual architecture, and it cannot start before step 1: a tensor method cannot
   depend on a capability that does not exist yet.
5. **Delete the broad family rows and the grouped `Execute<MatMulSpec>`
   adapters**, then delete the compatibility adapter in `cpu::canonical` and the
   `the_migration_is_recorded_as_incomplete` test, which is written to fail once
   the catalog is fully migrated so that the completion claim has to be a
   deliberate edit.
