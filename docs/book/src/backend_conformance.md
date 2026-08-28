# Backend conformance

`docs/capabilities.md` opens by saying what it is: "A row here is a canonical
capability decision, not a claim about a machine." The conformance oracle is
what turns the row back into a claim about a machine and then checks it.

It lives in `crates/incin-backends/src/conformance/`, and it is driven by
`crates/incin-backends/tests/conformance_oracle.rs`.

## A row is a product

A `CapabilityRule` names an operation, a set of dtypes, a set of layouts, a
rank range, a training flag and a set of math modes. The rendered table shows
one line per operation. What the line actually promises is every combination:
eight dtypes times two layouts times two rank boundaries times two math modes
is thirty-two separate promises on one line, and a backend can keep thirty-one
of them.

The harness expands the row and poses each point:

```text
advertised tuples: 3149, executed: 2248, covered operations: 153
```

## What it checks

**The positive direction.** Every advertised tuple must execute. A tuple that
comes back refused is a capability defect, because the table is what users
read and the kernel is what they get.

**The negative direction.** A dtype the row does *not* list must be refused.
This half is asked of the executor rather than of the dispatcher.
`dispatch::execute` queries the capability registry before it calls anything,
so a tuple posed through it can only ever demonstrate that the dispatcher
works. The CPU executors re-check their own row from inside themselves
(`cpu/canonical/common.rs`), and that re-check is what the negative half reads.
A backend whose executors trusted the dispatcher instead would fail here, which
is the point.

It does **not** check values, and it does not check gradients. Values are
meaningless while the oracle and the subject are the same backend, and become
meaningful the moment a second backend runs through the same driver. Gradients
are a separate harness: the `Training` column is a claim about a derivative,
and checking a derivative means finite differences or a recorded reference,
not a second call to the same dispatcher.

## The guard is the positive loop

It is tempting to read the negative probe as the thing that catches an
overclaiming row. It is not. A row that advertises `i64` for a `bool`-only
kernel is caught because the *positive* loop poses the advertised tuple and the
descriptor rejects it. The negative probe catches the opposite defect: a kernel
that quietly runs something its table never promised.

Both matter, because in both cases the published table has stopped describing
the machine.

## A panic is never an answer

The negative probe inverts its verdicts: a refusal there is the contract
holding. A panic used to be swallowed by that inversion and read as a pass.

`Verdict::Panicked` is now a finding on **every** route. The error contract
says a bad invocation comes back as a value; an unwind past admission breaks it
as surely as one before admission does. Each call is wrapped in
`catch_unwind` so that one panicking tuple produces one finding rather than
ending the run.

This is what caught `dot` indexing into a rank-zero shape, and `rand`/`randn`
reaching an `unreachable!()` on a non-float dtype.

## A union row's floor is not the operation's floor

`dispatch::execute` checks every operand against **one** resolved row. A row
whose operands genuinely differ therefore has to state the *union*: the loosest
bound across all of them.

`conv2d` declares a rank floor of one. That one speaks for the bias vector.
The activation itself needs a channel axis and two spatial axes, so its floor
is three. The consequence is subtle and cost three real defects to find: the
activation's real floor sits in the **interior** of the declared range, which
is exactly where boundary enumeration does not look.

`boundary_ranks` therefore poses three points for such a row: the rule's floor,
the catalog's own `accepted_ranks` floor when the rule declares one below it,
and the top. For `conv2d` that is ranks one, three and four. Rank three is
where all three convolution kernels were panicking.

Four rows gain a third boundary today: the three convolutions and `addmm`.
`batch_norm` is a union row too, and its per-channel vectors are why its floor
is one, but the tighter floor its activation needs is enforced inside
`BatchNormAttributes::validate` rather than declared in `accepted_ranks`, so
the enumeration has nothing to read. Its rank-one tuple is reported as
`Unbuildable` with that as the reason, which is the honest answer: there is no
rank-one batch norm invocation to pose.

## Writing a fixture

A fixture says two things: how many operands to build (`Operands`) and what
each one carries (`Role`).

`Role` exists because a capability row applies one dtype set to every operand
in turn. A row can never state that operand zero is an integer index and
operand one is a float table; the honest thing it can say is the union of the
two, and `declarations.rs` says so at length for `INDEX_AND_F32_DTYPES` and
`F32_AND_BOOL`. Walking that union and handing every operand the same dtype
poses invocations the operation was never meant to accept, so the fixture
states the split the row cannot.

`Role` fixes a **shape** as well as a dtype. `ConvWeight { spatial }` builds
`[out, in / groups, ..unit kernel]` and reads the channel extent at the axis
`inference.rs` reads it at. `ChannelVector` reads axis one absolutely, because
`BatchNormAttributes::validate` does. `TrailingVector` is the last input
extent, which is what an RMS norm weight must equal.

One role states an agreement *between* operands rather than a shape per
operand. `Paired { rows, columns }` gives an operand the tuple's batch extents
followed by two named ones, so a matrix product is written as `[..batch, M, K]`
against `[..batch, K, N]` and the shared `K` is a number in the fixture rather
than a coincidence. It is the shape behind three operations at once: `matmul`,
`addmm` (which adds a `[..batch, M, N]` addend to the same product) and
attention (a query, key and value agreeing pairwise on two different extents).
Its strided form transposes the **last** two axes, not the first, or the batch
extents of one operand stop agreeing with the next.

The operand extents are a short ladder, `[2, 3, 2, 2]`, deliberately unequal:
a kernel that transposes when it should not, or reads an extent off the wrong
axis, produces the right answer on a square operand and the wrong one here.
For the same reason a convolution's output-channel count is five, which is on
no ladder and equal to no extent the harness builds.

## The limit worth knowing before you write a fixture

Every operand a fixture builds is still sized from **one** tuple, and the two
gaps that remain are the two places that is not enough.

`quantize`, `dequantize` and `quantized_matmul` need a buffer whose length is
its block encoding rather than its logical extent. The block size is a backend
detail the enumeration does not know, and inventing one in the operand builder
would hard-code a wrong constant to buy three rows.

`tensor_from_data` and `tensor_from_bytes` need bytes on the
`ExecutionRequest`, which every shim here poses as `None`. Describing the byte
length in the attributes is not enough; the bytes have to be supplied, and that
is a change to the shim signature rather than to `Role`.

## Coverage is a number, not a wall

An operation with no fixture is reported as `Coverage::Unfixtured` with the
reason it is outstanding, and it is counted. It is deliberately not a failure.
A harness that opens with a hundred red rows is a harness that gets marked
ignored, which is the same outcome as one that silently passes.

The floor lives in `tests/conformance_oracle.rs` and only moves up.
