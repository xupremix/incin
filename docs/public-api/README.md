# Reviewed public API baselines

The CPU facade baseline in `incin-cpu.txt` is produced by `cargo-public-api`
with blanket, auto-trait, and auto-derived implementations omitted. It is a
reviewed snapshot of the symbols a normal CPU consumer can observe through
`incin`.

Run `python3 tools/check-public-api-baseline.py` after changing the facade.
An addition, removal, or signature change fails the check until the public
API change is reviewed and the snapshot is updated in the same commit.

The baseline intentionally covers the facade first. `incin-core` and
`incin-backends` expose separate authoring and backend contracts; their
explicit namespaces and architecture checks remain the review boundary while
their public surface is being reduced incrementally.
