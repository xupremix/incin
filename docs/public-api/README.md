# Reviewed public API baselines

The CPU facade baseline in `incin-cpu.txt` is produced by `cargo-public-api`
with blanket, auto-trait, and auto-derived implementations omitted. It is a
reviewed snapshot of the symbols a normal CPU consumer can observe through
`incin`.

Run `python3 tools/check-public-api-baseline.py` after changing the facade.
An addition, removal, or signature change fails the check until the public
API change is reviewed and the snapshot is updated in the same commit.

`incin::experimental::compiled` is intentionally excluded from this reviewed
baseline. It is a feature-gated preview namespace with no compatibility
guarantee; its CPU reference evaluator and plan-inspection types are tracked by
the compiled-preview fixtures instead. A future declarative API inventory will
encode this exclusion mechanically.

The baseline intentionally covers the facade first. `incin-core` and
`incin-backends` expose separate authoring and backend contracts; their
explicit namespaces and architecture checks remain the review boundary while
their public surface is being reduced incrementally. Package boundaries are
checked independently by `tools/check-package.sh` for every top-level shipped
crate, so this API baseline does not stand in for archive validation.
