# FND-004 test-results

Two generations of logs are present. Both are kept on purpose.

## `*-gate.txt` — authoritative

Produced by the FND-004 acceptance gate on the checkout that became the FND-004
commit. These are the logs the `summary.md` verdicts link to. `commands.log`
records the exact command, working directory, commit hash, start/end timestamp
and exit code for each.

The gate was run twice with this suffix: once on the pre-commit worktree
(`stage: worktree-pre-commit`) and once on the committed hash
(`stage: committed`), so no criterion is proved only against an unspecified
working-tree diff.

## `*-final.txt`, `*-rerun.txt`, `*-final2.txt` — superseded, retained

Produced by an **earlier FND-004 attempt that did not pass its gate**. They are
retained as an honest record of the failed attempt and must not be read as
evidence for the current result. Notably:

- `workspace-tests-final.txt` and `workspace-tests-rerun.txt` record failing
  workspace runs.
- `operation-surface-inventory.txt` is zero-length — the inventory that attempt
  claimed to produce was never generated. It is superseded by
  `../operation-inventory.md` and the `operation_inventory` test.
- `backend-metal-check-final.txt` records the `E0425` Metal compilation failure
  that is now fixed.
- `fmt-workspace-final.txt` records formatter drift, as does the current
  `fmt-workspace-gate.txt`; that drift is pre-existing and out of scope.

Where a `*-final.txt` and a `*-gate.txt` disagree, the `*-gate.txt` file is
current.
