# FND-005 test-results

Two runs of the same acceptance gate are present. Both are kept.

## `*-committed.txt` - authoritative

Produced on the FND-005 commit itself (`stage: committed` in `commands.log`), so
no criterion is proved only against an unspecified working-tree diff. These are
the logs `summary.md` links to.

## `*.txt` (no suffix) - pre-commit worktree

Produced on the worktree that became that commit (`stage: worktree-pre-commit`).
Retained so the two can be compared. Where they disagree, the `-committed.txt`
file is current; in this task they agree.

## `fmt-workspace*.txt`

Both record exit `1`. That is expected and is the one BLOCKED criterion: the
drift is pre-existing and outside the FND-005 diff. `known-limitations.md`
records the file count, the proof that no FND-005 file is in the drift set, and
a correction to the count FND-004 recorded for its own commit.
