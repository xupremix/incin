# Backend powerset partition status

Environment: Linux x86_64, `rustc 1.97.1`, `cargo 1.97.1`, `cargo-hack
0.6.45`; `/tmp` had approximately 72 GiB available before the attempt.

Exact command-list generation:

```text
CARGO_TARGET_DIR=/tmp/hnd004c-backend-final/target cargo hack check -p incin-backends --feature-powerset --all-targets --exclude-features external-candle --print-command-list
```

The generated list contains 8,212 commands and is archived as
`powerset-backends-command-list-current.txt`. A second exact retry archived
the same list as `powerset-backends-command-list-retry.txt`, divided it into
16 partitions, and used eight separate targets. That retry reached only the
initial dependency/backend compilation (70 commands across the active
partitions) before external termination after wall-time inspection. No
compiler error was observed in the saved output; the run exited by external
termination and is not a pass.

The first partial outputs are `powerset-backends-partition-00.txt` through
`powerset-backends-partition-03.txt`; retry outputs are
`powerset-backends-retry-00.txt` through `powerset-backends-retry-07.txt`.
The facade matrix was not started; its exact current command list previously
expanded to 36,608 commands.
