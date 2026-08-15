# Backend powerset partition status

Environment: Linux x86_64, `rustc 1.97.1`, `cargo 1.97.1`, `cargo-hack
0.6.45`; `/tmp` had approximately 72 GiB available before the attempt.

Exact command-list generation:

```text
CARGO_TARGET_DIR=/tmp/hnd004c-backend-final/target cargo hack check -p incin-backends --feature-powerset --all-targets --exclude-features external-candle --print-command-list
```

The generated list contains 8,212 commands and is archived as
`powerset-backends-command-list-current.txt`. The list was divided into four
partitions and executed against one shared Cargo target. The attempt was
stopped during initial compilation after resource/wall-time inspection. No
compiler error was observed in the saved output; the run exited by external
termination and is not a pass.

The partial outputs are `powerset-backends-partition-00.txt` through
`powerset-backends-partition-03.txt`. The facade matrix was not started; its
exact current command list previously expanded to 36,608 commands.
