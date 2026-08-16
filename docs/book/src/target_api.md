# Target-first construction

Incin's application-facing allocation API starts with a target. A target such
as `Cpu` selects the backend and device, while `shape!` supplies static proof
or runtime dimensions. This is part of the normal API and needs no feature
flag.

```rust,no_run
use incin::prelude::*;

let x = Cpu.zeros(shape![2, 3])?;
let batch = 4;
let y = Cpu.zeros(shape![batch, 3])?;
let z = Cpu.zeros([batch, 3])?;
# Ok::<(), incin::Error>(())
```

Static literals and explicit const paths preserve compile-time dimensions:

```rust,no_run
use incin::prelude::*;

const FEATURES: usize = 128;
let weights = Cpu.zeros(shape![const FEATURES, const FEATURES])?;
# Ok::<(), incin::Error>(())
```

The typed constructor remains available when code intentionally fixes the
backend type:

```rust,no_run
use incin::prelude::*;

type Backend = DefaultBackend;
let tensor = Tensor::<s![2, 2], Backend>::ones(())?;
# Ok::<(), incin::Error>(())
```

Arithmetic operators return tensors directly and include operation context in
their panic message when a runtime broadcast or backend check fails. Use the
`try_add`, `try_sub`, `try_mul`, `try_div`, and `try_neg` methods when the
failure must remain a recoverable `Result`.
