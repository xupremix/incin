# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected security vulnerability.
Use the repository's [private vulnerability report](https://github.com/xupremix/incin/security/advisories/new)
before sharing reproduction details. Include the affected commit, feature
flags, platform, and the smallest reproduction that demonstrates the impact.

Reports should describe whether the issue affects model data, arbitrary code
execution, memory safety, credential handling, or denial of service. Do not
include secrets or private user data in an initial report.

## Review boundary

Production `unsafe` code is inventoried in
[`docs/security/unsafe-ledger.md`](docs/security/unsafe-ledger.md). New
unsafe-bearing files must be added to that ledger and include a nearby
`SAFETY` explanation for non-obvious preconditions. The ledger checker runs in
CI and is part of the required validation for changes that touch unsafe code.

The project does not promise hardware-backed isolation for optional CUDA,
Metal, or WGPU execution. Applications should treat model files, custom
kernels, and external adapters as untrusted inputs until they have been
reviewed for their deployment environment.
