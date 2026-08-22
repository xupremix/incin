# Threat model

This document states what Incin 0.1.0 defends against, what it explicitly
does not, and where the trust boundaries sit. It describes the shipped
surface only; preview features carry the caveats in
[`docs/PROJECT_STATUS.md`](../PROJECT_STATUS.md) and are excluded from any
stronger claim made here.

## Assets and adversaries

The assets are process memory integrity, filesystem integrity of the cache
and working directories, and network credentials supplied by the user. The
adversaries are:

- a malformed or hostile model, dataset, or checkpoint file placed where a
  loader will read it;
- a hostile repository or transfer endpoint reached through a download or
  Hub call;
- an ordinary local process failure or malformed runtime input (shapes,
  dtypes, indices, conversions).

Incin does not defend against an attacker who can already run code in the
process, replace the binary or its dependencies, or read files the user can
read. Supply-chain controls are CI-level (`deny.toml`, pinned actions,
cargo-deny) rather than runtime guarantees.

## Trust boundaries

| Boundary | Entering data | Handling |
| --- | --- | --- |
| ONNX protobuf import | bytes from `import_model!` / initializer loading | checked-in generated decoder; `ResourceLimits` bounds; unknown initializers refused rather than guessed |
| State and checkpoint load | `.safetensors` snapshots with the incin envelope | versioned format key, transactional staging with rollback before live state changes |
| Foreign snapshot load | third-party-produced `.safetensors` | no incin envelope required; dtype/shape validation at the boundary |
| Dataset fetch | HTTP responses and gzip streams | relative-filename validation, temporary-file rename, bounded parse headers |
| Hub download | files from a configured Hub endpoint | resolved through the Hub client's cache; filenames are repo-controlled, not caller-assembled paths |
| Compiled-plan artifacts | local snapshot files | magic, format version, and Adler-32 framing before decode |
| Rendezvous and collectives | TCP peers named by the launcher | see [`distributed-security.md`](distributed-security.md); untrusted unless the launcher already authenticated every rank |

## Invariants enforced before unsafe code

Every path above ends in typed errors (`docs/ERROR_CONTRACT.md`) before it
can reach one of the seven invariant families in
[`unsafe-ledger.md`](unsafe-ledger.md). The mechanical guarantees:

- production `unsafe` is limited to ledger-listed files;
  `tools/check-unsafe-ledger.py` fails on any new source site;
- production panic sites are enumerated in
  `audit-evidence/FND-003/production-panic-sites.json`;
  `tools/check-panic-audit.py` fails on drift;
- recoverable invalid input returns a typed error instead of aborting;
  operator panics are a documented convenience boundary with fixed text.

## Known gaps for 0.1.0

Stated plainly so nobody assumes otherwise:

- compiled-artifact integrity is a checksum, not authenticity; nothing signs
  artifacts yet ([`artifact-signing.md`](artifact-signing.md));
- distributed transport is plaintext without peer authentication
  ([`distributed-security.md`](distributed-security.md));
- hardware-specific execution paths (CUDA, NCCL, NEON, WASM) have compile
  checks and invariant audits but not continuous dynamic sanitizer coverage on
  this infrastructure;
- fuzzing coverage for parsers is planned (#48), not present; parser bounds
  today are the static `ResourceLimits` checks cited above.
