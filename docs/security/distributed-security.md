# Distributed security

What the distributed preview protects against, and the transport guarantees
it actually makes for 0.1.0. The short version: correctness is engineered,
confidentiality and peer authentication are not implemented yet.

## What exists today

- **Planning is local.** Meshes, placement rules, collective plans, and
  preflight agreement are computed per rank from topology facts; no user
  data crosses a network to build them.
- **The rendezvous** (`crates/incin-core/src/dist/context/bootstrap.rs`)
  binds or connects plain TCP sockets, exchanges a handshake that fixes rank
  order and world size, and produces a live context.
- **NCCL bootstrap** hands the store-derived addresses to NCCL over the same
  unauthenticated channel.
- **Reference transports** used by tests are in-memory and never leave the
  process.

## Guarantees made

- Plan preflight rejects divergent collectives before any transport launch:
  ranks must agree on mesh identity, cardinality, collective count, and plan
  hash, so a misconfigured rank fails loudly instead of silently corrupting
  a training run.
- Sequence tokens and explicit dependency ordering prevent reordering and
  replay *within one agreed plan execution*.

## Guarantees not made

Stated plainly:

1. **No peer authentication.** Any process that can reach the rendezvous
   address can join the handshake. There are no tokens, certificates, or
   shared secrets at any layer (this is finding SEC-009 in the remediation
   register).
2. **No confidentiality.** Handshake metadata and tensor traffic are
   plaintext TCP. Anyone on the network path can observe both.
3. **No integrity beyond TCP's checksum.** A malicious on-path actor can
   modify traffic; nothing signs frames.
4. **Telemetry sockets are local-only by design**, but they are not part of
   the distributed control plane and make no confidentiality claim either;
   treat run directories as readable by anything that can read your user's
   files.

## Deployment guidance

Until authenticated transport lands:

- Run distributed jobs only on networks you control: a single host, an
  isolated cluster interconnect, or a trusted VPC - never the public
  internet or a shared network where co-tenants can bind or connect.
- Treat the launcher as the root of trust: it names every endpoint, so
  compromise of the launcher compromises the job regardless of what the
  transport verified.
- Do not pass secrets through distributed environment variables to ranks you
  cannot account for; each rank inherits the launcher's environment.

## Road to authenticated transport

1. Shared-secret or PSK handshake bound into the plan hash at preflight, so
   a mismatched credential fails before any collective runs.
2. TLS or Noise-class encrypted framing for the rendezvous and control
   channel; tensor transport follows once control is trusted.
3. Capability-scoped join tokens minted per job by the launcher with bounded
   lifetime, so a leaked address is not a standing admission ticket.
4. Negative conformance tests: wrong-secret ranks, late joiners after
   preflight, and replayed handshakes must all fail with typed errors.

Each step changes the wire format, so they land together behind the
existing `distributed-*` feature gates rather than incrementally on the
default path.
