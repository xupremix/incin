# Model trust

How to reason about whether a model file is safe to load, and what Incin
does and does not verify.

## What Incin loads

- **Incin-own snapshots** (`.safetensors` carrying the `incin.format.version`
  envelope) are loaded through the strict path: the format key must match,
  every leaf is staged against a live variable slot, and a failed commit
  rolls back all prior assignments before any caller-visible state changes.
- **Foreign snapshots** (safetensors written by other frameworks, loaded via
  `load_foreign_safetensors_snapshot` or the Hub client) are validated for
  dtype, shape, and byte length at the boundary. They carry no incin format
  key by definition, so no claim about their producer is implied.
- **ONNX models** go through the generated protobuf decoder with resource
  bounds. Initializer loading supports the documented subset; unknown or
  unsupported constructs are refused with typed errors rather than
  approximated.

## What Incin refuses

- PyTorch `.pt` / `.pth` files: these are pickle archives and execute code on
  load. Incin has no pickle reader anywhere in its dependency tree and does
  not accept this format. Convert to safetensors first; conversion moves the
  trust decision to a tool you chose deliberately, once, instead of into
  every load call.
- NumPy object arrays: object dtype requires executing arbitrary Python
  reconstruction logic. Incin reads numeric arrays only, through its own
  bounded parsers.
- Any file whose header claims sizes beyond `ResourceLimits` bounds: refused
  before allocation rather than trusted.

## What "loads successfully" means

A successful load proves the bytes were structurally valid and within
bounds. It does not prove:

- the weights do what their name claims;
- the producer did not publish a backdoored or trojaned checkpoint;
- the architecture metadata matches what you intended to run.

Those are provenance questions that require out-of-band verification of the
source repository (commit pinning, hash comparison against a reference you
trust). Treat model weights from unreviewed sources like untrusted input to
your training loop: gate their outputs, watch loss behavior on known data,
and prefer pinned revisions over moving tags when reproducing results.

## Practical checklist

1. Pin the exact revision (`revision = "<commit-sha>"`) rather than `main`.
2. Record the file's hash at first use if the artifact matters.
3. Load foreign snapshots with the strictest surface that accepts them, and
   let validation failures stop the run - they are typed errors, not
   warnings.
4. Keep Hub credentials in `INCIN_HUB_TOKEN`, not in scripts; the client
   reads it from the environment and never logs it.
