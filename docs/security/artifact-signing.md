# Artifact signing

Status of integrity and authenticity for compiled-plan artifacts, and what
remains open for 0.1.0.

## Current state

Compiled-plan snapshots (`incin::experimental::compiled`) are framed with:

- a binary magic marker (`ARTIFACT_MAGIC`) so truncated or foreign files are
  rejected before decoding;
- an `ArtifactVersion` (format version plus caller-supplied
  major/minor/patch) with a documented local compatibility policy - patch is
  ignored, format must match exactly;
- an Adler-32 checksum over the serialized plan bytes, verified on load.

That is **integrity against corruption, not authenticity**. Adler-32 is not
a cryptographic hash: it detects accidental damage and truncation. It does
not detect deliberate tampering, and nothing in the format signs an
artifact or binds it to a publisher.

## What this means in practice

- An artifact produced by your own build and stored on media you control:
  framing and checksum are adequate; a load failure means the file is
  damaged or from a different format version.
- An artifact received from another party: treat it as untrusted data.
  Verify it out of band (hash comparison over a channel you trust) before
  loading. The loader's validation bounds decode but cannot tell you who
  produced the bytes.
- Artifacts are explicitly **not** a deployment format or portable ABI; they
  are preview snapshots of one process's plan (`docs/PROJECT_STATUS.md`).
  Distribution scenarios are outside what the format claims today.

## Road to signed artifacts

The remediation plan requires artifact authenticity before artifacts can be
recommended for exchange. The intended shape, deliberately not implemented
yet so the format does not churn twice:

1. Add a signatures section to `ArtifactHeader` carrying detached signatures
   over the canonical plan bytes (not the JSON envelope), so encoding
   details cannot break signature validity.
2. Use a cryptographic hash (SHA-256 class) as the digest input; keep
   Adler-32 only as a fast pre-check if measurement shows it earns its
   place.
3. Define key distribution separately from Incin: the framework verifies
   against keys the embedder supplies; it does not fetch trust roots.
4. Extend the conformance tests with tamper, truncation, and wrong-key
   negatives alongside the existing checksum negatives.

Until that lands, the honest summary is the first section above: corruption
detection yes, provenance no.
