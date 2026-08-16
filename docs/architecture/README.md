# Repository architecture map

`incin-architecture.json` is the source specification for the repository map.
`incin-architecture.html` is the self-contained explorable rendering. The map
captures the supported facade, core contracts, backend implementations,
authoring boundary, and documentation delivery path at the pinned revision in
the JSON evidence block.

Regenerate and validate it with:

```bash
node /home/xupremix/.agents/skills/archify/bin/archify.mjs validate \
  architecture docs/architecture/incin-architecture.json \
  --repo-root . --quality showcase --json
node /home/xupremix/.agents/skills/archify/bin/archify.mjs deliver \
  architecture docs/architecture/incin-architecture.json \
  docs/architecture/incin-architecture.html --repo-root . \
  --quality showcase --json
```

The visual-check screenshots and receipts are temporary review output and are
not part of the source artifact.
