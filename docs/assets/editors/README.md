# Editor screenshot evidence

These images come from live editor sessions connected to the locally built
`incin-lsp` and a real rust-analyzer process. They are product evidence, not
mockups.

Capture them in an isolated profile and a disposable workspace. Disable chat,
AI assistants, telemetry, accounts, notifications, and unrelated extensions.
Before committing an image, inspect it at full resolution for usernames, home
directories, repository remotes, tokens, account avatars, machine names,
unrelated files, and AI-assistant UI. Strip PNG metadata after the visual
review. The checker also rejects PNG `caBX` content-credential manifests, so
the committed files cannot carry C2PA provenance metadata.

The copies under `docs/book/src/assets/editors/` must be byte-for-byte equal to
the files in this directory. `tools/check-docs.py` verifies that equality, PNG
checksums, minimum legibility, and the absence of text and EXIF metadata. Pixel
content still requires the full-resolution manual review described above.
