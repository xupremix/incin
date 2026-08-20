//! `incin-lsp` - a transparent LSP proxy that sits between an editor and
//! rust-analyzer: it spawns rust-analyzer as a child process and rewrites
//! only two message kinds through `incin-diagnostics` before forwarding
//! them on -
//!
//! - `textDocument/publishDiagnostics` notifications: each diagnostic's
//!   `message` gets its typenum expressions replaced with decimals.
//! - `textDocument/inlayHint` responses: each hint's `label` gets the same
//!   treatment via [`incin_diagnostics::humanize_inlay_label`].
//!
//! Every other message (completion, goto-definition, formatting, ...) is
//! forwarded byte-for-byte unchanged - the proxy never re-serializes a
//! message it isn't rewriting, so it cannot introduce whitespace/key-order
//! drift on anything it doesn't touch.
//!
//! Deliberately does **not** depend on `tower-lsp`/`lsp-types`: modeling the
//! entire LSP surface would be an ongoing maintenance burden for a proxy
//! that only ever needs to understand two message shapes. Frames are
//! read/written as raw `Content-Length`-prefixed bytes ([`frame`]) and, only
//! when a rewrite might apply, inspected as [`serde_json::Value`]
//! ([`rewrite`]) - everything else passes through as opaque bytes.

pub mod config;
pub mod frame;
pub mod rewrite;
