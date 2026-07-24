//! Pure JSON rewriting for the two LSP message kinds the proxy humanizes.
//! Kept free of any I/O so it can be unit-tested without spawning a process.

use kindle_diagnostics::{
    humanize_diagnostic, humanize_inlay_label, humanize_type_signature, strip_path_qualifiers,
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

/// Tracks JSON-RPC request ids sent to `textDocument/inlayHint` and
/// `textDocument/diagnostic`, so the corresponding response — which carries
/// no `method` of its own, only the matching `id` — can later be recognized
/// as one when it comes back from rust-analyzer.
#[derive(Default)]
pub struct PendingRequests {
    inlay_hint_ids: HashSet<String>,
    hover_ids: HashSet<String>,
    /// Pull-diagnostic request ids, mapped to the document URI from the
    /// request's `params.textDocument.uri` — the response itself carries no
    /// URI (unlike `publishDiagnostics`, which has one in `params`), but
    /// `relatedInformation` locations still need one.
    diagnostic_pull_ids: HashMap<String, Value>,
}

impl PendingRequests {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inspects a message flowing from the editor to rust-analyzer; records
    /// its id if it is a `textDocument/inlayHint`, `textDocument/hover`, or
    /// `textDocument/diagnostic` request (the pull-diagnostics model Neovim
    /// 0.10+ and other clients use automatically whenever the server
    /// advertises `diagnosticProvider`, in place of the older
    /// `textDocument/publishDiagnostics` push).
    pub fn observe_outgoing_to_server(&mut self, msg: &Value) {
        let Some(method) = msg.get("method").and_then(Value::as_str) else {
            return;
        };
        let Some(id) = msg.get("id") else { return };
        match method {
            "textDocument/inlayHint" => {
                self.inlay_hint_ids.insert(id_key(id));
            }
            "textDocument/hover" => {
                self.hover_ids.insert(id_key(id));
            }
            "textDocument/diagnostic" => {
                let uri = msg
                    .pointer("/params/textDocument/uri")
                    .cloned()
                    .unwrap_or(Value::Null);
                self.diagnostic_pull_ids.insert(id_key(id), uri);
            }
            _ => {}
        }
    }

    /// Returns `true` (and forgets the id) if `msg` is the response to a
    /// previously observed `textDocument/inlayHint` request.
    fn take_if_inlay_hint_response(&mut self, msg: &Value) -> bool {
        if msg.get("method").is_some() {
            return false; // requests/notifications are never responses
        }
        match msg.get("id") {
            Some(id) => self.inlay_hint_ids.remove(&id_key(id)),
            None => false,
        }
    }

    /// Returns `true` (and forgets the id) if `msg` is the response to a
    /// previously observed `textDocument/hover` request.
    fn take_if_hover_response(&mut self, msg: &Value) -> bool {
        if msg.get("method").is_some() {
            return false; // requests/notifications are never responses
        }
        match msg.get("id") {
            Some(id) => self.hover_ids.remove(&id_key(id)),
            None => false,
        }
    }

    /// Returns (and forgets) the request's document URI if `msg` is the
    /// response to a previously observed `textDocument/diagnostic` request.
    fn take_if_diagnostic_pull_response(&mut self, msg: &Value) -> Option<Value> {
        if msg.get("method").is_some() {
            return None; // requests/notifications are never responses
        }
        let id = id_key(msg.get("id")?);
        self.diagnostic_pull_ids.remove(&id)
    }
}

/// JSON-RPC ids are numbers or strings; stringify either uniformly as a
/// `HashSet` key (fine since it's only ever compared to itself).
fn id_key(id: &Value) -> String {
    id.to_string()
}

/// Rewrites one message flowing from rust-analyzer to the editor. Returns
/// `None` if the message needs no rewriting — the caller should then forward
/// the original bytes verbatim rather than re-serializing.
pub fn rewrite_incoming_from_server(
    msg: &Value,
    pending: &mut PendingRequests,
    hints_enabled: bool,
    shorten_backend: bool,
) -> Option<Value> {
    if msg.get("method").and_then(Value::as_str) == Some("textDocument/publishDiagnostics") {
        let uri = msg.pointer("/params/uri").cloned().unwrap_or(Value::Null);
        let mut msg = msg.clone();
        if let Some(diagnostics) = msg
            .pointer_mut("/params/diagnostics")
            .and_then(Value::as_array_mut)
        {
            humanize_diagnostic_list(diagnostics, &uri);
        }
        return Some(msg);
    }
    if let Some(uri) = pending.take_if_diagnostic_pull_response(msg) {
        return rewrite_diagnostic_pull_response(msg, &uri);
    }
    if hints_enabled && pending.take_if_inlay_hint_response(msg) {
        return Some(rewrite_inlay_hint_response(msg, shorten_backend));
    }
    if hints_enabled && pending.take_if_hover_response(msg) {
        return Some(rewrite_hover_response(msg, shorten_backend));
    }
    None
}

/// Humanizes every diagnostic's `message` in place and, for any that contain
/// typenum content, appends `relatedInformation` decimal-mapping hints.
/// Shared by both the `textDocument/publishDiagnostics` push notification and
/// the `textDocument/diagnostic` pull response — the `Diagnostic` shape
/// inside each is identical, only where the list and its URI live differs.
fn humanize_diagnostic_list(diagnostics: &mut [Value], uri: &Value) {
    for diagnostic in diagnostics.iter_mut() {
        let Some(message) = diagnostic
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            continue;
        };
        let translated = humanize_diagnostic(&message);
        diagnostic["message"] = json!(translated.text);

        if translated.hints.is_empty() {
            continue;
        }
        let range = diagnostic.get("range").cloned().unwrap_or(Value::Null);
        if let Some(obj) = diagnostic.as_object_mut() {
            let related = obj.entry("relatedInformation").or_insert_with(|| json!([]));
            if let Some(related) = related.as_array_mut() {
                for (decimal, original) in &translated.hints {
                    related.push(json!({
                        "location": { "uri": uri, "range": range },
                        "message": format!("{decimal} <= {original}"),
                    }));
                }
            }
        }
    }
}

/// Rewrites a `textDocument/diagnostic` pull response. Per LSP 3.17, `result`
/// is a `DocumentDiagnosticReport`: either `{"kind": "full", "items": [...]}`
/// (rewrite each item) or `{"kind": "unchanged", ...}` (no items — the client
/// is told to reuse what it already has, which was already humanized on
/// first delivery, so there's nothing to rewrite here).
fn rewrite_diagnostic_pull_response(msg: &Value, uri: &Value) -> Option<Value> {
    if msg.pointer("/result/kind").and_then(Value::as_str) != Some("full") {
        return None;
    }
    let mut msg = msg.clone();
    if let Some(items) = msg
        .pointer_mut("/result/items")
        .and_then(Value::as_array_mut)
    {
        humanize_diagnostic_list(items, uri);
    }
    Some(msg)
}

fn rewrite_inlay_hint_response(msg: &Value, shorten_backend: bool) -> Value {
    let mut msg = msg.clone();
    if let Some(hints) = msg.get_mut("result").and_then(Value::as_array_mut) {
        for hint in hints.iter_mut() {
            // rust-analyzer truncates deeply-nested generics (typenum shapes
            // routinely qualify) in `label` with a `…` ellipsis, discarding
            // the bits we need to humanize — but it always includes the
            // complete, fully-path-qualified type in `textEdits[0].newText`
            // (used for "insert the full type" instead of the ellipsis).
            // Prefer that as the source of truth whenever it's present.
            let from_text_edit = hint
                .pointer("/textEdits/0/newText")
                .and_then(Value::as_str)
                .map(|full| humanize_inlay_label(&strip_path_qualifiers(full), shorten_backend));
            match from_text_edit {
                Some(rewritten) => hint["label"] = json!(rewritten),
                None => {
                    if let Some(label) = hint.get_mut("label") {
                        rewrite_label_value(label, shorten_backend);
                    }
                }
            }
        }
    }
    msg
}

/// Rewrites an inlay-hint `label` or a hover `contents` value in place.
/// Per the LSP spec both can take any of several shapes: a plain string; an
/// object carrying the text under a `value` key (`InlayHintLabelPart`, or
/// hover's `MarkupContent`/deprecated `MarkedString`); or an array of either
/// (hover's deprecated `MarkedString[]`) — recursing into arrays covers that
/// last case for free.
fn rewrite_label_value(label: &mut Value, shorten_backend: bool) {
    match label {
        Value::String(s) => {
            *s = humanize_inlay_label(s, shorten_backend);
        }
        Value::Object(obj) => {
            if let Some(Value::String(s)) = obj.get_mut("value") {
                *s = humanize_inlay_label(s, shorten_backend);
            }
        }
        Value::Array(parts) => {
            for part in parts.iter_mut() {
                rewrite_label_value(part, shorten_backend);
            }
        }
        _ => {}
    }
}

/// Rewrites a `textDocument/hover` response's `result.contents`, which takes
/// the same shapes `rewrite_label_value` handles for inlay hints. Unlike
/// inlay hints, rust-analyzer doesn't truncate hover text with an ellipsis,
/// so there's no `textEdits`-style fallback needed here — but hover *does*
/// have room (unlike an inlay hint's cramped ghost text) for a legend
/// mapping each humanized number back to its raw typenum expression, so it
/// gets one appended, mirroring what diagnostics already show via
/// `relatedInformation`.
fn rewrite_hover_response(msg: &Value, shorten_backend: bool) -> Value {
    let mut msg = msg.clone();
    if let Some(contents) = msg.pointer_mut("/result/contents") {
        rewrite_hover_contents(contents, shorten_backend);
    }
    msg
}

fn rewrite_hover_contents(contents: &mut Value, shorten_backend: bool) {
    match contents {
        Value::String(s) => *s = humanize_with_legend(s, shorten_backend),
        Value::Object(obj) => {
            if let Some(Value::String(s)) = obj.get_mut("value") {
                *s = humanize_with_legend(s, shorten_backend);
            }
        }
        Value::Array(parts) => {
            for part in parts.iter_mut() {
                rewrite_hover_contents(part, shorten_backend);
            }
        }
        _ => {}
    }
}

/// Humanizes a hover type signature and, if any typenum content was found,
/// appends a legend mapping each decimal back to its raw typenum expression.
fn humanize_with_legend(text: &str, shorten_backend: bool) -> String {
    let translated = humanize_type_signature(text, shorten_backend);
    if translated.hints.is_empty() {
        return translated.text;
    }
    let mut out = translated.text;
    out.push_str("\n\n\u{1F4A1} Typenum Translation Hints:");
    for (decimal, original) in &translated.hints {
        out.push_str(&format!("\n  \u{2022} {decimal} <= {original}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrelated_notification_is_not_rewritten() {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "window/logMessage",
            "params": { "type": 3, "message": "some log line" }
        });
        let mut pending = PendingRequests::new();
        assert!(rewrite_incoming_from_server(&msg, &mut pending, true, false).is_none());
    }

    #[test]
    fn publish_diagnostics_message_is_humanized_and_hints_appended() {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": "file:///model.rs",
                "diagnostics": [{
                    "range": {"start": {"line": 10, "character": 14}, "end": {"line": 10, "character": 21}},
                    "severity": 1,
                    "message": "Cannot reshape: source has UInt<UInt<UInt<UTerm, B1>, B1>, B0> elements but the target shape has UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0> elements"
                }]
            }
        });
        let mut pending = PendingRequests::new();
        let rewritten = rewrite_incoming_from_server(&msg, &mut pending, true, false).unwrap();

        let diag = &rewritten["params"]["diagnostics"][0];
        assert_eq!(
            diag["message"],
            "Cannot reshape: source has 6 elements but the target shape has 8 elements"
        );
        let related = diag["relatedInformation"].as_array().unwrap();
        assert_eq!(related.len(), 2);
        assert_eq!(
            related[0]["message"],
            "6 <= UInt<UInt<UInt<UTerm, B1>, B1>, B0>"
        );
        assert_eq!(related[0]["location"]["uri"], "file:///model.rs");
        assert_eq!(
            related[1]["message"],
            "8 <= UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0>"
        );
    }

    #[test]
    fn diagnostic_with_no_typenum_content_gets_no_related_information() {
        let msg = json!({
            "method": "textDocument/publishDiagnostics",
            "params": { "uri": "file:///x.rs", "diagnostics": [{"message": "unused variable: `x`"}] }
        });
        let mut pending = PendingRequests::new();
        let rewritten = rewrite_incoming_from_server(&msg, &mut pending, true, false).unwrap();
        let diag = &rewritten["params"]["diagnostics"][0];
        assert_eq!(diag["message"], "unused variable: `x`");
        assert!(diag.get("relatedInformation").is_none());
    }

    /// Regression test for the real root cause of "diagnostics never show
    /// humanized text": Neovim (and other clients) automatically switch to
    /// the LSP 3.17 pull-diagnostics model — `textDocument/diagnostic`
    /// request/response — instead of the push-based `publishDiagnostics`
    /// notification whenever the server advertises `diagnosticProvider`,
    /// which rust-analyzer does. The proxy only watched for the push
    /// notification, so every diagnostic silently bypassed humanization.
    #[test]
    fn diagnostic_pull_response_full_report_is_humanized() {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "textDocument/diagnostic",
            "params": { "textDocument": { "uri": "file:///model.rs" } }
        });
        let response = json!({
            "jsonrpc": "2.0",
            "id": 42,
            "result": {
                "kind": "full",
                "resultId": "1",
                "items": [{
                    "range": {"start": {"line": 10, "character": 14}, "end": {"line": 10, "character": 21}},
                    "severity": 1,
                    "message": "Cannot reshape: source has UInt<UInt<UInt<UTerm, B1>, B1>, B0> elements but the target shape has UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0> elements"
                }]
            }
        });
        let mut pending = PendingRequests::new();
        // Before the request is observed, an unrelated response with the
        // same id shape must not be mistaken for a diagnostic pull response.
        assert!(rewrite_incoming_from_server(&response, &mut pending, true, false).is_none());

        pending.observe_outgoing_to_server(&request);
        let rewritten = rewrite_incoming_from_server(&response, &mut pending, true, false).unwrap();
        let item = &rewritten["result"]["items"][0];
        assert_eq!(
            item["message"],
            "Cannot reshape: source has 6 elements but the target shape has 8 elements"
        );
        let related = item["relatedInformation"].as_array().unwrap();
        assert_eq!(related[0]["location"]["uri"], "file:///model.rs");
        assert_eq!(
            related[0]["message"],
            "6 <= UInt<UInt<UInt<UTerm, B1>, B1>, B0>"
        );

        // The id is consumed on first use.
        assert!(rewrite_incoming_from_server(&response, &mut pending, true, false).is_none());
    }

    #[test]
    fn diagnostic_pull_response_unchanged_report_has_no_items_to_rewrite() {
        let request = json!({
            "id": 1,
            "method": "textDocument/diagnostic",
            "params": { "textDocument": { "uri": "file:///model.rs" } }
        });
        let response = json!({
            "id": 1,
            "result": { "kind": "unchanged", "resultId": "1" }
        });
        let mut pending = PendingRequests::new();
        pending.observe_outgoing_to_server(&request);
        assert!(rewrite_incoming_from_server(&response, &mut pending, true, false).is_none());
    }

    #[test]
    fn inlay_hint_response_is_only_rewritten_after_a_matching_request_was_observed() {
        let request =
            json!({"jsonrpc": "2.0", "id": 7, "method": "textDocument/inlayHint", "params": {}});
        let response = json!({
            "jsonrpc": "2.0",
            "id": 7,
            "result": [{ "position": {"line": 3, "character": 1}, "label": "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<f32, Cpu>>" }]
        });

        let mut pending = PendingRequests::new();
        // Before the request is observed, a same-shaped response must NOT be
        // mistaken for an inlay-hint response (nothing to correlate it to yet).
        assert!(rewrite_incoming_from_server(&response, &mut pending, true, false).is_none());

        pending.observe_outgoing_to_server(&request);
        let rewritten = rewrite_incoming_from_server(&response, &mut pending, true, false).unwrap();
        assert_eq!(
            rewritten["result"][0]["label"],
            "Tensor<[2, 3], CpuBackendImpl<f32, Cpu>>"
        );

        // The id is consumed on first use — a second identical response
        // (e.g. from a stray duplicate) is no longer recognized.
        assert!(rewrite_incoming_from_server(&response, &mut pending, true, false).is_none());
    }

    #[test]
    fn inlay_hint_response_respects_shorten_backend_flag() {
        let request = json!({"id": 1, "method": "textDocument/inlayHint"});
        let response = json!({
            "id": 1,
            "result": [{ "label": "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<f32, Cpu>>" }]
        });
        let mut pending = PendingRequests::new();
        pending.observe_outgoing_to_server(&request);
        let rewritten = rewrite_incoming_from_server(&response, &mut pending, true, true).unwrap();
        assert_eq!(rewritten["result"][0]["label"], "Tensor<[2, 3]>");
    }

    #[test]
    fn hints_disabled_leaves_inlay_hint_response_unrewritten() {
        let request = json!({"id": 1, "method": "textDocument/inlayHint"});
        let response = json!({"id": 1, "result": [{ "label": "Tensor<(UInt<UTerm, B1>,), CpuBackendImpl<f32, Cpu>>" }]});
        let mut pending = PendingRequests::new();
        pending.observe_outgoing_to_server(&request);
        assert!(rewrite_incoming_from_server(&response, &mut pending, false, false).is_none());
    }

    /// Regression test for a real, reported case: rust-analyzer truncates a
    /// deeply-nested typenum shape's `label` with a `…` ellipsis (verified
    /// against a live rust-analyzer response for a `Tensor<[3, 5], ...>`
    /// hint), so the per-part label rewrite in `rewrite_label_value` can't
    /// recover it — the full type only survives in `textEdits[0].newText`,
    /// which must be preferred when present.
    #[test]
    fn inlay_hint_response_recovers_truncated_label_from_text_edit() {
        let request = json!({"id": 5, "method": "textDocument/inlayHint"});
        let response = json!({
            "id": 5,
            "result": [{
                "position": {"line": 18, "character": 10},
                "label": [
                    {"value": ": "},
                    {"value": "Tensor"},
                    {"value": "<("},
                    {"value": "UInt"},
                    {"value": "<"},
                    {"value": "UInt"},
                    {"value": "<"},
                    {"value": "UTerm"},
                    {"value": ", …>, …>, …), …, …>"}
                ],
                "kind": 1,
                "textEdits": [{
                    "range": {"start": {"line": 18, "character": 10}, "end": {"line": 18, "character": 10}},
                    "newText": "kindle::cpu::Tensor<(typenum::UInt<typenum::UInt<typenum::UTerm, typenum::B1>, typenum::B1>, typenum::UInt<typenum::UInt<typenum::UInt<typenum::UTerm, typenum::B1>, typenum::B0>, typenum::B1>), kindle::cpu::CpuBackendImpl, f32>"
                }]
            }]
        });
        let mut pending = PendingRequests::new();
        pending.observe_outgoing_to_server(&request);
        let rewritten = rewrite_incoming_from_server(&response, &mut pending, true, false).unwrap();
        assert_eq!(
            rewritten["result"][0]["label"],
            "Tensor<[3, 5], CpuBackendImpl, f32>"
        );
    }

    #[test]
    fn inlay_hint_label_parts_array_form_is_rewritten() {
        let request = json!({"id": 9, "method": "textDocument/inlayHint"});
        let response = json!({
            "id": 9,
            "result": [{ "label": [{"value": "Tensor<(UInt<UTerm, B1>,), CpuBackendImpl<f32, Cpu>>"}] }]
        });
        let mut pending = PendingRequests::new();
        pending.observe_outgoing_to_server(&request);
        let rewritten = rewrite_incoming_from_server(&response, &mut pending, true, false).unwrap();
        assert_eq!(
            rewritten["result"][0]["label"][0]["value"],
            "Tensor<[1], CpuBackendImpl<f32, Cpu>>"
        );
    }

    /// Regression test for a real, reported gap: `textDocument/hover` was
    /// never rewritten at all (only diagnostics and inlay hints were), so
    /// hovering a Kindle tensor showed the raw typenum type. Verified against
    /// a live rust-analyzer hover response for `let t2: Tensor<...>`, which
    /// wraps the type in a markdown code fence followed by a `size = ...`
    /// trailer — the rewrite must touch only the type, not the surrounding
    /// markdown (which itself contains unrelated parens, e.g. `(0x48)`), and
    /// (unlike inlay hints, which have no room for it) append the same
    /// typenum-translation legend diagnostics already show.
    #[test]
    fn hover_response_humanizes_markup_content_and_appends_legend() {
        let request = json!({"id": 3, "method": "textDocument/hover", "params": {}});
        let response = json!({
            "id": 3,
            "result": {
                "contents": {
                    "kind": "markdown",
                    "value": "\n```rust\nlet t2: Tensor<(UInt<UInt<UInt<UTerm, B1>, B0>, B0>, UInt<UInt<UInt<UTerm, B1>, B0>, B1>), CpuBackendImpl, f32>\n```\n\n---\n\nsize = 72 (0x48), align = 0x8, needs Drop"
                },
                "range": {"start": {"line": 11, "character": 8}, "end": {"line": 11, "character": 10}}
            }
        });
        let mut pending = PendingRequests::new();
        pending.observe_outgoing_to_server(&request);
        let rewritten = rewrite_incoming_from_server(&response, &mut pending, true, false).unwrap();
        assert_eq!(
            rewritten["result"]["contents"]["value"],
            "\n```rust\nlet t2: Tensor<[4, 5], CpuBackendImpl, f32>\n```\n\n---\n\nsize = 72 (0x48), align = 0x8, needs Drop\n\n\u{1F4A1} Typenum Translation Hints:\n  \u{2022} 4 <= UInt<UInt<UInt<UTerm, B1>, B0>, B0>\n  \u{2022} 5 <= UInt<UInt<UInt<UTerm, B1>, B0>, B1>"
        );
    }

    #[test]
    fn hover_response_not_rewritten_when_hints_disabled() {
        let request = json!({"id": 4, "method": "textDocument/hover", "params": {}});
        let response = json!({
            "id": 4,
            "result": { "contents": { "kind": "markdown", "value": "Tensor<(UInt<UTerm, B1>,), CpuBackendImpl>" } }
        });
        let mut pending = PendingRequests::new();
        pending.observe_outgoing_to_server(&request);
        assert!(rewrite_incoming_from_server(&response, &mut pending, false, false).is_none());
    }

    #[test]
    fn ordinary_response_without_a_tracked_id_is_not_rewritten() {
        let msg = json!({"id": 123, "result": {"capabilities": {}}});
        let mut pending = PendingRequests::new();
        assert!(rewrite_incoming_from_server(&msg, &mut pending, true, false).is_none());
    }
}
