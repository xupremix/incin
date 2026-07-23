//! Pure JSON rewriting for the two LSP message kinds the proxy humanizes.
//! Kept free of any I/O so it can be unit-tested without spawning a process.

use kindle_diagnostics::{humanize_diagnostic, humanize_inlay_label};
use serde_json::{Value, json};
use std::collections::HashSet;

/// Tracks JSON-RPC request ids sent to `textDocument/inlayHint`, so the
/// corresponding response — which carries no `method` of its own, only the
/// matching `id` — can later be recognized as one when it comes back from
/// rust-analyzer.
#[derive(Default)]
pub struct PendingRequests {
    inlay_hint_ids: HashSet<String>,
}

impl PendingRequests {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inspects a message flowing from the editor to rust-analyzer; records
    /// its id if it is a `textDocument/inlayHint` request.
    pub fn observe_outgoing_to_server(&mut self, msg: &Value) {
        if msg.get("method").and_then(Value::as_str) == Some("textDocument/inlayHint")
            && let Some(id) = msg.get("id")
        {
            self.inlay_hint_ids.insert(id_key(id));
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
        return Some(rewrite_publish_diagnostics(msg));
    }
    if hints_enabled && pending.take_if_inlay_hint_response(msg) {
        return Some(rewrite_inlay_hint_response(msg, shorten_backend));
    }
    None
}

fn rewrite_publish_diagnostics(msg: &Value) -> Value {
    let mut msg = msg.clone();
    let uri = msg.pointer("/params/uri").cloned().unwrap_or(Value::Null);
    if let Some(diagnostics) = msg
        .pointer_mut("/params/diagnostics")
        .and_then(Value::as_array_mut)
    {
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
    msg
}

fn rewrite_inlay_hint_response(msg: &Value, shorten_backend: bool) -> Value {
    let mut msg = msg.clone();
    if let Some(hints) = msg.get_mut("result").and_then(Value::as_array_mut) {
        for hint in hints.iter_mut() {
            if let Some(label) = hint.get_mut("label") {
                rewrite_label_value(label, shorten_backend);
            }
        }
    }
    msg
}

/// An inlay hint's `label` is either a plain string or an array of
/// `InlayHintLabelPart` objects (each with its own `value` string) — the LSP
/// spec allows both.
fn rewrite_label_value(label: &mut Value, shorten_backend: bool) {
    match label {
        Value::String(s) => {
            *s = humanize_inlay_label(s, shorten_backend);
        }
        Value::Array(parts) => {
            for part in parts.iter_mut() {
                if let Some(Value::String(s)) = part.get_mut("value") {
                    *s = humanize_inlay_label(s, shorten_backend);
                }
            }
        }
        _ => {}
    }
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

    #[test]
    fn ordinary_response_without_a_tracked_id_is_not_rewritten() {
        let msg = json!({"id": 123, "result": {"capabilities": {}}});
        let mut pending = PendingRequests::new();
        assert!(rewrite_incoming_from_server(&msg, &mut pending, true, false).is_none());
    }
}
