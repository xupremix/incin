//! Typenum-to-decimal diagnostic humanization for Kindle tooling.
//!
//! Kindle's compile-time shapes are `typenum` type-level integers
//! (`UInt<UInt<UTerm, B1>, B0>`), which is what rustc renders verbatim in
//! diagnostics. This crate rewrites that rendering into plain decimals
//! (`2`) so `cargo kindle`, an editor LSP proxy, or any other tool can all
//! show identical, human-readable numbers instead of each re-implementing
//! this parser and drifting apart.

/// The result of humanizing a diagnostic: the rewritten text, plus the
/// `(decimal, original_typenum_expr)` hint pairs discovered along the way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Translated {
    /// The input text with every typenum expression replaced by its decimal value.
    pub text: String,
    /// Each unique `(decimal, original)` pair found, in first-seen order.
    pub hints: Vec<(String, String)>,
}

/// Rewrites a compiler diagnostic (or any text) by replacing every typenum
/// expression with its decimal value. This is the single entry point the
/// CLI and any IDE tooling should call, so their output stays identical.
pub fn humanize_diagnostic(text: &str) -> Translated {
    let (text, hints) = translate_typenum_text(text);
    Translated { text, hints }
}

/// Rewrites a rust-analyzer inlay-hint (or hover) label for a Kindle tensor
/// type into a compact shape form: `Tensor<(U2, U3), CpuBackendImpl<f32,
/// Cpu>>` becomes `Tensor<[2, 3], CpuBackendImpl<f32, Cpu>>` — or, with
/// `shorten_backend: true`, `Tensor<[2, 3]>`. Returns the input unchanged if
/// it does not contain a `Tensor<(...` shell (e.g. `Dyn`-shaped tensors,
/// which aren't a typenum tuple in the first place).
pub fn humanize_inlay_label(label: &str, shorten_backend: bool) -> String {
    let Some(tensor_start) = label.find("Tensor<(") else {
        return label.to_string();
    };
    let name_end = tensor_start + "Tensor".len(); // index just past "Tensor", i.e. at '<'
    let generic_open = name_end; // index of '<'
    let tuple_open = generic_open + 1; // index of '(', guaranteed by the "Tensor<(" match above

    // Find the shape tuple's own matching ')' via balanced-paren scanning
    // (a shape tuple never nests parens, but scanning is cheap and robust).
    let Some(tuple_close) = matching_bracket(&label[tuple_open..], '(', ')') else {
        return label.to_string();
    };
    let tuple_close = tuple_open + tuple_close;

    let (shape_digits, _hints) = translate_typenum_text(&label[tuple_open + 1..tuple_close]);
    // Rust's 1-tuple syntax (`(U8,)`) leaves a trailing comma inside the
    // parens that `translate_typenum_text` faithfully preserves (it only
    // rewrites typenum spans, not surrounding punctuation) — strip it so a
    // rank-1 shape renders as `[8]`, not `[8,]`.
    let shape_digits = shape_digits.trim_end_matches(|c: char| c == ',' || c.is_whitespace());
    let shape = format!("[{}]", shape_digits);

    if !shorten_backend {
        return format!(
            "{}<{}{}",
            &label[..generic_open],
            shape,
            &label[tuple_close + 1..]
        );
    }

    // Find the matching '>' for the `Tensor<`'s own '<' so the backend/dtype/
    // grad tail after the shape tuple can be dropped entirely.
    let Some(generic_close) = matching_bracket(&label[generic_open..], '<', '>') else {
        return label.to_string();
    };
    let generic_close = generic_open + generic_close;

    format!(
        "{}<{}>{}",
        &label[..name_end],
        shape,
        &label[generic_close + 1..]
    )
}

/// Returns the byte offset (relative to `s`) of the `close` bracket that
/// matches the `open` bracket at the start of `s`, accounting for nesting.
fn matching_bracket(s: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0i32;
    for (i, ch) in s.char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Recursively parses a typenum string like `UInt<UInt<UTerm, B1>, B0>` into an integer.
pub fn parse_single_typenum(s: &str) -> Option<usize> {
    let s = s.trim();
    if s == "UTerm" {
        return Some(0);
    }
    if let Some(rest) = s.strip_prefix("UInt<") {
        if !rest.ends_with('>') {
            return None;
        }
        let inner = &rest[..rest.len() - 1];
        if let Some(u_part) = inner.strip_suffix(", B0") {
            let val = parse_single_typenum(u_part)?;
            return Some(val * 2);
        } else if let Some(u_part) = inner.strip_suffix(", B1") {
            let val = parse_single_typenum(u_part)?;
            return Some(val * 2 + 1);
        }
    }
    None
}

/// Scans text for typenum expressions, translates them into human-readable numbers,
/// and collects translation mapping hints.
pub fn translate_typenum_text(text: &str) -> (String, Vec<(String, String)>) {
    // Built up span-by-span (verbatim text interleaved with translated
    // numbers) rather than via whole-string `.replace()`: a naive replace
    // of a small nested match (e.g. `UInt<UInt<UTerm, B1>, B0>` -> `"2"`)
    // would also strike that same literal text wherever it appears nested
    // inside a larger, not-yet-processed expression elsewhere in the
    // string, corrupting it before it can be matched as a whole.
    let mut result = String::with_capacity(text.len());
    let mut hints = Vec::new();
    let mut last_end = 0;

    let mut search_idx = 0;
    while search_idx < text.len() {
        let text_slice = &text[search_idx..];
        let find_uint = text_slice.find("UInt<");
        let find_uterm = text_slice.find("UTerm");

        let next_start = match (find_uint, find_uterm) {
            (Some(u), Some(t)) => Some(u.min(t)),
            (Some(u), None) => Some(u),
            (None, Some(t)) => Some(t),
            (None, None) => None,
        };

        if let Some(start) = next_start {
            let abs_start = search_idx + start;
            if text[abs_start..].starts_with("UTerm") {
                let orig = "UTerm";
                let translated = "0";
                result.push_str(&text[last_end..abs_start]);
                result.push_str(translated);
                last_end = abs_start + orig.len();
                if !hints.iter().any(|(t, o)| t == translated && o == orig) {
                    hints.push((translated.to_string(), orig.to_string()));
                }
                search_idx = abs_start + 5;
                continue;
            }

            let mut depth = 0;
            let mut end_idx = abs_start;
            for (i, ch) in text[abs_start..].char_indices() {
                if ch == '<' {
                    depth += 1;
                } else if ch == '>' {
                    depth -= 1;
                    if depth == 0 {
                        end_idx = abs_start + i + 1;
                        break;
                    }
                }
            }

            if depth == 0 && end_idx > abs_start {
                let candidate = &text[abs_start..end_idx];
                if let Some(num) = parse_single_typenum(candidate) {
                    let translated = num.to_string();
                    result.push_str(&text[last_end..abs_start]);
                    result.push_str(&translated);
                    last_end = end_idx;
                    if !hints
                        .iter()
                        .any(|(t, o)| t == &translated && o == candidate)
                    {
                        hints.push((translated, candidate.to_string()));
                    }
                }
                search_idx = end_idx;
            } else {
                search_idx = abs_start + 5;
            }
        } else {
            break;
        }
    }
    result.push_str(&text[last_end..]);

    (result, hints)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_typenum() {
        assert_eq!(parse_single_typenum("UTerm"), Some(0));
        assert_eq!(parse_single_typenum("UInt<UTerm, B1>"), Some(1));
        assert_eq!(parse_single_typenum("UInt<UInt<UTerm, B1>, B0>"), Some(2));
        assert_eq!(parse_single_typenum("UInt<UInt<UTerm, B1>, B1>"), Some(3));
        assert_eq!(
            parse_single_typenum("UInt<UInt<UInt<UTerm, B1>, B0>, B0>"),
            Some(4)
        );
    }

    #[test]
    fn test_translate_typenum_text_with_hints() {
        let input =
            "Cannot concatenate shape (UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)";
        let (translated, hints) = translate_typenum_text(input);
        assert_eq!(translated, "Cannot concatenate shape (2, 3)");
        assert_eq!(hints.len(), 2);
        assert_eq!(
            hints[0],
            ("2".to_string(), "UInt<UInt<UTerm, B1>, B0>".to_string())
        );
        assert_eq!(
            hints[1],
            ("3".to_string(), "UInt<UInt<UTerm, B1>, B1>".to_string())
        );
    }

    /// A smaller nested expression's translation (e.g. `2`) must not
    /// clobber the same literal substring where it recurs inside a
    /// larger, separately-translated expression later in the text —
    /// regression test for a real corrupted-diagnostic bug found while
    /// reviewing this file.
    #[test]
    fn test_translate_typenum_text_does_not_corrupt_nested_reoccurring_expressions() {
        let input = "expected `UInt<UInt<UInt<UTerm, B1>, B1>, B0>`, found `UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0>`";
        let (translated, _hints) = translate_typenum_text(input);
        assert_eq!(translated, "expected `6`, found `8`");
    }

    #[test]
    fn test_humanize_diagnostic_wraps_translate_typenum_text() {
        let input = "UInt<UInt<UTerm, B1>, B0>";
        let translated = humanize_diagnostic(input);
        assert_eq!(translated.text, "2");
        assert_eq!(translated.hints, vec![("2".to_string(), input.to_string())]);
    }

    #[test]
    fn test_humanize_inlay_label_keeps_backend_by_default() {
        let label = "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<f32, Cpu>>";
        assert_eq!(
            humanize_inlay_label(label, false),
            "Tensor<[2, 3], CpuBackendImpl<f32, Cpu>>"
        );
    }

    #[test]
    fn test_humanize_inlay_label_shortens_backend_when_requested() {
        let label = "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<f32, Cpu>>";
        assert_eq!(humanize_inlay_label(label, true), "Tensor<[2, 3]>");
    }

    /// Realistic case: all four `Tensor<S, B, K, G>` type params resolved
    /// and shown (rust-analyzer often renders defaulted params explicitly),
    /// with nested `<...>` inside the backend param — the balanced-bracket
    /// scan must not stop at the first `>` it sees.
    #[test]
    fn test_humanize_inlay_label_handles_nested_angle_brackets_in_backend_param() {
        let label = "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>), CpuBackendImpl<f32, Cpu>, f32, Grad>";
        assert_eq!(
            humanize_inlay_label(label, false),
            "Tensor<[2, 4], CpuBackendImpl<f32, Cpu>, f32, Grad>"
        );
        assert_eq!(humanize_inlay_label(label, true), "Tensor<[2, 4]>");
    }

    /// Regression test: Rust's 1-tuple syntax (`(U8,)`) leaves a trailing
    /// comma inside the parens that must not end up inside the `[...]`.
    #[test]
    fn test_humanize_inlay_label_strips_trailing_comma_on_rank_one_shape() {
        let label = "Tensor<(UInt<UInt<UTerm, B1>, B0>,), CpuBackendImpl<f32, Cpu>>";
        assert_eq!(
            humanize_inlay_label(label, false),
            "Tensor<[2], CpuBackendImpl<f32, Cpu>>"
        );
        assert_eq!(humanize_inlay_label(label, true), "Tensor<[2]>");
    }

    #[test]
    fn test_humanize_inlay_label_passes_through_non_tensor_labels_unchanged() {
        assert_eq!(humanize_inlay_label("i32", false), "i32");
        assert_eq!(
            humanize_inlay_label("Tensor<Dyn, CpuBackendImpl<f32, Cpu>>", false),
            "Tensor<Dyn, CpuBackendImpl<f32, Cpu>>"
        );
    }
}
