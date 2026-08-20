//! Diagnostic humanization: the crate's core public entry points for
//! rewriting a compiler diagnostic (or IDE inlay hint / hover label) so
//! every typenum expression in it reads as a decimal, and for stripping
//! internal backend path qualifiers before display.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::typenum::{collapse_dimcons_chains, matching_bracket, translate_typenum_text};

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
    let (text, mut hints) = expand_and_substitute_type_files(text);
    let (text, text_hints) = translate_typenum_text(&text);
    for h in text_hints {
        if !hints.contains(&h) {
            hints.push(h);
        }
    }
    let text = collapse_dimcons_chains(&text);
    Translated { text, hints }
}

/// Scans diagnostic text for rustc's "the full type name was written to 'PATH'" notes,
/// attempts to read the referenced files, humanizes their type signatures/typenums,
/// substitutes any truncated `...` type spans in the diagnostic message with the full types,
/// and appends an expanded full type section directly into the diagnostic text.
pub fn expand_and_substitute_type_files(text: &str) -> (String, Vec<(String, String)>) {
    let mut hints = Vec::new();
    let mut file_lines = Vec::new();
    let mut text = text.to_string();

    const PREFIXES: &[&str] = &[
        "the full name for the type has been written to ",
        "the full type name was written to ",
    ];

    let mut search_idx = 0;
    while search_idx < text.len() {
        let text_slice = &text[search_idx..];
        let mut next_match: Option<(usize, &'static str)> = None;
        for &prefix in PREFIXES {
            if let Some(pos) = text_slice.find(prefix) {
                match next_match {
                    Some((min_pos, _)) if pos < min_pos => next_match = Some((pos, prefix)),
                    None => next_match = Some((pos, prefix)),
                    _ => {}
                }
            }
        }

        if let Some((pos, prefix)) = next_match {
            let match_start = search_idx + pos;
            let after_prefix = &text[match_start + prefix.len()..];

            // Extract path until closing quote or line break.
            let quoted_path = |quote| {
                after_prefix.strip_prefix(quote).map(|stripped| {
                    if let Some(end_quote) = stripped.find(quote) {
                        (&stripped[..end_quote], 1 + end_quote + 1)
                    } else {
                        (stripped, after_prefix.len())
                    }
                })
            };
            let (path_str, bytes_consumed) = if let Some(quoted) =
                quoted_path(char::from(39)).or_else(|| quoted_path(char::from(34)))
            {
                quoted
            } else {
                let end = after_prefix
                    .find(|c: char| c.is_whitespace() || c == '\n' || c == '\r')
                    .unwrap_or(after_prefix.len());
                (&after_prefix[..end], end)
            };

            let note_end = match_start + prefix.len() + bytes_consumed;

            let clean_path = path_str.trim();
            #[cfg(feature = "std")]
            if let Ok(file_content) = std::fs::read_to_string(clean_path) {
                for line in file_content.lines() {
                    let line = line.trim();
                    if !line.is_empty() {
                        let expanded_translated = humanize_type_signature(line, false);
                        for hint in expanded_translated.hints {
                            if !hint.1.is_empty() && !hints.contains(&hint) {
                                hints.push(hint);
                            }
                        }
                        if !file_lines.contains(&expanded_translated.text) {
                            file_lines.push(expanded_translated.text);
                        }
                    }
                }
            }
            #[cfg(not(feature = "std"))]
            let _ = clean_path;

            search_idx = note_end;
        } else {
            break;
        }
    }

    if !file_lines.is_empty() {
        text = replace_truncated_spans(&text, &file_lines);
        text.push_str("\n  └── 📄 [Expanded Full Type]:\n      ");
        text.push_str(&file_lines.join("\n      "));
    }

    (text, hints)
}

/// Replaces backticked spans in `text` that contain `...` or `UInt<` with the corresponding
/// full humanized type strings read from a rustc long-type file.
pub fn replace_truncated_spans(text: &str, file_lines: &[String]) -> String {
    if file_lines.is_empty() {
        return text.to_string();
    }

    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;
    let mut line_idx = 0;

    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'`' {
            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i] != b'`' {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'`' {
                let end = i + 1;
                let span = &text[start + 1..end - 1];

                if span.contains("...") || span.contains("...") || span.contains("UInt<") {
                    let matched_line = file_lines
                        .iter()
                        .find(|l| {
                            let prefix = span.split("...").next().unwrap_or("").trim();
                            let prefix_clean = prefix.trim_end_matches(|c: char| {
                                c == ',' || c == '<' || c.is_whitespace()
                            });
                            !prefix_clean.is_empty() && l.starts_with(prefix_clean)
                        })
                        .or_else(|| file_lines.get(line_idx));

                    if let Some(replacement) = matched_line {
                        result.push_str(&text[last_end..start + 1]);
                        result.push_str(replacement);
                        result.push('`');
                        last_end = end;
                        line_idx = (line_idx + 1) % file_lines.len();
                    }
                }
                i += 1;
                continue;
            }
        }
        i += 1;
    }

    result.push_str(&text[last_end..]);
    result
}

/// Rewrites a rust-analyzer inlay-hint (or hover) label for a Incin tensor
/// type into a compact shape form: `Tensor<(U2, U3), CpuBackendImpl<Cpu>>`
/// becomes `Tensor<[2, 3], CpuBackendImpl<Cpu>>` — or, with
/// `shorten_backend: true`, `Tensor<[2, 3]>`.
///
/// Falls back to a generic, whole-label typenum-to-decimal rewrite (the same
/// one `humanize_diagnostic` uses) for any label that isn't specifically a
/// `Tensor<(...` shell — e.g. `let conv: Conv2d<(usize, usize, UInt<...>,
/// ...), CpuBackendImpl>`, which is just as common a hint to hit as a bare
/// `Tensor` (any `let` binding of a layer/module shows one) but has no single
/// tuple that's unambiguously "the shape" the way `Tensor`'s first generic
/// param always is, so it doesn't get the `[...]` bracket treatment — every
/// `UInt<...>`/`UTerm` chain in it still becomes a plain decimal in place,
/// which is the whole readability win either way. Truly non-Incin labels
/// (`i32`, `Dyn`-shaped tensors with no typenum content) pass through
/// byte-identical, since there's nothing for the underlying translator to find.
pub fn humanize_inlay_label(label: &str, shorten_backend: bool) -> String {
    humanize_type_signature(label, shorten_backend).text
}

/// Same rewrite as [`humanize_inlay_label`], but also returns the
/// `(decimal, original)` hint pairs discovered along the way — for callers
/// like hover, which (unlike an inlay hint's cramped ghost text) have room
/// to show a legend mapping each humanized number back to its raw typenum
/// expression.
pub fn humanize_type_signature(label: &str, shorten_backend: bool) -> Translated {
    let no_op = || Translated {
        text: label.to_string(),
        hints: Vec::new(),
    };

    let Some(tensor_start) = label.find("Tensor<(") else {
        let (text, hints) = translate_typenum_text(label);
        return Translated { text, hints };
    };
    let name_end = tensor_start + "Tensor".len(); // index just past "Tensor", i.e. at '<'
    let generic_open = name_end; // index of '<'
    let tuple_open = generic_open + 1; // index of '(', guaranteed by the "Tensor<(" match above

    // Find the shape tuple's own matching ')' via balanced-paren scanning
    // (a shape tuple never nests parens, but scanning is cheap and robust).
    let Some(tuple_close) = matching_bracket(&label[tuple_open..], '(', ')') else {
        return no_op();
    };
    let tuple_close = tuple_open + tuple_close;

    let (shape_digits, hints) = translate_typenum_text(&label[tuple_open + 1..tuple_close]);
    // Rust's 1-tuple syntax (`(U8,)`) leaves a trailing comma inside the
    // parens that `translate_typenum_text` faithfully preserves (it only
    // rewrites typenum spans, not surrounding punctuation) — strip it so a
    // rank-1 shape renders as `[8]`, not `[8,]`.
    let shape_digits = shape_digits.trim_end_matches(|c: char| c == ',' || c.is_whitespace());
    let shape = format!("[{}]", shape_digits);

    if !shorten_backend {
        let text = format!(
            "{}<{}{}",
            &label[..generic_open],
            shape,
            &label[tuple_close + 1..]
        );
        return Translated { text, hints };
    }

    // Find the matching '>' for the `Tensor<`'s own '<' so the backend/dtype/
    // grad tail after the shape tuple can be dropped entirely.
    let Some(generic_close) = matching_bracket(&label[generic_open..], '<', '>') else {
        return no_op();
    };
    let generic_close = generic_open + generic_close;

    let text = format!(
        "{}<{}>{}",
        &label[..name_end],
        shape,
        &label[generic_close + 1..]
    );
    Translated { text, hints }
}

/// Collapses every qualified path in a type signature down to its last
/// segment: `incin::cpu::Tensor` becomes `Tensor`, `typenum::B1` becomes
/// `B1`. Intended for rust-analyzer's inlay-hint `textEdits[0].newText`,
/// which spells every type fully-qualified (so it inserts correctly at any
/// scope) — that fully-qualified form is otherwise identical in shape to the
/// short-path form `humanize_inlay_label` already knows how to parse, so
/// normalizing it first lets the same parser handle both.
pub fn strip_path_qualifiers(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut result = String::with_capacity(s.len());
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut i = 0;
    while i < chars.len() {
        if is_ident(chars[i]) && (i == 0 || !is_ident(chars[i - 1])) {
            let mut j = i;
            let mut last_seg_start = i;
            loop {
                while j < chars.len() && is_ident(chars[j]) {
                    j += 1;
                }
                if j + 1 < chars.len() && chars[j] == ':' && chars[j + 1] == ':' {
                    j += 2;
                    last_seg_start = j;
                } else {
                    break;
                }
            }
            result.extend(&chars[last_seg_start..j]);
            i = j;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    result
}
