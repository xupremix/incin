//! Typenum-to-decimal diagnostic humanization for Incin tooling.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

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
/// type into a compact shape form: `Tensor<(U2, U3), CpuBackendImpl<f32,
/// Cpu>>` becomes `Tensor<[2, 3], CpuBackendImpl<Cpu>>` — or, with
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

/// A `matmul` shape mismatch, parsed from the `MatMulShape` trait's
/// `#[diagnostic::on_unimplemented]` message, with the specific conflicting
/// dimension identified and ready to render as a pointed-out explanation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMulMismatch {
    /// The lhs shape's elements, humanized for display (e.g. `["2", "4"]`).
    pub lhs: Vec<String>,
    /// The rhs shape's elements, humanized for display (e.g. `["5", "6"]`).
    pub rhs: Vec<String>,
    /// Index into `lhs` of the conflicting "inner" dimension (always the
    /// last element — matmul's `K` from the lhs side).
    pub lhs_inner_index: usize,
    /// Index into `rhs` of the conflicting "inner" dimension (the
    /// second-to-last element — matmul's `K` from the rhs side; usually
    /// index `0` for a plain `(K, N)`, but shifts right with batch dims).
    pub rhs_inner_index: usize,
}

impl MatMulMismatch {
    /// Renders the mismatch as a multi-line, ready-to-print explanation:
    /// both shapes shown with the conflicting dimension pointed out via a
    /// `^` marker underneath, plus a concrete suggested fix.
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        let lhs_inner = &self.lhs[self.lhs_inner_index];
        let rhs_inner = &self.rhs[self.rhs_inner_index];

        let lhs_label = format!("{INDENT}lhs shape = (");
        let rhs_label = format!("{INDENT}rhs shape = (");
        // `{:>N$}` right-aligns within a field of width `N`, so the marker
        // ends up at index `N - 1` — add 1 to land it exactly at the target
        // column (the offset of the element's first character).
        let lhs_caret_width =
            lhs_label.len() + joined_prefix_len(&self.lhs, self.lhs_inner_index) + 1;
        let rhs_caret_width =
            rhs_label.len() + joined_prefix_len(&self.rhs, self.rhs_inner_index) + 1;

        format!(
            "{lhs_label}{lhs_shape})\n\
             {lhs_caret:>lhs_caret_width$} inner dim = {lhs_inner}\n\
             {rhs_label}{rhs_shape})\n\
             {rhs_caret:>rhs_caret_width$} inner dim = {rhs_inner}\n\
             {INDENT}{lhs_inner} \u{2260} {rhs_inner} \u{2192} change the lhs inner dim from {lhs_inner} to {rhs_inner}, or the rhs inner dim from {rhs_inner} to {lhs_inner}.",
            lhs_shape = self.lhs.join(", "),
            rhs_shape = self.rhs.join(", "),
            lhs_caret = "^",
            rhs_caret = "^",
        )
    }
}

/// Length, in characters, of everything that would precede `elements[index]`
/// once `elements` is joined with `", "` — i.e. where that element's own
/// text starts within the joined string.
fn joined_prefix_len(elements: &[String], index: usize) -> usize {
    elements[..index].iter().map(|e| e.len() + 2).sum()
}

/// Parses the `MatMulShape` trait's fixed on_unimplemented message —
/// `` Cannot matrix-multiply shape `{Self}` with `{Rhs}` `` — and, if the
/// inner dimensions (last element of `Self`, second-to-last of `Rhs`, per
/// the trait's own rule) actually differ, returns a [`MatMulMismatch`]
/// ready to render. Returns `None` if `text` isn't this message, either
/// shape isn't a plain tuple, or the inner dimensions match (nothing to
/// explain — the real failure is something else, e.g. a rank mismatch).
pub fn parse_matmul_mismatch(text: &str) -> Option<MatMulMismatch> {
    // Search rather than `strip_prefix` on the whole input: callers (e.g.
    // `cargo incin --explain`) pass the *entire* rendered diagnostic —
    // "error[E0277]: Cannot matrix-multiply shape `...` with `...`\n   -->
    // file:line:col\n..." — not just this one message in isolation.
    let start = text.find("Cannot matrix-multiply shape `")?;
    let after_prefix = &text[start + "Cannot matrix-multiply shape `".len()..];
    let (lhs_raw, after_lhs) = after_prefix.split_once('`')?;
    let after_lhs = after_lhs.strip_prefix(" with `")?;
    let (rhs_raw, _) = after_lhs.split_once('`')?;

    let lhs_inner_text = lhs_raw.strip_prefix('(')?.strip_suffix(')')?;
    let rhs_inner_text = rhs_raw.strip_prefix('(')?.strip_suffix(')')?;

    let lhs_elems: Vec<&str> = split_top_level_commas(lhs_inner_text);
    let rhs_elems: Vec<&str> = split_top_level_commas(rhs_inner_text);
    if lhs_elems.is_empty() || rhs_elems.len() < 2 {
        return None;
    }

    let lhs_inner_index = lhs_elems.len() - 1;
    let rhs_inner_index = rhs_elems.len() - 2;

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let lhs: Vec<String> = lhs_elems.iter().map(|e| humanize(e)).collect();
    let rhs: Vec<String> = rhs_elems.iter().map(|e| humanize(e)).collect();

    if lhs[lhs_inner_index] == rhs[rhs_inner_index] {
        return None; // inner dims already agree — a different rule failed
    }

    Some(MatMulMismatch {
        lhs,
        rhs,
        lhs_inner_index,
        rhs_inner_index,
    })
}

/// Splits `s` on top-level commas (i.e. not nested inside `<...>`) — a
/// typenum shape tuple's elements never contain their own parens, only
/// angle-bracket generics, so tracking `<`/`>` depth alone is sufficient.
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() || !parts.is_empty() {
        parts.push(last);
    }
    parts
}

/// A `concat` shape mismatch, parsed from the `ConcatShape` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatMismatch {
    pub lhs: Vec<String>,
    pub rhs: Vec<String>,
    pub axis: usize,
    pub mismatch_index: usize,
}

impl ConcatMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        let lhs_val = &self.lhs[self.mismatch_index];
        let rhs_val = &self.rhs[self.mismatch_index];

        let lhs_label = format!("{INDENT}lhs shape = (");
        let rhs_label = format!("{INDENT}rhs shape = (");
        let lhs_caret_width =
            lhs_label.len() + joined_prefix_len(&self.lhs, self.mismatch_index) + 1;
        let rhs_caret_width =
            rhs_label.len() + joined_prefix_len(&self.rhs, self.mismatch_index) + 1;

        format!(
            "{lhs_label}{lhs_shape})\n\
             {lhs_caret:>lhs_caret_width$} non-concat dim = {lhs_val}\n\
             {rhs_label}{rhs_shape})\n\
             {rhs_caret:>rhs_caret_width$} non-concat dim = {rhs_val}\n\
             {INDENT}{lhs_val} \u{2260} {rhs_val} \u{2192} concatenating along axis {axis} requires all other dimensions to match exactly.",
            lhs_shape = self.lhs.join(", "),
            rhs_shape = self.rhs.join(", "),
            lhs_caret = "^",
            rhs_caret = "^",
            axis = self.axis,
        )
    }
}

pub fn parse_concat_mismatch(text: &str) -> Option<ConcatMismatch> {
    let start = text.find("Cannot concatenate shape `")?;
    let after_prefix = &text[start + "Cannot concatenate shape `".len()..];
    let (lhs_raw, after_lhs) = after_prefix.split_once('`')?;
    let after_lhs = after_lhs.strip_prefix(" with `")?;
    let (rhs_raw, after_rhs) = after_lhs.split_once('`')?;
    let axis_str = after_rhs.strip_prefix(" along axis `")?.split_once('`')?.0;

    let axis: usize = axis_str.trim().parse().ok()?;

    let lhs_inner_text = lhs_raw.strip_prefix('(')?.strip_suffix(')')?;
    let rhs_inner_text = rhs_raw.strip_prefix('(')?.strip_suffix(')')?;

    let lhs_elems: Vec<&str> = split_top_level_commas(lhs_inner_text);
    let rhs_elems: Vec<&str> = split_top_level_commas(rhs_inner_text);

    if lhs_elems.len() != rhs_elems.len() {
        return None;
    }

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let lhs: Vec<String> = lhs_elems.iter().map(|e| humanize(e)).collect();
    let rhs: Vec<String> = rhs_elems.iter().map(|e| humanize(e)).collect();

    for i in 0..lhs.len() {
        if i != axis && lhs[i] != rhs[i] {
            return Some(ConcatMismatch {
                lhs,
                rhs,
                axis,
                mismatch_index: i,
            });
        }
    }

    None
}

/// A `broadcast` shape mismatch, parsed from the `BroadcastShape` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastMismatch {
    pub lhs: Vec<String>,
    pub rhs: Vec<String>,
    pub lhs_mismatch_index: usize,
    pub rhs_mismatch_index: usize,
}

impl BroadcastMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        let lhs_val = &self.lhs[self.lhs_mismatch_index];
        let rhs_val = &self.rhs[self.rhs_mismatch_index];

        let lhs_label = format!("{INDENT}lhs shape = (");
        let rhs_label = format!("{INDENT}rhs shape = (");
        let lhs_caret_width =
            lhs_label.len() + joined_prefix_len(&self.lhs, self.lhs_mismatch_index) + 1;
        let rhs_caret_width =
            rhs_label.len() + joined_prefix_len(&self.rhs, self.rhs_mismatch_index) + 1;

        format!(
            "{lhs_label}{lhs_shape})\n\
             {lhs_caret:>lhs_caret_width$} dim = {lhs_val}\n\
             {rhs_label}{rhs_shape})\n\
             {rhs_caret:>rhs_caret_width$} dim = {rhs_val}\n\
             {INDENT}{lhs_val} \u{2260} {rhs_val} \u{2192} broadcast requires corresponding dimensions to be equal, or one of them to be 1.",
            lhs_shape = self.lhs.join(", "),
            rhs_shape = self.rhs.join(", "),
            lhs_caret = "^",
            rhs_caret = "^",
        )
    }
}

pub fn parse_broadcast_mismatch(text: &str) -> Option<BroadcastMismatch> {
    let start = text.find("Cannot broadcast shape `")?;
    let after_prefix = &text[start + "Cannot broadcast shape `".len()..];
    let (lhs_raw, after_lhs) = after_prefix.split_once('`')?;
    let after_lhs = after_lhs.strip_prefix(" to `")?;
    let (rhs_raw, _) = after_lhs.split_once('`')?;

    let lhs_inner_text = lhs_raw.strip_prefix('(')?.strip_suffix(')')?;
    let rhs_inner_text = rhs_raw.strip_prefix('(')?.strip_suffix(')')?;

    let lhs_elems: Vec<&str> = split_top_level_commas(lhs_inner_text);
    let rhs_elems: Vec<&str> = split_top_level_commas(rhs_inner_text);

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let lhs: Vec<String> = lhs_elems.iter().map(|e| humanize(e)).collect();
    let rhs: Vec<String> = rhs_elems.iter().map(|e| humanize(e)).collect();

    let min_rank = lhs.len().min(rhs.len());
    for i in 1..=min_rank {
        let l_idx = lhs.len() - i;
        let r_idx = rhs.len() - i;
        let l_val = &lhs[l_idx];
        let r_val = &rhs[r_idx];
        if l_val != r_val && l_val != "1" && r_val != "1" {
            return Some(BroadcastMismatch {
                lhs,
                rhs,
                lhs_mismatch_index: l_idx,
                rhs_mismatch_index: r_idx,
            });
        }
    }

    None
}

/// A `reshape` shape mismatch, parsed from the `ReshapeShape` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReshapeMismatch {
    pub src: Vec<String>,
    pub target: Vec<String>,
    pub src_count: usize,
    pub target_count: usize,
}

impl ReshapeMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}src shape    = ({src_shape})  [total elements = {src_count}]\n\
             {INDENT}target shape = ({target_shape})  [total elements = {target_count}]\n\
             {INDENT}{src_count} \u{2260} {target_count} \u{2192} reshape requires the product of dimensions (total element count) to remain identical.",
            src_shape = self.src.join(", "),
            target_shape = self.target.join(", "),
            src_count = self.src_count,
            target_count = self.target_count,
        )
    }
}

pub fn parse_reshape_mismatch(text: &str) -> Option<ReshapeMismatch> {
    let start = text.find("Cannot reshape from `")?;
    let after_prefix = &text[start + "Cannot reshape from `".len()..];
    let (src_raw, after_src) = after_prefix.split_once('`')?;
    let after_src = after_src.strip_prefix(" to `")?;
    let (target_raw, _) = after_src.split_once('`')?;

    let src_inner_text = src_raw.strip_prefix('(')?.strip_suffix(')')?;
    let target_inner_text = target_raw.strip_prefix('(')?.strip_suffix(')')?;

    let src_elems: Vec<&str> = split_top_level_commas(src_inner_text);
    let target_elems: Vec<&str> = split_top_level_commas(target_inner_text);

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let src: Vec<String> = src_elems.iter().map(|e| humanize(e)).collect();
    let target: Vec<String> = target_elems.iter().map(|e| humanize(e)).collect();

    let mut src_count = 1;
    for s in &src {
        src_count *= s.parse::<usize>().ok()?;
    }
    let mut target_count = 1;
    for t in &target {
        target_count *= t.parse::<usize>().ok()?;
    }

    if src_count != target_count {
        Some(ReshapeMismatch {
            src,
            target,
            src_count,
            target_count,
        })
    } else {
        None
    }
}

/// A `conv2d` shape mismatch, parsed from the `Conv2dShape` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv2dMismatch {
    pub input: Vec<String>,
    pub kernel: Vec<String>,
    pub input_channel_idx: usize,
    pub kernel_channel_idx: usize,
}

impl Conv2dMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        let in_c = &self.input[self.input_channel_idx];
        let k_c = &self.kernel[self.kernel_channel_idx];

        let in_label = format!("{INDENT}input shape  = (");
        let k_label = format!("{INDENT}kernel shape = (");
        let in_caret_width =
            in_label.len() + joined_prefix_len(&self.input, self.input_channel_idx) + 1;
        let k_caret_width =
            k_label.len() + joined_prefix_len(&self.kernel, self.kernel_channel_idx) + 1;

        format!(
            "{in_label}{in_shape})\n\
             {in_caret:>in_caret_width$} input channels = {in_c}\n\
             {k_label}{k_shape})\n\
             {k_caret:>k_caret_width$} kernel in_channels = {k_c}\n\
             {INDENT}{in_c} \u{2260} {k_c} \u{2192} Conv2D input channels must match kernel input channels.",
            in_shape = self.input.join(", "),
            k_shape = self.kernel.join(", "),
            in_caret = "^",
            k_caret = "^",
        )
    }
}

pub fn parse_conv2d_mismatch(text: &str) -> Option<Conv2dMismatch> {
    let start = text.find("Cannot apply Conv2D: input shape `")?;
    let after_prefix = &text[start + "Cannot apply Conv2D: input shape `".len()..];
    let (in_raw, after_in) = after_prefix.split_once('`')?;
    let after_in = after_in.strip_prefix(" is incompatible with kernel shape `")?;
    let (k_raw, _) = after_in.split_once('`')?;

    let in_inner_text = in_raw.strip_prefix('(')?.strip_suffix(')')?;
    let k_inner_text = k_raw.strip_prefix('(')?.strip_suffix(')')?;

    let in_elems: Vec<&str> = split_top_level_commas(in_inner_text);
    let k_elems: Vec<&str> = split_top_level_commas(k_inner_text);

    if in_elems.len() < 2 || k_elems.len() < 2 {
        return None;
    }

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let input: Vec<String> = in_elems.iter().map(|e| humanize(e)).collect();
    let kernel: Vec<String> = k_elems.iter().map(|e| humanize(e)).collect();

    if input[1] != kernel[1] {
        Some(Conv2dMismatch {
            input,
            kernel,
            input_channel_idx: 1,
            kernel_channel_idx: 1,
        })
    } else {
        None
    }
}

/// A `transpose` shape mismatch, parsed from the `Transpose` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransposeMismatch {
    pub shape: Vec<String>,
    pub d1: usize,
    pub d2: usize,
    pub rank: usize,
}

impl TransposeMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}shape = ({shape})  [rank = {rank}]\n\
             {INDENT}transpose indices = ({d1}, {d2})\n\
             {INDENT}invalid dimension index \u{2192} both transpose indices must be < rank ({rank}).",
            shape = self.shape.join(", "),
            rank = self.rank,
            d1 = self.d1,
            d2 = self.d2,
        )
    }
}

pub fn parse_transpose_mismatch(text: &str) -> Option<TransposeMismatch> {
    let start = text.find("Cannot transpose dimensions `")?;
    let after_prefix = &text[start + "Cannot transpose dimensions `".len()..];
    let (d1_str, after_d1) = after_prefix.split_once('`')?;
    let after_d1 = after_d1.strip_prefix(" and `")?;
    let (d2_str, after_d2) = after_d1.split_once('`')?;
    let after_d2 = after_d2.strip_prefix(" on shape `")?;
    let (shape_raw, _) = after_d2.split_once('`')?;

    let d1: usize = d1_str.trim().parse().ok()?;
    let d2: usize = d2_str.trim().parse().ok()?;

    let shape_inner = shape_raw.strip_prefix('(')?.strip_suffix(')')?;
    let elems: Vec<&str> = split_top_level_commas(shape_inner);

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let shape: Vec<String> = elems.iter().map(|e| humanize(e)).collect();
    let rank = shape.len();

    if d1 >= rank || d2 >= rank {
        Some(TransposeMismatch {
            shape,
            d1,
            d2,
            rank,
        })
    } else {
        None
    }
}

/// A `reduce` shape mismatch, parsed from the `ReduceDim` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceDimMismatch {
    pub shape: Vec<String>,
    pub dim: usize,
    pub rank: usize,
}

impl ReduceDimMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}shape = ({shape})  [rank = {rank}]\n\
             {INDENT}reduce dim = {dim}\n\
             {INDENT}{dim} \u{2265} {rank} \u{2192} reduction dimension must be < rank ({rank}).",
            shape = self.shape.join(", "),
            rank = self.rank,
            dim = self.dim,
        )
    }
}

pub fn parse_reduce_dim_mismatch(text: &str) -> Option<ReduceDimMismatch> {
    let start = text.find("Cannot reduce dimension `")?;
    let after_prefix = &text[start + "Cannot reduce dimension `".len()..];
    let (dim_str, after_dim) = after_prefix.split_once('`')?;
    let after_dim = after_dim
        .find("on shape `")
        .map(|p| &after_dim[p + "on shape `".len()..])?;
    let (shape_raw, _) = after_dim.split_once('`')?;

    let dim: usize = dim_str.trim().parse().ok()?;

    let shape_inner = shape_raw.strip_prefix('(')?.strip_suffix(')')?;
    let elems: Vec<&str> = split_top_level_commas(shape_inner);

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let shape: Vec<String> = elems.iter().map(|e| humanize(e)).collect();
    let rank = shape.len();

    if dim >= rank {
        Some(ReduceDimMismatch { shape, dim, rank })
    } else {
        None
    }
}

/// A `flatten` shape mismatch, parsed from the `Flatten` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlattenMismatch {
    pub shape: Vec<String>,
    pub start_dim: usize,
    pub end_dim: usize,
    pub rank: usize,
}

impl FlattenMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}shape = ({shape})  [rank = {rank}]\n\
             {INDENT}flatten range = [{start_dim}, {end_dim}]\n\
             {INDENT}invalid range \u{2192} flatten requires start <= end and end < rank ({rank}).",
            shape = self.shape.join(", "),
            rank = self.rank,
            start_dim = self.start_dim,
            end_dim = self.end_dim,
        )
    }
}

pub fn parse_flatten_mismatch(text: &str) -> Option<FlattenMismatch> {
    let start = text.find("Cannot flatten shape `")?;
    let after_prefix = &text[start + "Cannot flatten shape `".len()..];
    let (shape_raw, after_shape) = after_prefix.split_once('`')?;
    let after_shape = after_shape.strip_prefix(" from dimension `")?;
    let (start_str, after_start) = after_shape.split_once('`')?;
    let after_start = after_start.strip_prefix(" to `")?;
    let (end_str, _) = after_start.split_once('`')?;

    let start_dim: usize = start_str.trim().parse().ok()?;
    let end_dim: usize = end_str.trim().parse().ok()?;

    let shape_inner = shape_raw.strip_prefix('(')?.strip_suffix(')')?;
    let elems: Vec<&str> = split_top_level_commas(shape_inner);

    let humanize = |s: &str| humanize_diagnostic(s.trim()).text;
    let shape: Vec<String> = elems.iter().map(|e| humanize(e)).collect();
    let rank = shape.len();

    if start_dim > end_dim || end_dim >= rank {
        Some(FlattenMismatch {
            shape,
            start_dim,
            end_dim,
            rank,
        })
    } else {
        None
    }
}

/// A `Module::forward` input shape mismatch, parsed from compiler output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleForwardMismatch {
    pub actual_input: String,
    pub expected_input: String,
}

impl ModuleForwardMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}provided input shape = {actual}\n\
             {INDENT}expected input shape = {expected}\n\
             {INDENT}shape mismatch \u{2192} layer forward pass expects {expected}, but received {actual}.",
            actual = self.actual_input,
            expected = self.expected_input,
        )
    }
}

pub fn parse_module_forward_mismatch(text: &str) -> Option<ModuleForwardMismatch> {
    let mut search = text;
    let mut actual_raw = None;
    while let Some(start) = search.find("Module<") {
        let after = &search[start + "Module<".len()..];
        if let Some(close) = matching_bracket(after, '<', '>') {
            let inner = &after[..close];
            if inner.contains("Tensor") {
                actual_raw = Some(inner);
                break;
            }
        }
        if after.is_empty() {
            break;
        }
        search = &after[1..];
    }
    let actual_raw = actual_raw?;

    let impl_start = text.find("implements `Module<")?;
    let after_impl = &text[impl_start + "implements `Module<".len()..];
    let close_impl = matching_bracket(after_impl, '<', '>')?;
    let expected_raw = &after_impl[..close_impl];

    let clean = |s: &str| {
        let mut res = humanize_diagnostic(s.trim()).text.trim().to_string();
        let open_count = res.chars().filter(|&c| c == '<').count();
        let close_count = res.chars().filter(|&c| c == '>').count();
        if open_count > close_count {
            res.push('>');
        }
        res
    };

    let actual = clean(actual_raw);
    let expected = clean(expected_raw);

    if actual != expected {
        Some(ModuleForwardMismatch {
            actual_input: actual,
            expected_input: expected,
        })
    } else {
        None
    }
}

/// A slice target shape mismatch, parsed from the `SliceTarget` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceMismatch {
    pub slice_spec: String,
    pub in_shape: String,
}

impl SliceMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}input shape = {in_shape}\n\
             {INDENT}slice spec  = {slice_spec}\n\
             {INDENT}invalid slice \u{2192} slice ranges must fall within input dimension bounds.",
            in_shape = self.in_shape,
            slice_spec = self.slice_spec,
        )
    }
}

pub fn parse_slice_mismatch(text: &str) -> Option<SliceMismatch> {
    let start = text.find("Cannot slice dimension with `")?;
    let after = &text[start + "Cannot slice dimension with `".len()..];
    let (spec_raw, after_spec) = after.split_once('`')?;

    let in_shape_raw = if let Some(pos) = after_spec.find("for shape `") {
        let after_for = &after_spec[pos + "for shape `".len()..];
        after_for.split_once('`').map(|(s, _)| s)
    } else {
        None
    };

    let slice_spec = humanize_diagnostic(spec_raw.trim()).text;
    let in_shape = in_shape_raw
        .map(|s| humanize_diagnostic(s.trim()).text)
        .unwrap_or_else(|| "unknown".to_string());

    Some(SliceMismatch {
        slice_spec,
        in_shape,
    })
}

/// A `conv1d` shape mismatch, parsed from the `SpatialConv1d` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv1dMismatch {
    pub input_shape: String,
}

impl Conv1dMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}input shape = {shape}\n\
             {INDENT}invalid shape \u{2192} Conv1D requires a 2D or 3D tensor (C, L) or (B, C, L).",
            shape = self.input_shape,
        )
    }
}

pub fn parse_conv1d_mismatch(text: &str) -> Option<Conv1dMismatch> {
    let start = text.find("Cannot apply 1D convolution to shape `")?;
    let after = &text[start + "Cannot apply 1D convolution to shape `".len()..];
    let (raw_shape, _) = after.split_once('`')?;

    let input_shape = humanize_diagnostic(raw_shape.trim()).text;
    Some(Conv1dMismatch { input_shape })
}

/// A 2D pooling shape mismatch, parsed from the `Pool2dShape` trait's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pool2dMismatch {
    pub input_shape: String,
}

impl Pool2dMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}input shape = {shape}\n\
             {INDENT}invalid shape \u{2192} Pool2D requires a 3D or 4D tensor (C, H, W) or (B, C, H, W).",
            shape = self.input_shape,
        )
    }
}

pub fn parse_pool2d_mismatch(text: &str) -> Option<Pool2dMismatch> {
    let start = text.find("Cannot apply 2D pooling to shape `")?;
    let after = &text[start + "Cannot apply 2D pooling to shape `".len()..];
    let (raw_shape, _) = after.split_once('`')?;

    let input_shape = humanize_diagnostic(raw_shape.trim()).text;
    Some(Pool2dMismatch { input_shape })
}

/// A shape equality mismatch, parsed from the `ShapeEq` trait's compile-time error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeEqMismatch {
    pub message: String,
}

impl ShapeEqMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!("{INDENT}shape mismatch \u{2192} {msg}", msg = self.message,)
    }
}

pub fn parse_shape_eq_mismatch(text: &str) -> Option<ShapeEqMismatch> {
    if let Some(start) = text.find("Shape Mismatch:") {
        let after = &text[start + "Shape Mismatch:".len()..];
        let end = after.find('\n').unwrap_or(after.len());
        let message = after[..end].trim().to_string();
        Some(ShapeEqMismatch { message })
    } else {
        None
    }
}

/// A `bmm` rank or shape mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BmmMismatch {
    pub message: String,
}

impl BmmMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!("{INDENT}bmm mismatch \u{2192} {msg}", msg = self.message)
    }
}

pub fn parse_bmm_mismatch(text: &str) -> Option<BmmMismatch> {
    if text.contains("bmm") || text.contains("batched matrix multiplication") {
        Some(BmmMismatch {
            message: "BMM requires 3D tensors (B, M, K) x (B, K, N)".to_string(),
        })
    } else {
        None
    }
}

/// An `unfold` dimension bound mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnfoldMismatch {
    pub message: String,
}

impl UnfoldMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!("{INDENT}unfold mismatch \u{2192} {msg}", msg = self.message)
    }
}

pub fn parse_unfold_mismatch(text: &str) -> Option<UnfoldMismatch> {
    if text.contains("unfold size cannot exceed dimension length") {
        Some(UnfoldMismatch {
            message: "unfold size exceeds target dimension length".to_string(),
        })
    } else {
        None
    }
}

/// A `pixel_shuffle` channel divisibility mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelShuffleMismatch {
    pub message: String,
}

impl PixelShuffleMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}pixel_shuffle mismatch \u{2192} {msg}",
            msg = self.message
        )
    }
}

pub fn parse_pixel_shuffle_mismatch(text: &str) -> Option<PixelShuffleMismatch> {
    if text.contains("pixel_shuffle channels must be divisible") {
        Some(PixelShuffleMismatch {
            message: "channel count must be divisible by upscale_factor^2".to_string(),
        })
    } else {
        None
    }
}

/// A `group_norm` channel divisibility mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupNormMismatch {
    pub message: String,
}

impl GroupNormMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}group_norm mismatch \u{2192} {msg}",
            msg = self.message
        )
    }
}

pub fn parse_group_norm_mismatch(text: &str) -> Option<GroupNormMismatch> {
    if text.contains("group_norm: channels must be divisible by groups") {
        Some(GroupNormMismatch {
            message: "channels count must be divisible by groups".to_string(),
        })
    } else {
        None
    }
}

/// A math domain error diagnostic (e.g., asin/acos out of bounds, rsqrt non-positive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MathDomainError {
    pub message: String,
}

impl MathDomainError {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}math domain error \u{2192} {msg}",
            msg = self.message
        )
    }
}

pub fn parse_math_domain_error(text: &str) -> Option<MathDomainError> {
    if text.contains("out of domain")
        || text.contains("NaN domain")
        || text.contains("domain error")
    {
        Some(MathDomainError {
            message: "argument value is outside the real domain of the function".to_string(),
        })
    } else {
        None
    }
}

/// An in-place shape mismatch diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InPlaceShapeMismatch {
    pub message: String,
}

impl InPlaceShapeMismatch {
    pub fn render(&self) -> String {
        const INDENT: &str = "      ";
        format!(
            "{INDENT}in-place shape mismatch \u{2192} {msg}",
            msg = self.message
        )
    }
}

pub fn parse_inplace_shape_mismatch(text: &str) -> Option<InPlaceShapeMismatch> {
    if text.contains("in-place operand shape mismatch") || text.contains("cannot mutate in-place") {
        Some(InPlaceShapeMismatch {
            message:
                "target tensor and operand tensor must have identical shapes for in-place mutation"
                    .to_string(),
        })
    } else {
        None
    }
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
    // The crate is `no_std` without the `std` feature, so `vec!` is not in the
    // prelude for either configuration of this module. Importing it from
    // `alloc` is what makes the tests compile under both.
    use alloc::vec;

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
        let label =
            "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<Cpu>>";
        assert_eq!(
            humanize_inlay_label(label, false),
            "Tensor<[2, 3], CpuBackendImpl<Cpu>>"
        );
    }

    #[test]
    fn test_humanize_inlay_label_shortens_backend_when_requested() {
        let label =
            "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<Cpu>>";
        assert_eq!(humanize_inlay_label(label, true), "Tensor<[2, 3]>");
    }

    /// Realistic case: all four `Tensor<S, B, K, G>` type params resolved
    /// and shown (rust-analyzer often renders defaulted params explicitly),
    /// with nested `<...>` inside the backend param — the balanced-bracket
    /// scan must not stop at the first `>` it sees.
    #[test]
    fn test_humanize_inlay_label_handles_nested_angle_brackets_in_backend_param() {
        let label = "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>), CpuBackendImpl<Cpu>, f32, Grad>";
        assert_eq!(
            humanize_inlay_label(label, false),
            "Tensor<[2, 4], CpuBackendImpl<Cpu>, f32, Grad>"
        );
        assert_eq!(humanize_inlay_label(label, true), "Tensor<[2, 4]>");
    }

    /// Regression test: Rust's 1-tuple syntax (`(U8,)`) leaves a trailing
    /// comma inside the parens that must not end up inside the `[...]`.
    #[test]
    fn test_humanize_inlay_label_strips_trailing_comma_on_rank_one_shape() {
        let label = "Tensor<(UInt<UInt<UTerm, B1>, B0>,), CpuBackendImpl<Cpu>>";
        assert_eq!(
            humanize_inlay_label(label, false),
            "Tensor<[2], CpuBackendImpl<Cpu>>"
        );
        assert_eq!(humanize_inlay_label(label, true), "Tensor<[2]>");
    }

    /// Regression test for a real, reported case: a `let` binding of a
    /// layer/module (not a bare `Tensor`) shows its full inferred type,
    /// which is just as typenum-heavy but has no `Tensor<(...` shell —
    /// previously passed through completely raw.
    #[test]
    fn test_humanize_inlay_label_rewrites_non_tensor_struct_types_generically() {
        let label = "Conv2d<(usize, usize, UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>), UInt<UTerm, B1>>, CpuBackendImpl>";
        assert_eq!(
            humanize_inlay_label(label, false),
            "Conv2d<(usize, usize, 3, 3, 3), 1>, CpuBackendImpl>"
        );
    }

    #[test]
    fn test_strip_path_qualifiers_collapses_qualified_paths() {
        assert_eq!(strip_path_qualifiers("typenum::B1"), "B1");
        assert_eq!(strip_path_qualifiers("incin::cpu::Tensor"), "Tensor");
        assert_eq!(strip_path_qualifiers("f32"), "f32");
        assert_eq!(
            strip_path_qualifiers(
                "incin::cpu::Tensor<(typenum::UInt<typenum::UInt<typenum::UTerm, typenum::B1>, typenum::B1>,), incin::cpu::CpuBackendImpl, f32>"
            ),
            "Tensor<(UInt<UInt<UTerm, B1>, B1>,), CpuBackendImpl, f32>"
        );
    }

    /// Regression test for the real bug this was written to fix: rust-analyzer
    /// truncates deeply-nested inlay-hint `label`s with a `…` ellipsis once
    /// they exceed its default nesting depth, which loses the B0/B1 bits
    /// entirely — no rewrite of `label` can recover data that was never sent.
    /// The full type only survives in `textEdits[0].newText`, fully
    /// path-qualified; `strip_path_qualifiers` normalizes that back into the
    /// same shape `humanize_inlay_label` already parses.
    #[test]
    fn test_humanize_inlay_label_recovers_truncated_hint_via_stripped_text_edit() {
        let full_text_edit = "incin::cpu::Tensor<(typenum::UInt<typenum::UInt<typenum::UTerm, typenum::B1>, typenum::B1>, typenum::UInt<typenum::UInt<typenum::UInt<typenum::UTerm, typenum::B1>, typenum::B0>, typenum::B1>), incin::cpu::CpuBackendImpl, f32>";
        let normalized = strip_path_qualifiers(full_text_edit);
        assert_eq!(
            humanize_inlay_label(&normalized, false),
            "Tensor<[3, 5], CpuBackendImpl, f32>"
        );
        assert_eq!(humanize_inlay_label(&normalized, true), "Tensor<[3, 5]>");
    }

    #[test]
    fn test_humanize_inlay_label_passes_through_non_tensor_labels_unchanged() {
        assert_eq!(humanize_inlay_label("i32", false), "i32");
        assert_eq!(
            humanize_inlay_label("Tensor<Dyn, CpuBackendImpl<Cpu>>", false),
            "Tensor<Dyn, CpuBackendImpl<Cpu>>"
        );
    }

    /// Regression test for a real, reported request: `cargo incin --explain`
    /// only printed a generic static rule for matmul errors. This verifies
    /// the parser pulls out the actual conflicting values from the trait's
    /// fixed `on_unimplemented` message — the exact scenario requested:
    /// multiplying a `(2, 4)` shape by a `(5, 6)` shape.
    #[test]
    fn test_parse_matmul_mismatch_extracts_conflicting_inner_dims() {
        let text = "Cannot matrix-multiply shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>)` with `(UInt<UInt<UInt<UTerm, B1>, B0>, B1>, UInt<UInt<UInt<UTerm, B1>, B1>, B0>)`";
        let mismatch = parse_matmul_mismatch(text).unwrap();
        assert_eq!(mismatch.lhs, vec!["2".to_string(), "4".to_string()]);
        assert_eq!(mismatch.rhs, vec!["5".to_string(), "6".to_string()]);
        assert_eq!(mismatch.lhs_inner_index, 1);
        assert_eq!(mismatch.rhs_inner_index, 0);
    }

    #[test]
    fn test_matmul_mismatch_render_points_at_conflicting_dims_and_suggests_fix() {
        let mismatch = MatMulMismatch {
            lhs: vec!["2".to_string(), "4".to_string()],
            rhs: vec!["5".to_string(), "6".to_string()],
            lhs_inner_index: 1,
            rhs_inner_index: 0,
        };
        let rendered = mismatch.render();
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.len(), 5);
        assert_eq!(lines[0], "      lhs shape = (2, 4)");
        assert_eq!(lines[2], "      rhs shape = (5, 6)");

        // The `^` marker must sit directly under the first character of the
        // conflicting element on the shape line above it, not just "close".
        let lhs_col = lines[0].find('4').unwrap();
        let rhs_col = lines[2].find('5').unwrap();
        assert_eq!(lines[1].find('^').unwrap(), lhs_col);
        assert_eq!(lines[3].find('^').unwrap(), rhs_col);
        assert_eq!(lines[1].trim(), "^ inner dim = 4");
        assert_eq!(lines[3].trim(), "^ inner dim = 5");

        assert_eq!(
            lines[4],
            "      4 \u{2260} 5 \u{2192} change the lhs inner dim from 4 to 5, or the rhs inner dim from 5 to 4."
        );
    }

    #[test]
    fn test_parse_matmul_mismatch_returns_none_when_inner_dims_already_match() {
        // lhs=(2,3), rhs=(3,4): inner dims are both 3, so this text isn't
        // actually describing an inner-dim mismatch — some other rule must
        // have failed, and inventing a bogus "3 != 3" explanation would be
        // actively misleading.
        let text = "Cannot matrix-multiply shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)` with `(UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>)`";
        assert!(parse_matmul_mismatch(text).is_none());
    }

    #[test]
    fn test_parse_matmul_mismatch_returns_none_for_unrelated_text() {
        assert!(parse_matmul_mismatch("some other diagnostic entirely").is_none());
    }

    /// Regression test for a real bug: `cargo incin --explain` passes the
    /// *entire* rendered rustc diagnostic ("error[E0277]: Cannot
    /// matrix-multiply shape `...` with `...`\n   --> file:line:col\n...",
    /// with `help`/`note` lines following), not just the message in
    /// isolation — an earlier version used `strip_prefix`, which silently
    /// returned `None` for anything but the bare message on its own.
    #[test]
    fn test_parse_matmul_mismatch_finds_message_embedded_in_full_rendered_diagnostic() {
        let rendered = "error[E0277]: Cannot matrix-multiply shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>)` with `(UInt<UInt<UInt<UTerm, B1>, B0>, B1>, UInt<UInt<UInt<UTerm, B1>, B1>, B0>)`\n   --> src/main.rs:19:24\n    |\n = help: the trait `MatMulShape<(...)>` is not implemented for `(...)`\n";
        let mismatch = parse_matmul_mismatch(rendered).unwrap();
        assert_eq!(mismatch.lhs, vec!["2".to_string(), "4".to_string()]);
        assert_eq!(mismatch.rhs, vec!["5".to_string(), "6".to_string()]);
    }

    #[test]
    fn test_parse_matmul_mismatch_handles_batched_lhs_shape() {
        let text = "Cannot matrix-multiply shape `(usize, UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>)` with `(UInt<UInt<UInt<UTerm, B1>, B0>, B1>, UInt<UInt<UInt<UTerm, B1>, B1>, B0>)`";
        let mismatch = parse_matmul_mismatch(text).unwrap();
        assert_eq!(
            mismatch.lhs,
            vec!["usize".to_string(), "2".to_string(), "4".to_string()]
        );
        assert_eq!(mismatch.lhs_inner_index, 2);
        assert_eq!(mismatch.rhs_inner_index, 0);
    }

    #[test]
    fn test_parse_concat_mismatch() {
        let text = "Cannot concatenate shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B0>)` with `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)` along axis `0`";
        let mismatch = parse_concat_mismatch(text).unwrap();
        assert_eq!(mismatch.lhs, vec!["2".to_string(), "2".to_string()]);
        assert_eq!(mismatch.rhs, vec!["2".to_string(), "3".to_string()]);
        assert_eq!(mismatch.axis, 0);
        assert_eq!(mismatch.mismatch_index, 1);
        let rendered = mismatch.render();
        assert!(rendered.contains("2 \u{2260} 3"));
    }

    #[test]
    fn test_parse_broadcast_mismatch() {
        let text = "Cannot broadcast shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B0>)` to `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)`";
        let mismatch = parse_broadcast_mismatch(text).unwrap();
        assert_eq!(mismatch.lhs, vec!["2".to_string(), "2".to_string()]);
        assert_eq!(mismatch.rhs, vec!["2".to_string(), "3".to_string()]);
        let rendered = mismatch.render();
        assert!(rendered.contains("2 \u{2260} 3"));
    }

    #[test]
    fn test_parse_reshape_mismatch() {
        let text = "Cannot reshape from `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B0>)` to `(UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>)`";
        let mismatch = parse_reshape_mismatch(text).unwrap();
        assert_eq!(mismatch.src_count, 4);
        assert_eq!(mismatch.target_count, 9);
        let rendered = mismatch.render();
        assert!(rendered.contains("4 \u{2260} 9"));
    }

    #[test]
    fn test_parse_conv2d_mismatch() {
        let text = "Cannot apply Conv2D: input shape `(UInt<UTerm, B1>, UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B0>)` is incompatible with kernel shape `(UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>, UInt<UInt<UTerm, B1>, B1>)`";
        let mismatch = parse_conv2d_mismatch(text).unwrap();
        assert_eq!(mismatch.input[1], "2");
        assert_eq!(mismatch.kernel[1], "3");
        let rendered = mismatch.render();
        assert!(rendered.contains("2 \u{2260} 3"));
    }

    #[test]
    fn test_parse_transpose_mismatch() {
        let text = "Cannot transpose dimensions `2` and `3` on shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)`";
        let mismatch = parse_transpose_mismatch(text).unwrap();
        assert_eq!(mismatch.rank, 2);
        assert_eq!(mismatch.d1, 2);
        assert_eq!(mismatch.d2, 3);
        let rendered = mismatch.render();
        assert!(rendered.contains("must be < rank (2)"));
    }

    #[test]
    fn test_parse_reduce_dim_mismatch() {
        let text = "Cannot reduce dimension `3` on shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)`";
        let mismatch = parse_reduce_dim_mismatch(text).unwrap();
        assert_eq!(mismatch.rank, 2);
        assert_eq!(mismatch.dim, 3);
        let rendered = mismatch.render();
        assert!(rendered.contains("3 \u{2265} 2"));
    }

    #[test]
    fn test_parse_flatten_mismatch() {
        let text = "Cannot flatten shape `(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>)` from dimension `1` to `3`";
        let mismatch = parse_flatten_mismatch(text).unwrap();
        assert_eq!(mismatch.rank, 2);
        assert_eq!(mismatch.start_dim, 1);
        assert_eq!(mismatch.end_dim, 3);
        let rendered = mismatch.render();
        assert!(rendered.contains("invalid range"));
    }

    #[test]
    fn test_parse_module_forward_mismatch() {
        let text = "the trait bound `MyLayer<Cpu>: Module<Tensor<(UInt<UTerm, B1>, UInt<UInt<UInt<UTerm, B1>, B0>, B0>), Cpu>>` is not satisfied\nthe following other types implement trait `Module<Input>`:\n  `MyLayer<Cpu>` implements `Module<Tensor<(UInt<UTerm, B1>, UInt<UInt<UInt<UTerm, B1>, B0>, B1>), Cpu>>`";
        let mismatch = parse_module_forward_mismatch(text).unwrap();
        assert_eq!(mismatch.actual_input, "Tensor<(1, 4), Cpu>");
        assert_eq!(mismatch.expected_input, "Tensor<(1, 5), Cpu>");
        let rendered = mismatch.render();
        assert!(rendered.contains("provided input shape = Tensor<(1, 4), Cpu>"));
    }

    #[test]
    fn test_parse_slice_mismatch() {
        let text = "Cannot slice dimension with `Slice<U1, U10, U9>` for shape `(UInt<UInt<UTerm, B1>, B0>,)`";
        let mismatch = parse_slice_mismatch(text).unwrap();
        assert_eq!(mismatch.in_shape, "(2,)");
        let rendered = mismatch.render();
        assert!(rendered.contains("invalid slice"));
    }

    #[test]
    fn test_parse_conv1d_mismatch() {
        let text = "Cannot apply 1D convolution to shape `(UInt<UInt<UTerm, B1>, B0>,)`";
        let mismatch = parse_conv1d_mismatch(text).unwrap();
        assert_eq!(mismatch.input_shape, "(2,)");
        let rendered = mismatch.render();
        assert!(rendered.contains("Conv1D requires a 2D or 3D tensor"));
    }

    #[test]
    fn test_parse_pool2d_mismatch() {
        let text = "Cannot apply 2D pooling to shape `(UInt<UInt<UTerm, B1>, B0>,)`";
        let mismatch = parse_pool2d_mismatch(text).unwrap();
        assert_eq!(mismatch.input_shape, "(2,)");
        let rendered = mismatch.render();
        assert!(rendered.contains("Pool2D requires a 3D or 4D tensor"));
    }

    #[test]
    fn test_parse_shape_eq_mismatch() {
        let text = "evaluation of constant value failed\nShape Mismatch: Attempted to operate on tensors of incompatible shapes.";
        let mismatch = parse_shape_eq_mismatch(text).unwrap();
        assert!(
            mismatch
                .message
                .contains("Attempted to operate on tensors of incompatible shapes")
        );
        let rendered = mismatch.render();
        assert!(rendered.contains("shape mismatch"));
    }

    #[test]
    fn test_parse_medium_prio_diagnostics() {
        let text_bmm = "bmm error";
        assert!(parse_bmm_mismatch(text_bmm).is_some());

        let text_unfold = "unfold size cannot exceed dimension length";
        assert!(parse_unfold_mismatch(text_unfold).is_some());

        let text_pixel = "pixel_shuffle channels must be divisible";
        assert!(parse_pixel_shuffle_mismatch(text_pixel).is_some());

        let text_group = "group_norm: channels must be divisible by groups";
        assert!(parse_group_norm_mismatch(text_group).is_some());

        let text_domain = "out of domain error";
        assert!(parse_math_domain_error(text_domain).is_some());

        let text_inplace = "in-place operand shape mismatch";
        assert!(parse_inplace_shape_mismatch(text_inplace).is_some());
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_expand_type_file_notes_reads_and_humanizes_file() {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_type_file_notes.txt");
        let sample_type =
            "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<Cpu>>";
        std::fs::write(&temp_file, sample_type).unwrap();

        let diagnostic = format!(
            "note: the full type name was written to '{}'",
            temp_file.display()
        );

        let translated = humanize_diagnostic(&diagnostic);
        assert!(translated.text.contains("📄 [Expanded Full Type]"));
        assert!(
            translated
                .text
                .contains("Tensor<[2, 3], CpuBackendImpl<Cpu>>")
        );
        assert!(
            translated
                .hints
                .contains(&("2".to_string(), "UInt<UInt<UTerm, B1>, B0>".to_string()))
        );
        assert!(
            translated
                .hints
                .contains(&("3".to_string(), "UInt<UInt<UTerm, B1>, B1>".to_string()))
        );

        let _ = std::fs::remove_file(temp_file);
    }
}
