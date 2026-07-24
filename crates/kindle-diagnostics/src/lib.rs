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
/// which is the whole readability win either way. Truly non-Kindle labels
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
/// segment: `kindle::cpu::Tensor` becomes `Tensor`, `typenum::B1` becomes
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
    // `cargo kindle --explain`) pass the *entire* rendered diagnostic —
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
        assert_eq!(strip_path_qualifiers("kindle::cpu::Tensor"), "Tensor");
        assert_eq!(strip_path_qualifiers("f32"), "f32");
        assert_eq!(
            strip_path_qualifiers(
                "kindle::cpu::Tensor<(typenum::UInt<typenum::UInt<typenum::UTerm, typenum::B1>, typenum::B1>,), kindle::cpu::CpuBackendImpl, f32>"
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
        let full_text_edit = "kindle::cpu::Tensor<(typenum::UInt<typenum::UInt<typenum::UTerm, typenum::B1>, typenum::B1>, typenum::UInt<typenum::UInt<typenum::UInt<typenum::UTerm, typenum::B1>, typenum::B0>, typenum::B1>), kindle::cpu::CpuBackendImpl, f32>";
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
            humanize_inlay_label("Tensor<Dyn, CpuBackendImpl<f32, Cpu>>", false),
            "Tensor<Dyn, CpuBackendImpl<f32, Cpu>>"
        );
    }

    /// Regression test for a real, reported request: `cargo kindle --explain`
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

    /// Regression test for a real bug: `cargo kindle --explain` passes the
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

    /// Batched shapes (`(B, M, K) x (K, N)`) shift which element is "last"
    /// on the lhs, but the rule — last-of-lhs vs second-to-last-of-rhs — is
    /// unchanged and must still be found correctly.
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
}
