//! Low-level typenum expression parsing and text substitution: bracket
//! matching, splitting `DimCons` chains at top-level commas, and the
//! `translate_typenum_text`/`collapse_dimcons_chains` passes that
//! `humanize` builds its public API on top of.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Returns the byte offset (relative to `s`) of the `close` bracket that
/// matches the `open` bracket at the start of `s`, accounting for nesting.
pub(crate) fn matching_bracket(s: &str, open: char, close: char) -> Option<usize> {
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

/// Splits `s` on top-level (bracket-depth-0) commas.
///
/// `DimCons`'s head can itself be a bracketed type (`NamedDim<Batch, 4>`),
/// so a plain `str::split(',')` would cut the wrong argument in two. This
/// mirrors `matching_bracket`'s depth tracking, generalized to `<`/`>` only
/// (the sole nesting delimiter a `DimCons` chain's arguments ever carry).
fn split_top_level_comma(s: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    for (i, ch) in s.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => return Some((&s[..i], &s[i + 1..])),
            _ => {}
        }
    }
    None
}

/// Parses a `DimCons<H, DimCons<H2, ... DimCons<Hn, Nil>...>>` chain into
/// `[H, H2, ..., Hn]`'s inner elements, or `None` if `s` (the content
/// between `DimCons`'s own `<` and its matching `>`) isn't exactly that
/// shape. Fails closed on anything else — a shape list that only sometimes
/// collapses because a rare case was silently dropped would be worse than
/// one that never collapses in that case at all.
fn parse_dimcons_elements(s: &str) -> Option<alloc::vec::Vec<String>> {
    let (head, tail) = split_top_level_comma(s)?;
    let head = head.trim();
    let tail = tail.trim();
    if tail == "Nil" {
        return Some(alloc::vec![head.to_string()]);
    }
    // `matching_bracket` expects its argument to start AT the open bracket
    // (see its use elsewhere in this file against `label[tuple_open..]`),
    // so only `DimCons` is stripped here, leaving the `<` in place.
    let rest = tail.strip_prefix("DimCons")?;
    let close = matching_bracket(rest, '<', '>')?;
    // The tail must be nothing but the nested DimCons: a `DimCons<...>Rest`
    // with trailing content is not a cons list this parser understands.
    if rest[close + 1..].trim() != "" {
        return None;
    }
    let mut elements = alloc::vec![head.to_string()];
    elements.extend(parse_dimcons_elements(&rest[1..close])?);
    Some(elements)
}

/// Rewrites every `DimCons<H, DimCons<H2, ... DimCons<Hn, Nil>...>>` chain
/// in `text` into `[H, H2, ..., Hn]` — the same bracket-list rendering
/// [`crate::humanize_type_signature`] already gives a `Tensor<(...)>` shape tuple,
/// extended to the cons-list shape encoding that shows up bare in trait-bound
/// diagnostics (`MatMulShape<DimCons<...>>` and similar), which never passes
/// through the `Tensor<(` special case at all.
///
/// Intended to run after [`translate_typenum_text`], so a chain's heads are
/// already plain decimals (`DimCons<4, DimCons<8, Nil>>`) rather than raw
/// `UInt<...>` walls by the time this collapses the shell around them —
/// nothing here depends on that order, but the numbers read best that way.
pub fn collapse_dimcons_chains(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;
    let mut search_idx = 0;

    while let Some(rel_start) = text[search_idx..].find("DimCons<") {
        let start = search_idx + rel_start;
        let open = start + "DimCons".len(); // index of this DimCons's own '<'
        let Some(close_rel) = matching_bracket(&text[open..], '<', '>') else {
            search_idx = start + "DimCons<".len();
            continue;
        };
        let close = open + close_rel;
        let inner = &text[open + 1..close];

        match parse_dimcons_elements(inner) {
            Some(elements) => {
                result.push_str(&text[last_end..start]);
                result.push('[');
                for (i, element) in elements.iter().enumerate() {
                    if i > 0 {
                        result.push_str(", ");
                    }
                    result.push_str(element);
                }
                result.push(']');
                last_end = close + 1;
                search_idx = close + 1;
            }
            None => {
                search_idx = start + "DimCons<".len();
            }
        }
    }
    result.push_str(&text[last_end..]);
    result
}
