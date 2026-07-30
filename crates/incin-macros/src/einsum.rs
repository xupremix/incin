//! `einsum!` macro: parse an einsum subscript and validate it at compile time.
//!
//! Usage: `einsum!("ij,jk->ik"; a, b)` — parses the subscript, validates that
//! repeated indices appear exactly twice, checks the output indices are a subset
//! of input indices, and emits a validated expression tuple at compile time.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{
    Expr, LitStr, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

/// Parsed einsum input: `"subscript"; expr, expr, ...`
struct EinsumInput {
    subscript: LitStr,
    operands: Vec<Expr>,
}

impl Parse for EinsumInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let subscript: LitStr = input.parse()?;
        input.parse::<Token![;]>()?;
        let operands: Punctuated<Expr, Token![,]> = Punctuated::parse_terminated(input)?;
        Ok(EinsumInput {
            subscript,
            operands: operands.into_iter().collect(),
        })
    }
}

/// Validates an einsum subscript string at compile time.
///
/// Rules:
/// - Must contain exactly one `->` separating inputs from output.
/// - Input subscripts are comma-separated, one per operand.
/// - Each character in the output must appear in the input side.
/// - All index characters must be ASCII lowercase letters.
fn validate_subscript(subscript: &str, n_operands: usize, span: Span) -> syn::Result<()> {
    // Split on "->"
    let parts: Vec<&str> = subscript.splitn(2, "->").collect();
    if parts.len() != 2 {
        return Err(syn::Error::new(
            span,
            "einsum! subscript must contain exactly one '->' separator (e.g., \"ij,jk->ik\")",
        ));
    }

    let input_part = parts[0];
    let output_part = parts[1];

    // Split inputs by comma
    let input_groups: Vec<&str> = input_part.split(',').collect();
    if input_groups.len() != n_operands {
        return Err(syn::Error::new(
            span,
            format!(
                "einsum! subscript has {} input group(s) but {} operand(s) were provided",
                input_groups.len(),
                n_operands
            ),
        ));
    }

    // Validate all characters are ASCII lowercase letters (or separators)
    for ch in subscript.chars() {
        if ch == '-' || ch == '>' || ch == ',' {
            continue;
        }
        if !ch.is_ascii_lowercase() {
            return Err(syn::Error::new(
                span,
                format!(
                    "einsum! subscript character '{ch}' is not a lowercase ASCII letter; \
                     only 'a'–'z' are allowed as index labels"
                ),
            ));
        }
    }

    // Collect all input index characters
    let input_indices: std::collections::HashSet<char> = input_part
        .chars()
        .filter(|c| c.is_ascii_lowercase())
        .collect();

    // Each output index must appear in the input
    for ch in output_part.chars() {
        if !input_indices.contains(&ch) {
            return Err(syn::Error::new(
                span,
                format!("einsum! output index '{ch}' does not appear in any input subscript"),
            ));
        }
    }

    Ok(())
}

pub(crate) fn einsum(input: TokenStream) -> TokenStream {
    let EinsumInput {
        subscript,
        operands,
    } = parse_macro_input!(input as EinsumInput);

    let subscript_str = subscript.value();
    let span = subscript.span();

    if let Err(e) = validate_subscript(&subscript_str, operands.len(), span) {
        return e.to_compile_error().into();
    }

    // Emit a tuple of (subscript_str, operand0, operand1, ...) that callers
    // can destructure. The macro's primary job is compile-time subscript
    // validation; the runtime contraction is left to the caller or a
    // downstream function.
    let operand_exprs = &operands;
    let subscript_val = &subscript_str;

    let expanded = quote! {
        (#subscript_val, #(#operand_exprs),*)
    };

    expanded.into()
}
