use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Expr, RangeLimits, Token, parse::Parse, parse::ParseStream, parse_macro_input,
    punctuated::Punctuated,
};

/// Idx list.
struct IdxList {
    items: Punctuated<Expr, Token![,]>,
}

impl Parse for IdxList {
    /// Parse.
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(IdxList {
            items: Punctuated::parse_terminated(input)?,
        })
    }
}

pub(crate) fn idx(input: TokenStream) -> TokenStream {
    let items = parse_macro_input!(input as IdxList);
    let mut output = Vec::new();

    for item in items.items {
        output.push(match item {
            Expr::Lit(expr_lit) => {
                if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                    let val: usize = match lit_int.base10_parse() {
                        Ok(v) => v,
                        Err(e) => {
                            return syn::Error::new_spanned(
                                lit_int,
                                format!("Invalid integer: {}", e),
                            )
                            .to_compile_error()
                            .into();
                        }
                    };
                    let path = quote! { ::incin::advanced:: };
                    crate::shape::lit_to_typenum(val, &path)
                } else {
                    return syn::Error::new_spanned(
                        &expr_lit,
                        "idx! only supports integers, identifiers, -1, and ranges",
                    )
                    .to_compile_error()
                    .into();
                }
            }
            Expr::Unary(expr_unary) => {
                if let syn::UnOp::Neg(_) = expr_unary.op {
                    if let Expr::Lit(expr_lit) = &*expr_unary.expr {
                        if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                            let val: i64 = match lit_int.base10_parse() {
                                Ok(v) => v,
                                Err(e) => {
                                    return syn::Error::new_spanned(
                                        lit_int,
                                        format!("Invalid integer: {}", e),
                                    )
                                    .to_compile_error()
                                    .into();
                                }
                            };
                            if val == 1 {
                                quote! { ::incin::advanced::InferDim }
                            } else {
                                return syn::Error::new_spanned(
                                    lit_int,
                                    "idx! only supports -1 for negative numbers",
                                )
                                .to_compile_error()
                                .into();
                            }
                        } else {
                            return syn::Error::new_spanned(expr_lit, "idx! only supports -1")
                                .to_compile_error()
                                .into();
                        }
                    } else {
                        return syn::Error::new_spanned(&expr_unary.expr, "idx! only supports -1")
                            .to_compile_error()
                            .into();
                    }
                } else {
                    return syn::Error::new_spanned(&expr_unary, "idx! only supports -1")
                        .to_compile_error()
                        .into();
                }
            }
            Expr::Path(expr_path) => {
                if let Some(ident) = expr_path.path.get_ident() {
                    quote! { ::incin::advanced::NamedDim<#ident, usize> }
                } else {
                    return syn::Error::new_spanned(
                        &expr_path,
                        "idx! expects simple identifiers for named runtime axes",
                    )
                    .to_compile_error()
                    .into();
                }
            }
            Expr::Range(expr_range) => match (&expr_range.start, &expr_range.end) {
                (None, None) => {
                    quote! { ::incin::advanced::Ellipsis }
                }
                (Some(start), Some(end)) => {
                    let start_val = if let Expr::Lit(expr_lit) = &**start {
                        if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                            match lit_int.base10_parse::<usize>() {
                                Ok(v) => v,
                                Err(_) => {
                                    return syn::Error::new_spanned(
                                        lit_int,
                                        "idx! range requires integers",
                                    )
                                    .to_compile_error()
                                    .into();
                                }
                            }
                        } else {
                            return syn::Error::new_spanned(
                                expr_lit,
                                "idx! range requires integers",
                            )
                            .to_compile_error()
                            .into();
                        }
                    } else {
                        return syn::Error::new_spanned(&**start, "idx! range requires integers")
                            .to_compile_error()
                            .into();
                    };

                    let mut end_val = if let Expr::Lit(expr_lit) = &**end {
                        if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                            match lit_int.base10_parse::<usize>() {
                                Ok(v) => v,
                                Err(_) => {
                                    return syn::Error::new_spanned(
                                        lit_int,
                                        "idx! range requires integers",
                                    )
                                    .to_compile_error()
                                    .into();
                                }
                            }
                        } else {
                            return syn::Error::new_spanned(
                                expr_lit,
                                "idx! range requires integers",
                            )
                            .to_compile_error()
                            .into();
                        }
                    } else {
                        return syn::Error::new_spanned(&**end, "idx! range requires integers")
                            .to_compile_error()
                            .into();
                    };

                    if let RangeLimits::Closed(_) = expr_range.limits {
                        end_val += 1;
                    }

                    if start_val > end_val {
                        return syn::Error::new_spanned(
                            &expr_range,
                            "idx! range start cannot be greater than end",
                        )
                        .to_compile_error()
                        .into();
                    }

                    let diff = end_val - start_val;
                    let path = quote! { ::incin::advanced:: };
                    let start_type = crate::shape::lit_to_typenum(start_val, &path);
                    let end_type = crate::shape::lit_to_typenum(end_val, &path);
                    let diff_type = crate::shape::lit_to_typenum(diff, &path);
                    quote! { ::incin::advanced::Slice<#start_type, #end_type, #diff_type> }
                }
                _ => {
                    return syn::Error::new_spanned(
                        &expr_range,
                        "idx! currently only supports `..` or `start..end`",
                    )
                    .to_compile_error()
                    .into();
                }
            },
            _ => {
                return syn::Error::new_spanned(&item, "Unsupported token in idx! macro")
                    .to_compile_error()
                    .into();
            }
        });
    }

    // Keep indexing and reshape targets on the same canonical structural
    // engine as tensor shapes.  The macro deliberately hides the cons list;
    // callers still write the compact `idx![...]` syntax.
    let mut target = quote! { ::incin::types::Nil };
    for item in output.into_iter().rev() {
        target = quote! { ::incin::types::DimCons<#item, #target> };
    }
    target.into()
}
