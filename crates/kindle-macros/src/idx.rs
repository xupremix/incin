use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Expr, RangeLimits, Token, parse::Parse, parse::ParseStream, parse_macro_input,
    punctuated::Punctuated,
};

/// Core abstraction for `IdxList` within the Kindle framework.
struct IdxList {
    items: Punctuated<Expr, Token![,]>,
}

impl Parse for IdxList {
    /// Core abstraction for `parse` within the Kindle framework.
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
                    let path = quote! { kindle::prelude:: };
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
                                quote! { kindle::prelude::InferDim }
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
                    quote! { kindle::prelude::NamedDyn<#ident> }
                } else {
                    return syn::Error::new_spanned(
                        &expr_path,
                        "idx! expects simple identifiers for NamedDyn",
                    )
                    .to_compile_error()
                    .into();
                }
            }
            Expr::Range(expr_range) => match (&expr_range.start, &expr_range.end) {
                (None, None) => {
                    quote! { kindle::prelude::Ellipsis }
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
                    let path = quote! { kindle::prelude:: };
                    let start_type = crate::shape::lit_to_typenum(start_val, &path);
                    let end_type = crate::shape::lit_to_typenum(end_val, &path);
                    let diff_type = crate::shape::lit_to_typenum(diff, &path);
                    quote! { kindle::prelude::Slice<#start_type, #end_type, #diff_type> }
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

    if output.len() == 1 {
        let first = &output[0];
        quote! { (#first,) }.into()
    } else {
        quote! { (#(#output),*) }.into()
    }
}
