use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Expr, Token, parse::Parse, parse::ParseStream, parse_macro_input, punctuated::Punctuated, RangeLimits
};

struct IdxList {
    items: Punctuated<Expr, Token![,]>,
}

impl Parse for IdxList {
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
                    let val: usize = lit_int.base10_parse().unwrap();
                    let path = quote! { kindle::prelude:: };
                    crate::shape::lit_to_typenum(val, &path)
                } else {
                    panic!("idx! only supports integers, identifiers, -1, and ranges");
                }
            }
            Expr::Unary(expr_unary) => {
                if let syn::UnOp::Neg(_) = expr_unary.op {
                    if let Expr::Lit(expr_lit) = &*expr_unary.expr {
                        if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                            let val: i64 = lit_int.base10_parse().unwrap();
                            if val == 1 {
                                quote! { kindle::shapes::InferDim }
                            } else {
                                panic!("idx! only supports -1 for negative numbers");
                            }
                        } else {
                            panic!("idx! only supports -1");
                        }
                    } else {
                        panic!("idx! only supports -1");
                    }
                } else {
                    panic!("idx! only supports -1");
                }
            }
            Expr::Path(expr_path) => {
                if let Some(ident) = expr_path.path.get_ident() {
                    quote! { kindle::shapes::NamedDyn<#ident> }
                } else {
                    panic!("idx! expects simple identifiers for NamedDyn");
                }
            }
            Expr::Range(expr_range) => {
                match (&expr_range.start, &expr_range.end) {
                    (None, None) => {
                        quote! { kindle::shapes::Ellipsis }
                    }
                    (Some(start), Some(end)) => {
                        // We only support integer literals for slice ranges
                        let start_val = if let Expr::Lit(expr_lit) = &**start {
                            if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                                lit_int.base10_parse::<usize>().unwrap()
                            } else { panic!("idx! range requires integers") }
                        } else { panic!("idx! range requires integers") };
                        
                        let mut end_val = if let Expr::Lit(expr_lit) = &**end {
                            if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                                lit_int.base10_parse::<usize>().unwrap()
                            } else { panic!("idx! range requires integers") }
                        } else { panic!("idx! range requires integers") };
                        
                        if let RangeLimits::Closed(_) = expr_range.limits {
                            end_val += 1;
                        }
                        
                        
                        let diff = end_val - start_val;
                        let path = quote! { kindle::prelude:: };
                        let start_type = crate::shape::lit_to_typenum(start_val, &path);
                        let end_type = crate::shape::lit_to_typenum(end_val, &path);
                        let diff_type = crate::shape::lit_to_typenum(diff, &path);
                        quote! { kindle::shapes::Slice<#start_type, #end_type, #diff_type> }
    
                    }
                    _ => panic!("idx! currently only supports `..` or `start..end`"),
                }
            }
            _ => panic!("Unsupported token in idx! macro"),
        });
    }

    if output.len() == 1 {
        let first = &output[0];
        quote! { (#first,) }.into()
    } else {
        quote! { (#(#output),*) }.into()
    }
}
