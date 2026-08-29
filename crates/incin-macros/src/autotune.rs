//! Proc-macro attribute implementation for `#[autotune(...)]`.

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, ExprArray, ExprTuple, Ident, ItemFn, LitStr, Token, parse2};

#[derive(Debug)]
pub enum AutotunePolicy {
    Disabled,
    Heuristic,
    Warmup,
    Profile,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct AutotuneArgs {
    pub key: String,
    pub key_span: Span,
    pub params: Vec<Expr>,
    pub params_span: Span,
    pub policy: AutotunePolicy,
    pub policy_span: Span,
}

impl Parse for AutotuneArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut key = None;
        let mut key_span = Span::call_site();
        let mut params = None;
        let mut params_span = Span::call_site();
        let mut policy = None;
        let mut policy_span = Span::call_site();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match ident.to_string().as_str() {
                "key" => {
                    let lit: LitStr = input.parse()?;
                    if lit.value().trim().is_empty() {
                        return Err(syn::Error::new_spanned(
                            lit,
                            "autotune key must not be empty",
                        ));
                    }
                    key_span = lit.span();
                    key = Some(lit.value());
                }
                "params" => {
                    params_span = input.span();
                    let expr: Expr = input.parse()?;
                    match &expr {
                        Expr::Array(ExprArray { elems, .. }) => {
                            if elems.is_empty() {
                                return Err(syn::Error::new_spanned(
                                    &expr,
                                    "autotune params must not be empty",
                                ));
                            }
                            params = Some(elems.iter().cloned().collect());
                        }
                        Expr::Tuple(ExprTuple { elems, .. }) => {
                            if elems.is_empty() {
                                return Err(syn::Error::new_spanned(
                                    &expr,
                                    "autotune params must not be empty",
                                ));
                            }
                            params = Some(elems.iter().cloned().collect());
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &expr,
                                "autotune params must be an array or tuple of candidates",
                            ));
                        }
                    }
                }
                "policy" => {
                    policy_span = input.span();
                    let policy_ident: Ident = input.parse()?;
                    let pol = match policy_ident.to_string().as_str() {
                        "disabled" => AutotunePolicy::Disabled,
                        "heuristic" => AutotunePolicy::Heuristic,
                        "warmup" => AutotunePolicy::Warmup,
                        "profile" => AutotunePolicy::Profile,
                        _ => {
                            return Err(syn::Error::new_spanned(
                                policy_ident,
                                "invalid autotune policy: expected disabled, heuristic, warmup, or profile",
                            ));
                        }
                    };
                    policy = Some(pol);
                }
                other => {
                    return Err(syn::Error::new_spanned(
                        ident,
                        format!("unknown autotune attribute parameter: `{other}`"),
                    ));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        let key = key.ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                "missing required `key` in #[autotune(...)]",
            )
        })?;
        let params = params.ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                "missing required `params` in #[autotune(...)]",
            )
        })?;
        let policy = policy.unwrap_or(AutotunePolicy::Heuristic);

        Ok(Self {
            key,
            key_span,
            params,
            params_span,
            policy,
            policy_span,
        })
    }
}

pub fn expand_autotune(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = match parse2::<AutotuneArgs>(attr) {
        Ok(a) => a,
        Err(err) => return err.to_compile_error(),
    };

    let function = match parse2::<ItemFn>(item) {
        Ok(f) => f,
        Err(err) => return err.to_compile_error(),
    };

    let key = &args.key;
    let params = &args.params;
    let param_count = params.len();

    let fn_vis = &function.vis;
    let fn_sig = &function.sig;
    let fn_block = &function.block;

    // Generate wrapped function with static candidate selection metadata
    quote! {
        #fn_vis #fn_sig {
            // Autotune registration metadata
            const _AUTOTUNE_KEY: &'static str = #key;
            const _AUTOTUNE_NUM_CANDIDATES: usize = #param_count;

            #fn_block
        }
    }
}
