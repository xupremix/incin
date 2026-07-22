use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

/// Dim.
enum Dim {
    /// Dyn.
    Dyn,
    /// Lit.
    Lit(syn::LitInt),
    /// Path.
    Path(syn::Path),
}

impl Parse for Dim {
    /// Parse.
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Token![dyn]) {
            input.parse::<Token![dyn]>()?;
            return Ok(Dim::Dyn);
        }
        if input.peek(Token![_]) {
            input.parse::<Token![_]>()?;
            return Ok(Dim::Dyn);
        }
        if input.peek(syn::LitInt) {
            return Ok(Dim::Lit(input.parse::<syn::LitInt>()?));
        }

        Ok(Dim::Path(input.parse::<syn::Path>()?))
    }
}

enum ShapeInput {
    List(Vec<Dim>),
    Repeat { dim: Dim, count: usize },
    Tail(Vec<Dim>),
    Head(Vec<Dim>),
    Span { head: Vec<Dim>, tail: Vec<Dim> },
}

/// Number list.
struct NumberList {
    internal: bool,
    input: ShapeInput,
}

impl Parse for NumberList {
    /// Parse.
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut internal = false;
        if input.peek(Token![@]) {
            input.parse::<Token![@]>()?;
            internal = true;
        }

        if !input.peek(Token![..]) {
            let fork = input.fork();
            if fork.parse::<Dim>().is_ok() && fork.peek(Token![;]) {
                let dim: Dim = input.parse()?;
                input.parse::<Token![;]>()?;
                let count_lit: syn::LitInt = input.parse()?;
                let count: usize = count_lit.base10_parse()?;
                return Ok(NumberList {
                    internal,
                    input: ShapeInput::Repeat { dim, count },
                });
            }
        }

        let mut before_dotdot: Vec<Dim> = Vec::new();
        let mut after_dotdot: Vec<Dim> = Vec::new();
        let mut has_dotdot = false;

        while !input.is_empty() {
            if input.peek(Token![..]) {
                if has_dotdot {
                    return Err(syn::Error::new(
                        input.span(),
                        "Only a single '..' ellipsis is permitted in a shape",
                    ));
                }
                input.parse::<Token![..]>()?;
                has_dotdot = true;
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }

            let dim: Dim = input.parse()?;
            if has_dotdot {
                after_dotdot.push(dim);
            } else {
                before_dotdot.push(dim);
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        if has_dotdot {
            if before_dotdot.is_empty() && !after_dotdot.is_empty() {
                Ok(NumberList {
                    internal,
                    input: ShapeInput::Tail(after_dotdot),
                })
            } else if !before_dotdot.is_empty() && after_dotdot.is_empty() {
                Ok(NumberList {
                    internal,
                    input: ShapeInput::Head(before_dotdot),
                })
            } else if !before_dotdot.is_empty() && !after_dotdot.is_empty() {
                Ok(NumberList {
                    internal,
                    input: ShapeInput::Span {
                        head: before_dotdot,
                        tail: after_dotdot,
                    },
                })
            } else {
                Ok(NumberList {
                    internal,
                    input: ShapeInput::Tail(Vec::new()),
                })
            }
        } else {
            Ok(NumberList {
                internal,
                input: ShapeInput::List(before_dotdot),
            })
        }
    }
}

pub(crate) fn lit_to_typenum(
    n: usize,
    path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    if n == 0 {
        return quote! { #path typenum::UTerm };
    }
    let bit = if n.is_multiple_of(2) {
        quote! { #path typenum::B0 }
    } else {
        quote! { #path typenum::B1 }
    };
    let rest = lit_to_typenum(n / 2, path);
    quote! { #path typenum::UInt<#rest, #bit> }
}

pub(crate) fn shape(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as NumberList);
    let internal = parsed.internal;
    let path = if internal {
        quote! { crate::prelude:: }
    } else {
        quote! { kindle::prelude:: }
    };

    let render_dim = |elem: &Dim| -> proc_macro2::TokenStream {
        match elem {
            Dim::Dyn => quote! { usize },
            Dim::Lit(lit_int) => {
                let val: usize = lit_int.base10_parse().unwrap_or(0);
                lit_to_typenum(val, &path)
            }
            Dim::Path(p) => quote! { #p },
        }
    };

    match parsed.input {
        ShapeInput::List(list) => {
            let output: Vec<_> = list.iter().map(render_dim).collect();
            quote! {
                ( #(#output,)* )
            }
        }
        ShapeInput::Repeat { dim, count } => {
            let rendered = render_dim(&dim);
            let output: Vec<_> = (0..count).map(|_| rendered.clone()).collect();
            quote! {
                ( #(#output,)* )
            }
        }
        ShapeInput::Tail(list) => {
            let output: Vec<_> = list.iter().map(render_dim).collect();
            quote! {
                #path TailShape<( #(#output,)* )>
            }
        }
        ShapeInput::Head(list) => {
            let output: Vec<_> = list.iter().map(render_dim).collect();
            quote! {
                #path HeadShape<( #(#output,)* )>
            }
        }
        ShapeInput::Span { head, tail } => {
            let head_output: Vec<_> = head.iter().map(render_dim).collect();
            let tail_output: Vec<_> = tail.iter().map(render_dim).collect();
            quote! {
                #path SpanShape<( #(#head_output,)* ), ( #(#tail_output,)* )>
            }
        }
    }
    .into()
}
