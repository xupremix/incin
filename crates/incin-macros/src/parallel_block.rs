//! `parallel!` block macro: scoped execution block with optional mesh context.

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Block, Expr, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

/// Input structure for `parallel!`:
/// Form 1: `parallel!(mesh => { ... })`
/// Form 2: `parallel!({ ... })`
enum ParallelBlockInput {
    WithMesh { mesh: Expr, body: Block },
    Simple { body: Block },
}

impl Parse for ParallelBlockInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek2(Token![=>]) {
            let mesh: Expr = input.parse()?;
            input.parse::<Token![=>]>()?;
            let body: Block = input.parse()?;
            Ok(ParallelBlockInput::WithMesh { mesh, body })
        } else {
            let body: Block = input.parse()?;
            Ok(ParallelBlockInput::Simple { body })
        }
    }
}

pub(crate) fn parallel_block(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as ParallelBlockInput);

    let expanded = match parsed {
        ParallelBlockInput::WithMesh { mesh, body } => {
            quote! {
                {
                    let _mesh_ctx = &#mesh;
                    #body
                }
            }
        }
        ParallelBlockInput::Simple { body } => {
            quote! {
                #body
            }
        }
    };

    expanded.into()
}
