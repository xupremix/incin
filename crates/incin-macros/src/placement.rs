use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Ident, Token, parenthesized,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

enum PlacementAst {
    Local,
    Replicated { mesh: syn::Type },
    Sharded { axis: syn::LitInt, mesh: syn::Type },
    Partial { reduction: Ident, mesh: syn::Type },
    PipelineStage { stage: syn::LitInt, mesh: syn::Type },
}

struct PlacementInput {
    internal: bool,
    placement: PlacementAst,
}

fn parse_on_mesh(input: ParseStream) -> syn::Result<syn::Type> {
    let lookahead: Ident = input
        .parse()
        .map_err(|_| syn::Error::new(input.span(), "distributed placement requires `on <Mesh>`"))?;
    if lookahead != "on" {
        return Err(syn::Error::new(
            lookahead.span(),
            format!("expected `on <Mesh>`, found `{lookahead}`"),
        ));
    }
    input.parse::<syn::Type>()
}

impl Parse for PlacementInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut internal = false;
        if input.peek(Token![@]) {
            input.parse::<Token![@]>()?;
            internal = true;
        }

        let kind: Ident = input.parse()?;
        let kind_str = kind.to_string();

        let placement = match kind_str.as_str() {
            "Local" => PlacementAst::Local,
            "Replicated" => {
                let mesh = parse_on_mesh(input)?;
                PlacementAst::Replicated { mesh }
            }
            "Sharded" => {
                let content;
                parenthesized!(content in input);
                let axis: syn::LitInt = content.parse()?;
                let mesh = parse_on_mesh(input)?;
                PlacementAst::Sharded { axis, mesh }
            }
            "Partial" => {
                let content;
                parenthesized!(content in input);
                let reduction: Ident = content.parse()?;
                let red_str = reduction.to_string();
                if !matches!(red_str.as_str(), "Sum" | "Mean" | "Max" | "Min" | "Prod") {
                    return Err(syn::Error::new(
                        reduction.span(),
                        format!(
                            "invalid reduction `{red_str}`; valid reductions are `Sum`, `Mean`, `Max`, `Min`, `Prod`"
                        ),
                    ));
                }
                let mesh = parse_on_mesh(input)?;
                PlacementAst::Partial { reduction, mesh }
            }
            "PipelineStage" => {
                let content;
                parenthesized!(content in input);
                let stage: syn::LitInt = content.parse()?;
                let mesh = parse_on_mesh(input)?;
                PlacementAst::PipelineStage { stage, mesh }
            }
            other => {
                return Err(syn::Error::new(
                    kind.span(),
                    format!(
                        "unknown placement `{other}`; valid placements are `Local`, `Replicated`, `Sharded`, `Partial`, and `PipelineStage`"
                    ),
                ));
            }
        };

        Ok(PlacementInput {
            internal,
            placement,
        })
    }
}

pub(crate) fn placement(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as PlacementInput);
    let path = if parsed.internal {
        quote! { crate:: }
    } else {
        quote! { ::incin_core:: }
    };

    match parsed.placement {
        PlacementAst::Local => quote! { #path dist::Local },
        PlacementAst::Replicated { mesh } => quote! { #path dist::Replicated<#mesh> },
        PlacementAst::Sharded { axis, mesh } => {
            let axis_val: usize = axis.base10_parse().unwrap_or(0);
            let axis_typenum = crate::shape::lit_to_typenum(axis_val, &path);
            quote! { #path dist::Sharded<#mesh, #axis_typenum> }
        }
        PlacementAst::Partial { reduction, mesh } => {
            quote! { #path dist::Partial<#mesh, #path dist::#reduction> }
        }
        PlacementAst::PipelineStage { stage, mesh } => {
            quote! { #path dist::PipelineStage<#mesh, #stage> }
        }
    }
    .into()
}
