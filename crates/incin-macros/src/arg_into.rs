use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{LitInt, parse_macro_input};

/// Maximum arity covered by the ergonomic mixed-shape argument adapter.
///
/// This is an argument-conversion convenience boundary, not a shape/rank
/// representability limit. Canonical structural shapes and ShapeBuf remain
/// arbitrary-rank.
struct MaxRank(usize);

impl Parse for MaxRank {
    /// Parse.
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lit: LitInt = input.parse()?;
        Ok(MaxRank(lit.base10_parse()?))
    }
}

/// Generates exact compressed layer-argument tuple projections.
pub fn impl_layer_args(input: TokenStream) -> TokenStream {
    let MaxRank(max_rank) = parse_macro_input!(input as MaxRank);
    let mut output = quote!();

    for rank in 1..=max_rank {
        for mask in 0..(1usize << rank) {
            let mut generics = Vec::new();
            let mut target_types = Vec::new();
            let mut source_types = Vec::new();
            let mut body = Vec::new();
            let mut dynamic_index = 0usize;

            for position in 0..rank {
                if (mask >> position) & 1 == 1 {
                    target_types.push(quote! { () });
                    body.push(quote! { () });
                } else {
                    let ty = format_ident!("T{}", dynamic_index);
                    let index = syn::Index::from(dynamic_index);
                    dynamic_index += 1;
                    generics.push(quote! { #ty: NotUnit });
                    target_types.push(quote! { #ty });
                    source_types.push(quote! { #ty });
                    body.push(quote! { self.#index });
                }
            }

            if source_types.is_empty() {
                output.extend(quote! {
                    impl LayerArgInto<(#(#target_types,)*)> for () {
                        fn into_layer_arg(self) -> (#(#target_types,)*) {
                            (#(#body,)*)
                        }
                    }
                });
            } else {
                output.extend(quote! {
                    impl<#(#generics),*> LayerArgInto<(#(#target_types,)*)>
                        for (#(#source_types,)*)
                    {
                        fn into_layer_arg(self) -> (#(#target_types,)*) {
                            (#(#body,)*)
                        }
                    }
                });
                if source_types.len() == 1 {
                    let source = &source_types[0];
                    let mut scalar_body = Vec::new();
                    for position in 0..rank {
                        if (mask >> position) & 1 == 1 {
                            scalar_body.push(quote! { () });
                        } else {
                            scalar_body.push(quote! { self });
                        }
                    }
                    output.extend(quote! {
                        impl<#(#generics),*> LayerArgInto<(#(#target_types,)*)> for #source {
                            fn into_layer_arg(self) -> (#(#target_types,)*) {
                                (#(#scalar_body,)*)
                            }
                        }
                    });
                }
            }
        }
    }

    output.into()
}
