use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{LitInt, parse_macro_input};

/// Max rank.
struct MaxRank(usize);

impl Parse for MaxRank {
    /// Parse.
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lit: LitInt = input.parse()?;
        Ok(MaxRank(lit.base10_parse()?))
    }
}

/// Impl arg into.
pub fn impl_arg_into(input: TokenStream) -> TokenStream {
    let MaxRank(max_rank) = parse_macro_input!(input as MaxRank);
    let mut output = quote!();

    // Loop over all ranks from 1 to MAX
    for rank in 1..=max_rank {
        // Loop over permutations (mask).
        // 0 = Dynamic, 1 = Static.
        // We SKIP mask 0 (all dynamic) to avoid conflict with blanket impls like `impl<T> ArgInto<T> for T`.
        let num_permutations = 1 << rank;
        for mask in 1..num_permutations {
            let mut impl_generics = Vec::new();
            let mut target_types = Vec::new();
            let mut source_types = Vec::new();

            let mut tuple_body_items = Vec::new();
            let mut scalar_body_items = Vec::new();

            let _const_counter = 1u32;
            let mut dyn_counter = 0u32;

            for i in 0..rank {
                let is_const = (mask >> i) & 1 == 1;

                if is_const {
                    // Static: ()
                    target_types.push(quote! { () });

                    // Body inserts the unit constructor '()'
                    tuple_body_items.push(quote! { () });
                    scalar_body_items.push(quote! { () });
                } else {
                    // Dynamic: Generic Dim
                    // Generates: D0: Dim
                    let dim_name = format_ident!("D{}", dyn_counter);
                    let idx = syn::Index::from(dyn_counter as usize);
                    dyn_counter += 1;

                    impl_generics.push(quote! { #dim_name: Dim });
                    target_types.push(quote! { usize });
                    source_types.push(quote! { #dim_name });

                    // Body converts the dim to usize using .size()
                    tuple_body_items.push(quote! { self.#idx.size() });
                    scalar_body_items.push(quote! { self.size() });
                }
            }

            // 1. Tuple Implementation
            // e.g. impl<D0: Dim, const N1: usize> ArgInto<(usize, Const<N1>)> for (D0,)
            output.extend(quote! {
                impl<#(#impl_generics),*> ArgInto<(#(#target_types,)*)> for (#(#source_types,)*) {
                    /// Into arg.
                    fn into_arg(self) -> (#(#target_types,)*) {
                        (#(#tuple_body_items,)*)
                    }
                }
            });

            // 2. Scalar Implementation (if exactly 1 dynamic arg)
            // e.g. impl<D0: Dim, const N1: usize> ArgInto<(usize, Const<N1>)> for D0
            if source_types.len() == 1 {
                // For scalar impl, source is just the type D0 (no tuple), body uses scalar_body_items
                let single_source = &source_types[0];
                output.extend(quote! {
                    impl<#(#impl_generics),*> ArgInto<(#(#target_types,)*)> for #single_source {
                        /// Into arg.
                        fn into_arg(self) -> (#(#target_types,)*) {
                            (#(#scalar_body_items,)*)
                        }
                    }
                });
            }
        }
    }

    output.into()
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
