use proc_macro::TokenStream;
use quote::{format_ident, quote};

/// Generate shape ops.
pub fn generate_shape_ops(_input: TokenStream) -> TokenStream {
    let mut tokens = proc_macro2::TokenStream::new();
    let max_rank = 8;

    // 1. Generate Transpose<D1, D2>
    for rank in 1..=max_rank {
        for d1 in 0..rank {
            for d2 in 0..rank {
                let in_types: Vec<_> = (0..rank).map(|i| format_ident!("D{}", i)).collect();
                let mut out_dims = in_types.clone();
                out_dims.swap(d1, d2);

                tokens.extend(quote! {
                    impl<#(#in_types: crate::prelude::Dim,)*> crate::shapes::shape_ops::Transpose<#d1, #d2> for (#(#in_types,)*) {
                        /// Output.
                        type Output = (#(#out_dims,)*);
                    }
                });
            }
        }
    }

    // 2. Generate ReduceDim<D> (Removes dimension)
    for rank in 1..=max_rank {
        for d in 0..rank {
            let in_types: Vec<_> = (0..rank).map(|i| format_ident!("D{}", i)).collect();
            let mut out_types = in_types.clone();
            out_types.remove(d);

            if out_types.is_empty() {
                tokens.extend(quote! {
                    impl<#(#in_types: crate::prelude::Dim,)*> crate::shapes::shape_ops::ReduceDim<#d> for (#(#in_types,)*) {
                        /// Output.
                        type Output = (); // Scalar
                    }
                });
            } else {
                tokens.extend(quote! {
                    impl<#(#in_types: crate::prelude::Dim,)*> crate::shapes::shape_ops::ReduceDim<#d> for (#(#in_types,)*) {
                        /// Output.
                        type Output = (#(#out_types,)*);
                    }
                });
            }
        }
    }

    // 3. Generate ReduceKeepDim<D> (Replaces with U1)
    for rank in 1..=max_rank {
        for d in 0..rank {
            let in_types: Vec<_> = (0..rank).map(|i| format_ident!("D{}", i)).collect();
            let mut out_types = Vec::new();
            for (i, ty) in in_types.iter().enumerate() {
                if i == d {
                    out_types.push(quote! { crate::prelude::typenum::U1 });
                } else {
                    out_types.push(quote! { #ty });
                }
            }

            tokens.extend(quote! {
                impl<#(#in_types: crate::prelude::Dim,)*> crate::shapes::shape_ops::ReduceKeepDim<#d> for (#(#in_types,)*) {
                    /// Output.
                    type Output = (#(#out_types,)*);
                }
            });
        }
    }

    // 4. Generate Flatten<START, END>
    // 4. Generate Flatten<START, END>
    for rank in 1..=max_rank {
        for start in 0..rank {
            for end in start..rank {
                let in_types: Vec<_> = (0..rank).map(|i| format_ident!("D{}", i)).collect();

                let before: Vec<_> = in_types[0..start].iter().map(|id| quote! { #id }).collect();
                let after: Vec<_> = if end + 1 < rank {
                    in_types[(end + 1)..rank]
                        .iter()
                        .map(|id| quote! { #id })
                        .collect()
                } else {
                    Vec::new()
                };

                let start_id = &in_types[start];
                let mut prod = quote! { #start_id };

                for next in in_types.iter().take(end + 1).skip(start + 1) {
                    prod = quote! { crate::shapes::dim::ProdDim<#prod, #next> };
                }

                let out_type = if before.is_empty() && after.is_empty() {
                    quote! { (#prod,) }
                } else {
                    quote! { (#(#before,)* #prod, #(#after,)*) }
                };

                tokens.extend(quote! {
                    impl<#(#in_types: crate::prelude::Dim,)*> crate::shapes::shape_ops::Flatten<#start, #end> for (#(#in_types,)*) {
                        /// Output.
                        type Output = #out_type;
                    }
                });
            }
        }
    }

    tokens.into()
}
