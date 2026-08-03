use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Ident, LitInt, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

enum MeshKey {
    Data,
    Tensor,
    Pipeline,
}

struct MeshField {
    key: MeshKey,
    key_ident: Ident,
    val: usize,
}

impl Parse for MeshField {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key_ident: Ident = input.parse()?;
        let key_str = key_ident.to_string();
        let key = match key_str.as_str() {
            "dp" | "data" => MeshKey::Data,
            "tp" | "tensor" | "tensor_parallel" => MeshKey::Tensor,
            "pp" | "pipeline" | "pipeline_parallel" => MeshKey::Pipeline,
            _ => {
                return Err(syn::Error::new_spanned(
                    &key_ident,
                    format!("unknown mesh axis key `{key_str}`; expected `dp`, `tp`, or `pp`"),
                ));
            }
        };

        input.parse::<Token![=]>()?;
        let lit: LitInt = input.parse()?;
        let val: usize = lit.base10_parse()?;
        if val == 0 {
            return Err(syn::Error::new_spanned(
                &lit,
                "mesh degree must be a non-zero positive integer",
            ));
        }

        Ok(MeshField {
            key,
            key_ident,
            val,
        })
    }
}

struct MeshInput {
    internal: bool,
    fields: Vec<MeshField>,
}

impl Parse for MeshInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut internal = false;
        if input.peek(Token![@]) {
            input.parse::<Token![@]>()?;
            internal = true;
        }

        let punctuated: Punctuated<MeshField, Token![,]> = Punctuated::parse_terminated(input)?;
        Ok(MeshInput {
            internal,
            fields: punctuated.into_iter().collect(),
        })
    }
}

pub(crate) fn mesh(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as MeshInput);

    let mut data_val = 1;
    let mut tensor_val = 1;
    let mut pipeline_val = 1;

    let mut has_data = false;
    let mut has_tensor = false;
    let mut has_pipeline = false;

    for field in parsed.fields {
        match field.key {
            MeshKey::Data => {
                if has_data {
                    return syn::Error::new_spanned(
                        field.key_ident,
                        "duplicate `dp` / `data` axis in mesh!",
                    )
                    .to_compile_error()
                    .into();
                }
                has_data = true;
                data_val = field.val;
            }
            MeshKey::Tensor => {
                if has_tensor {
                    return syn::Error::new_spanned(
                        field.key_ident,
                        "duplicate `tp` / `tensor` axis in mesh!",
                    )
                    .to_compile_error()
                    .into();
                }
                has_tensor = true;
                tensor_val = field.val;
            }
            MeshKey::Pipeline => {
                if has_pipeline {
                    return syn::Error::new_spanned(
                        field.key_ident,
                        "duplicate `pp` / `pipeline` axis in mesh!",
                    )
                    .to_compile_error()
                    .into();
                }
                has_pipeline = true;
                pipeline_val = field.val;
            }
        }
    }

    let (mesh_prefix, typenum_prefix) = if parsed.internal {
        (quote! { crate::dist::mesh:: }, quote! { crate:: })
    } else {
        (
            quote! { ::incin::experimental::distributed::mesh:: },
            quote! { ::incin:: },
        )
    };

    let dp_typenum = crate::shape::lit_to_typenum(data_val, &typenum_prefix);
    let tp_typenum = crate::shape::lit_to_typenum(tensor_val, &typenum_prefix);
    let pp_typenum = crate::shape::lit_to_typenum(pipeline_val, &typenum_prefix);

    quote! {
        #mesh_prefix MeshSpec<
            #mesh_prefix Data<#dp_typenum>,
            #mesh_prefix TensorParallel<#tp_typenum>,
            #mesh_prefix Pipeline<#pp_typenum>
        >
    }
    .into()
}
