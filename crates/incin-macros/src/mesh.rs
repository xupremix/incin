use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Ident, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

enum AxisDegree {
    Lit(syn::LitInt),
    Type(Box<syn::Type>),
}

impl Parse for AxisDegree {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(syn::LitInt) {
            let lit: syn::LitInt = input.parse()?;
            let val: usize = lit.base10_parse()?;
            if val == 0 {
                return Err(syn::Error::new(
                    lit.span(),
                    "mesh axis degree must be nonzero",
                ));
            }
            Ok(AxisDegree::Lit(lit))
        } else {
            Ok(AxisDegree::Type(Box::new(input.parse::<syn::Type>()?)))
        }
    }
}

struct KeyVal {
    key: Ident,
    degree: AxisDegree,
}

impl Parse for KeyVal {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let degree: AxisDegree = input.parse()?;
        Ok(KeyVal { key, degree })
    }
}

struct MeshInput {
    internal: bool,
    dp: Option<(Ident, AxisDegree)>,
    tp: Option<(Ident, AxisDegree)>,
    pp: Option<(Ident, AxisDegree)>,
}

impl Parse for MeshInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut internal = false;
        if input.peek(Token![@]) {
            input.parse::<Token![@]>()?;
            internal = true;
        }

        let mut dp = None;
        let mut tp = None;
        let mut pp = None;

        while !input.is_empty() {
            let kv: KeyVal = input.parse()?;
            let key_str = kv.key.to_string();
            match key_str.as_str() {
                "dp" => {
                    if dp.is_some() {
                        return Err(syn::Error::new(kv.key.span(), "duplicate mesh axis `dp`"));
                    }
                    dp = Some((kv.key, kv.degree));
                }
                "tp" => {
                    if tp.is_some() {
                        return Err(syn::Error::new(kv.key.span(), "duplicate mesh axis `tp`"));
                    }
                    tp = Some((kv.key, kv.degree));
                }
                "pp" => {
                    if pp.is_some() {
                        return Err(syn::Error::new(kv.key.span(), "duplicate mesh axis `pp`"));
                    }
                    pp = Some((kv.key, kv.degree));
                }
                other => {
                    return Err(syn::Error::new(
                        kv.key.span(),
                        format!("unknown mesh axis `{other}`; valid axes are `dp`, `tp`, and `pp`"),
                    ));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(MeshInput {
            internal,
            dp,
            tp,
            pp,
        })
    }
}

fn render_degree(
    deg: &Option<(Ident, AxisDegree)>,
    path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match deg {
        None => crate::shape::lit_to_typenum(1, path),
        Some((_, AxisDegree::Lit(lit))) => {
            let val: usize = lit.base10_parse().unwrap_or(1);
            crate::shape::lit_to_typenum(val, path)
        }
        Some((_, AxisDegree::Type(ty))) => quote! { #ty },
    }
}

pub(crate) fn mesh(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as MeshInput);
    let path = if parsed.internal {
        quote! { crate:: }
    } else {
        quote! { ::incin_core:: }
    };

    let dp_rendered = render_degree(&parsed.dp, &path);
    let tp_rendered = render_degree(&parsed.tp, &path);
    let pp_rendered = render_degree(&parsed.pp, &path);

    quote! {
        #path dist::mesh::MeshSpec<
            #path dist::mesh::Data<#dp_rendered>,
            #path dist::mesh::TensorParallel<#tp_rendered>,
            #path dist::mesh::Pipeline<#pp_rendered>
        >
    }
    .into()
}
