use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, ReturnType};

/// The `#[kindle::module]` macro.
/// Currently acts as a pass-through.
pub(crate) fn module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// The `#[kindle::forward]` macro.
/// AST shape tracer implementation for demonstration.
pub(crate) fn forward(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut func = parse_macro_input!(item as ItemFn);
    
    let fn_name = &func.sig.ident;
    let ret_type_str = match &func.sig.output {
        ReturnType::Type(_, ty) => quote!(#ty).to_string(),
        _ => "()".to_string(),
    };

    println!("\n--------------------------------------------------");
    println!("⚙️  [AST Tracer] Analyzing #[forward] on function `{}`", fn_name);
    println!("⚙️  [AST Tracer] Target return type: {}", ret_type_str);

    if let Some(last_stmt) = func.block.stmts.last_mut()
        && let syn::Stmt::Expr(expr, _semi) = last_stmt
            && let syn::Expr::Call(call) = expr
                && let syn::Expr::Path(path) = &*call.func
                    && path.path.is_ident("Ok")
                        && let Some(arg) = call.args.first_mut() {
                            println!("⚙️  [AST Tracer] Found return expression: `Ok({})`", quote!(#arg));
                            
                            let rewritten_arg: syn::Expr = syn::parse_quote! {
                                (#arg).into_shape()?
                            };
                            
                            println!("⚙️  [AST Tracer] 🚀 REWRITING TO: `Ok({})` to enforce boundary!", quote!(#rewritten_arg));
                            *arg = rewritten_arg;
                        }
    
    println!("--------------------------------------------------\n");

    TokenStream::from(quote!(#func))
}
