use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

pub fn distributed_main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let block = &input_fn.block;
    let attrs = &input_fn.attrs;
    let vis = &input_fn.vis;
    let inputs = &input_fn.sig.inputs;

    let expanded = quote! {
        #(#attrs)*
        #vis fn #fn_name(#inputs) {
            let args: Vec<String> = ::std::env::args().collect();
            let mut rank: usize = 0;
            let mut world_size: usize = 1;

            let mut i = 0;
            while i < args.len() {
                if args[i] == "--rank" && i + 1 < args.len() {
                    rank = args[i + 1].parse().unwrap_or(0);
                } else if args[i] == "--world-size" && i + 1 < args.len() {
                    world_size = args[i + 1].parse().unwrap_or(1);
                }
                i += 1;
            }

            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                #block
            }));

            match result {
                Ok(_) => {},
                Err(_) => {
                    eprintln!("[#[distributed_main]] Rank {} / {} failed or panicked.", rank, world_size);
                    ::std::process::exit(1);
                }
            }
        }
    };

    TokenStream::from(expanded)
}
