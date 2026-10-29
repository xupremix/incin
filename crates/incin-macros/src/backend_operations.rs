//! Public backend-authoring surface: `define_backend_operations!`.
//!
//! Built-in backends keep their repetitive `Execute` shells DRY with
//! workspace-private macros. Downstream authors cannot reach those, so this
//! module public-izes the pattern as a procedural macro: one invocation
//! declares the `Execute` shells and the companion `Capabilities` routing,
//! with the coverage obligation checked at expansion time.
//!
//! A declarative macro could emit the same impls, but its coverage failure
//! would be a bare trait-bound error pointing at generated code. The
//! procedural form reports the offending operation with its own span instead.

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{
    Path, Token, Type, braced,
    parse::{Parse, ParseStream},
    spanned::Spanned,
};

/// One `Operation => Output = handler;` executor entry.
struct ExecutorEntry {
    operation: Type,
    output: Type,
    handler: Path,
}

/// One `Operation => handler;` capability entry.
struct CapabilityEntry {
    operation: Type,
    handler: Path,
}

/// The full macro input: executors plus an optional companion capabilities block.
struct BackendOperations {
    backend: Type,
    executors: Vec<ExecutorEntry>,
    capabilities_backend: Option<Type>,
    capabilities: Vec<CapabilityEntry>,
}

impl Parse for BackendOperations {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        input.parse::<Token![for]>()?;
        let backend: Type = input.parse()?;
        let executors_content;
        braced!(executors_content in input);
        let mut executors = Vec::new();
        while !executors_content.is_empty() {
            let operation: Type = executors_content.parse()?;
            executors_content.parse::<Token![=>]>()?;
            let output: Type = executors_content.parse()?;
            executors_content.parse::<Token![=]>()?;
            let handler: Path = executors_content.parse()?;
            executors_content.parse::<Token![;]>()?;
            executors.push(ExecutorEntry {
                operation,
                output,
                handler,
            });
        }
        if executors.is_empty() {
            return Err(syn::Error::new_spanned(
                &backend,
                "define_backend_operations! requires at least one executor entry \
                 of the form `Operation => Output = handler;`",
            ));
        }

        let mut capabilities_backend = None;
        let mut capabilities = Vec::new();
        if !input.is_empty() {
            let keyword: syn::Ident = input.parse()?;
            if keyword != "capabilities" {
                return Err(syn::Error::new_spanned(
                    &keyword,
                    "expected `capabilities for <Backend> { ... }` or the end of input",
                ));
            }
            input.parse::<Token![for]>()?;
            let declared: Type = input.parse()?;
            let capabilities_content;
            braced!(capabilities_content in input);
            while !capabilities_content.is_empty() {
                let operation: Type = capabilities_content.parse()?;
                capabilities_content.parse::<Token![=>]>()?;
                let handler: Path = capabilities_content.parse()?;
                capabilities_content.parse::<Token![;]>()?;
                capabilities.push(CapabilityEntry { operation, handler });
            }
            capabilities_backend = Some(declared);
        }

        Ok(BackendOperations {
            backend,
            executors,
            capabilities_backend,
            capabilities,
        })
    }
}

/// Whitespace-normalized spelling of a type, used to match capability entries
/// against executor entries. Entries match by spelling, not by identity: an
/// operation imported under two different paths reads as two operations.
fn type_key(operation: &Type) -> String {
    quote!(#operation).to_string()
}

/// Whether an operation path names the canonical `op::` catalog module, e.g.
/// `op::Zeros` or `operations::op::Zeros`. Single-segment paths and custom
/// module paths count as custom operations.
fn is_builtin_operation(operation: &Type) -> bool {
    let Type::Path(path) = operation else {
        return false;
    };
    let Some(first) = path.path.segments.first() else {
        return false;
    };
    if !first.arguments.is_empty() {
        return false;
    }
    let segments: Vec<_> = path
        .path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    segments.len() >= 2 && segments[segments.len() - 2] == "op"
}

/// Check the coverage obligations and emit the `Execute` and `Capabilities`
/// impls. Every diagnostic names its operation and points at that entry.
pub fn define_backend_operations(input: TokenStream) -> TokenStream {
    let parsed = syn::parse_macro_input!(input as BackendOperations);
    match render(parsed) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn render(parsed: BackendOperations) -> syn::Result<proc_macro2::TokenStream> {
    let backend = &parsed.backend;

    let mut seen_executors = std::collections::BTreeMap::new();
    for entry in &parsed.executors {
        let key = type_key(&entry.operation);
        if seen_executors.insert(key, entry.operation.span()).is_some() {
            return Err(syn::Error::new_spanned(
                &entry.operation,
                format!(
                    "duplicate executor entry for operation `{}`; \
                     each operation needs exactly one handler",
                    type_key(&entry.operation)
                ),
            ));
        }
    }

    let mut seen_capabilities = std::collections::BTreeMap::new();
    for entry in &parsed.capabilities {
        let key = type_key(&entry.operation);
        if seen_capabilities
            .insert(key, entry.operation.span())
            .is_some()
        {
            return Err(syn::Error::new_spanned(
                &entry.operation,
                format!(
                    "duplicate capability entry for operation `{}`; \
                     each operation needs exactly one support handler",
                    type_key(&entry.operation)
                ),
            ));
        }
    }

    if let Some(capabilities_backend) = &parsed.capabilities_backend
        && quote!(#capabilities_backend).to_string() != quote!(#backend).to_string()
    {
        return Err(syn::Error::new_spanned(
            capabilities_backend,
            format!(
                "capabilities backend `{}` does not match executors backend `{}`; \
                 one invocation describes one backend",
                quote!(#capabilities_backend),
                quote!(#backend)
            ),
        ));
    }

    for entry in &parsed.capabilities {
        if !seen_executors.contains_key(&type_key(&entry.operation)) {
            return Err(syn::Error::new_spanned(
                &entry.operation,
                format!(
                    "capability advertises operation `{}` without an executor entry; \
                     add `{} => <Output> = <handler>` to the executors list",
                    type_key(&entry.operation),
                    type_key(&entry.operation)
                ),
            ));
        }
    }

    for entry in &parsed.executors {
        if is_builtin_operation(&entry.operation)
            && !seen_capabilities.contains_key(&type_key(&entry.operation))
        {
            return Err(syn::Error::new_spanned(
                &entry.operation,
                format!(
                    "executor implements operation `{}` without a capability entry; \
                     add `{} => <handler>` to the capabilities list. \
                     Custom operations are exempt: keep them executor-only, \
                     admission stays on `Execute::supports_custom`",
                    type_key(&entry.operation),
                    type_key(&entry.operation)
                ),
            ));
        }
    }

    let executor_impls = parsed.executors.iter().map(|entry| {
        let operation = &entry.operation;
        let output = &entry.output;
        let handler = &entry.handler;
        quote_spanned! {handler.span()=>
            impl ::incin::backend_authoring::Execute<#operation> for #backend {
                type Output = #output;

                fn execute(
                    &self,
                    request: ::incin::backend_authoring::ExecutionRequest<'_, #operation, Self>,
                ) -> ::core::result::Result<Self::Output, ::incin::BackendError> {
                    #handler(self, request)
                }
            }

            /// Handler-signature assertion. A handler returning the wrong
            /// output fails here as a mismatched-types error naming the
            /// operation and the declared output, rather than deep inside
            /// dispatch. Anonymous so one invocation can declare any number
            /// of entries and a module can invoke the macro more than once.
            /// The allowance travels with the assertion: the spelled-out
            /// handler signature is the diagnostic, not accidental complexity.
            #[allow(clippy::type_complexity)]
            const _: fn(
                &#backend,
                ::incin::backend_authoring::ExecutionRequest<'_, #operation, #backend>,
            ) -> ::core::result::Result<#output, ::incin::BackendError> = #handler;
        }
    });

    let capabilities_impl = if parsed.capabilities_backend.is_some() {
        let entries = &parsed.capabilities;
        let routes = entries.iter().map(|entry| {
            let operation = &entry.operation;
            let handler = &entry.handler;
            quote! {
                ::incin::backend_authoring::OperationIdentity::Builtin(
                    <#operation as ::incin::backend_authoring::operations::CanonicalOperation>::ID
                ) => #handler(self, query),
            }
        });
        let obligations = entries.iter().map(|entry| {
            let operation = &entry.operation;
            quote! {
                let _ = assert_executor::<#backend, #operation>;
            }
        });
        quote! {
            const _: () = {
                fn assert_executor<B, O>()
                where
                    O: ::incin::backend_authoring::operations::CanonicalOperation,
                    B: ::incin::backend_authoring::Execute<O>,
                {}
                #(#obligations)*
            };

            impl ::incin::backend_authoring::Capabilities for #backend {
                fn support(
                    &self,
                    query: &::incin::backend_authoring::CapabilityQuery,
                ) -> ::incin::backend_authoring::SupportLevel {
                    match &query.operation {
                        #(#routes)*
                        ::incin::backend_authoring::OperationIdentity::Builtin(operation) => {
                            ::incin::backend_authoring::SupportLevel::Unsupported(
                                ::incin::backend_authoring::UnsupportedReason::Operation {
                                    operation: *operation,
                                },
                            )
                        }
                        ::incin::backend_authoring::OperationIdentity::Custom(operation) => {
                            ::incin::backend_authoring::SupportLevel::Unsupported(
                                ::incin::backend_authoring::UnsupportedReason::CustomOperation {
                                    operation: operation.clone(),
                                },
                            )
                        }
                    }
                }
            }
        }
    } else {
        quote! {}
    };

    Ok(quote! {
        #(#executor_impls)*
        #capabilities_impl
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_err(source: &str) -> String {
        let parsed: BackendOperations = syn::parse_str(source).expect("fixture must parse");
        match render(parsed) {
            Ok(_) => panic!("fixture must fail"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn capability_without_executor_names_the_operation() {
        let message = render_err(
            "for Backend {
                op::Zeros => f64 = zeros;
            }
            capabilities for Backend {
                op::Zeros => support;
                op::Ones => support;
            }",
        );
        assert!(
            message.contains("op :: Ones") && message.contains("without an executor entry"),
            "unexpected diagnostic: {message}"
        );
    }

    #[test]
    fn builtin_executor_without_capability_names_the_operation() {
        let message = render_err(
            "for Backend {
                op::Zeros => f64 = zeros;
                op::Ones => f64 = ones;
            }
            capabilities for Backend {
                op::Zeros => support;
            }",
        );
        assert!(
            message.contains("op :: Ones") && message.contains("without a capability entry"),
            "unexpected diagnostic: {message}"
        );
    }

    #[test]
    fn custom_executors_need_no_capability_entry() {
        let parsed: BackendOperations = syn::parse_str(
            "for Backend {
                op::Zeros => f64 = zeros;
                CustomOp => f64 = custom;
            }
            capabilities for Backend {
                op::Zeros => support;
            }",
        )
        .expect("fixture must parse");
        assert!(render(parsed).is_ok());
    }

    #[test]
    fn duplicate_entries_name_the_operation() {
        let message = render_err(
            "for Backend {
                op::Zeros => f64 = zeros;
                op::Zeros => f64 = zeros_again;
            }",
        );
        assert!(
            message.contains("op :: Zeros") && message.contains("duplicate executor entry"),
            "unexpected diagnostic: {message}"
        );
    }
}
