/// Declares exact executor shells for a concrete backend type.
///
/// Each handler receives `&self` and the original validated request by value,
/// returning `Result<Output, BackendError>` without conversion. Built-in and
/// custom `Operation` types are accepted. Custom admission keeps `Execute`'s
/// defaults; write the implementation by hand when custom admission is restricted.
/// Generic impl parameters and where clauses are not supported, but concrete
/// types such as `MyBackend<Cpu>` are accepted.
///
/// # Examples
///
/// ```
/// use incin::backend_authoring::{declare_executors, ExecutionRequest, StorageBackend,
///     TensorMeta, ShapeBuf, operations::op};
/// use incin::{BackendError, Cpu, DType};
/// struct MyBackend;
/// impl StorageBackend for MyBackend {
///     const BACKEND_NAME: &'static str = "example";
///     type Storage<K: DType> = TensorMeta;
///     type Device = Cpu;
///     fn metadata<K: DType>(storage: &TensorMeta) -> &TensorMeta { storage }
/// }
/// fn zeros(_: &MyBackend, request: ExecutionRequest<'_, op::Zeros, MyBackend>)
///     -> Result<ShapeBuf, BackendError>
/// {
///     Ok(ShapeBuf::from_slice(&request.operation.descriptor().attributes().shape))
/// }
/// declare_executors! {
///     for MyBackend {
///         op::Zeros => ShapeBuf = zeros;
///     }
/// }
/// ```
#[cfg(feature = "backend-authoring")]
#[macro_export]
macro_rules! declare_executors {
    (for $backend:ty { $($operation:ty => $output:ty = $handler:path;)* }) => {
        $(
            impl $crate::backend_authoring::Execute<$operation> for $backend {
                type Output = $output;

                fn execute(
                    &self,
                    request: $crate::backend_authoring::ExecutionRequest<'_, $operation, Self>,
                ) -> ::core::result::Result<Self::Output, $crate::BackendError> {
                    $handler(self, request)
                }
            }
        )*
    };
}

/// Declares built-in capability routing and proves every entry has an executor.
///
/// Each handler receives `(&Backend, &CapabilityQuery)` and returns its exact
/// `SupportLevel`; dispatch still enforces execution policy. Entries match the
/// built-in identity from `CanonicalOperation::ID`. Unlisted built-ins and custom
/// queries return typed unsupported reasons. Custom admission belongs to
/// `Execute`, not this declaration, and restricted custom admission must be
/// handwritten. The executor obligations are checked even without a dispatch call.
///
/// # Examples
///
/// ```
/// use incin::backend_authoring::{declare_capabilities, declare_executors,
///     Capabilities, CapabilityQuery, ExecutionRequest, StorageBackend, SupportLevel,
///     TensorMeta, operations::op};
/// use incin::{BackendError, Cpu, DType};
/// struct MyBackend;
/// impl StorageBackend for MyBackend {
///     const BACKEND_NAME: &'static str = "example";
///     type Storage<K: DType> = TensorMeta;
///     type Device = Cpu;
///     fn metadata<K: DType>(storage: &TensorMeta) -> &TensorMeta { storage }
/// }
/// fn zeros(_: &MyBackend, _: ExecutionRequest<'_, op::Zeros, MyBackend>)
///     -> Result<f64, BackendError> { Ok(0.0) }
/// fn support(_: &MyBackend, _: &CapabilityQuery) -> SupportLevel {
///     SupportLevel::Native
/// }
/// declare_executors! { for MyBackend { op::Zeros => f64 = zeros; } }
/// declare_capabilities! { for MyBackend { op::Zeros => support; } }
/// ```
#[cfg(feature = "backend-authoring")]
#[macro_export]
macro_rules! declare_capabilities {
    (for $backend:ty { $($operation:ty => $handler:path;)* }) => {
        const _: () = {
            fn assert_executor<B, O>()
            where
                O: $crate::backend_authoring::operations::CanonicalOperation,
                B: $crate::backend_authoring::Execute<O>,
            {}
            $(let _ = assert_executor::<$backend, $operation>;)*
        };

        impl $crate::backend_authoring::Capabilities for $backend {
            fn support(
                &self,
                query: &$crate::backend_authoring::CapabilityQuery,
            ) -> $crate::backend_authoring::SupportLevel {
                match &query.operation {
                    $(
                        $crate::backend_authoring::OperationIdentity::Builtin(
                            <$operation as $crate::backend_authoring::operations::CanonicalOperation>::ID
                        ) => $handler(self, query),
                    )*
                    $crate::backend_authoring::OperationIdentity::Builtin(operation) => {
                        $crate::backend_authoring::SupportLevel::Unsupported(
                            $crate::backend_authoring::UnsupportedReason::Operation {
                                operation: *operation,
                            },
                        )
                    }
                    $crate::backend_authoring::OperationIdentity::Custom(operation) => {
                        $crate::backend_authoring::SupportLevel::Unsupported(
                            $crate::backend_authoring::UnsupportedReason::CustomOperation {
                                operation: operation.clone(),
                            },
                        )
                    }
                }
            }
        }
    };
}
