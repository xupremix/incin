#[cfg(not(feature = "authoring"))]
use fire::{declare_capabilities, declare_executors};

#[cfg(feature = "authoring")]
mod authoring {
    use fire::backend_authoring::{
        CapabilityQuery, ExecutionRequest, StorageBackend, SupportLevel, TensorMeta, operations::op,
    };
    use fire::{BackendError, Cpu, DType};

    pub struct Backend;

    impl StorageBackend for Backend {
        const BACKEND_NAME: &'static str = "negative-fixture";
        type Storage<K: DType> = TensorMeta;
        type Device = Cpu;

        fn metadata<K: DType>(storage: &TensorMeta) -> &TensorMeta {
            storage
        }
    }

    fn support(_: &Backend, _: &CapabilityQuery) -> SupportLevel {
        SupportLevel::Native
    }

    fn scalar(
        _: &Backend,
        _: ExecutionRequest<'_, op::Zeros, Backend>,
    ) -> Result<f64, BackendError> {
        Ok(1.0)
    }

    #[cfg(not(feature = "wrong-output"))]
    fire::backend_authoring::declare_executors! {
        for Backend { op::Zeros => f64 = scalar; }
    }

    #[cfg(feature = "wrong-output")]
    fire::backend_authoring::declare_executors! {
        for Backend { op::Zeros => Vec<f64> = scalar; }
    }

    #[cfg(feature = "missing-executor")]
    fire::backend_authoring::declare_capabilities! {
        for Backend {
            op::Zeros => support;
            op::Ones => support;
        }
    }

    #[cfg(not(any(feature = "missing-executor", feature = "custom-capability")))]
    fire::declare_capabilities! {
        for Backend { op::Zeros => support; }
    }

    #[cfg(feature = "custom-capability")]
    mod custom {
        use super::*;
        use fire::backend_authoring::operations::NoAttributes;
        use fire::backend_authoring::{
            DescriptorError, LogicalTensorMeta, Operation, OperationKey,
        };

        #[derive(Debug, Clone)]
        struct Custom;

        impl Operation for Custom {
            type Attributes = NoAttributes;
            const KEY: OperationKey = OperationKey {
                namespace: std::borrow::Cow::Borrowed("fixture"),
                name: std::borrow::Cow::Borrowed("custom"),
                version: 1,
            };

            fn infer_outputs(
                _: &NoAttributes,
                _: &[LogicalTensorMeta],
            ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
                Ok(vec![])
            }
        }

        fn custom(
            _: &Backend,
            _: ExecutionRequest<'_, Custom, Backend>,
        ) -> Result<(), BackendError> {
            Ok(())
        }

        fire::declare_executors! { for Backend { Custom => () = custom; } }
        fire::declare_capabilities! { for Backend { Custom => support; } }
    }
}
