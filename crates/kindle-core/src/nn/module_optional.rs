#[macro_export]
macro_rules! impl_module_for_optional_field {
    (
        $name:ident,
        $shape_trait:ident,
        $weight_shape:ident,
        $bias_shape:ident,
        {
            $( $field:ident: $field_ty_macro:tt ),* $(,)?
        }
    ) => {
        impl<S: $shape_trait, B: crate::prelude::Backend, Bias: crate::nn::optional::OptionalField> crate::nn::module::Parameters<B> for $name<S, B, Bias> {
            /// Collects named trainable parameters into `map` under the given `prefix`.
            fn named_parameters(&self, prefix: &str, map: &mut alloc::collections::BTreeMap<String, <B as crate::prelude::Backend>::RawVar>) {
                use crate::nn::module::{AutorefParameters, AutorefParametersFallback};
                let prefix = if prefix.is_empty() { "".to_string() } else { format!("{}.", prefix) };
                $(
                    (&&self.$field).maybe_parameters(core::marker::PhantomData::<B>, &alloc::format!("{}{}", prefix, stringify!($field)), map);
                )*
            }
        }

        impl<S: $shape_trait, B: crate::prelude::Backend, Bias: crate::nn::optional::OptionalField> crate::nn::StateDict<B> for $name<S, B, Bias> {
            /// Loads parameters from a flat name→tensor map, in-place.
            fn load_state_dict(&mut self, prefix: &str, tensors: &alloc::collections::BTreeMap<String, crate::prelude::Tensor<crate::prelude::Dyn, B>>) -> crate::prelude::Result<()> {
                use crate::nn::module::{AutorefStateDict, AutorefStateDictFallback};
                let prefix = if prefix.is_empty() { "".to_string() } else { format!("{}.", prefix) };
                $(
                    (&mut &mut self.$field).maybe_load_state_dict(core::marker::PhantomData::<B>, &alloc::format!("{}{}", prefix, stringify!($field)), tensors)?;
                )*
                Ok(())
            }
            /// Returns a flat map from parameter name to its raw tensor value.
            fn state_dict(&self, prefix: &str, tensors: &mut alloc::collections::BTreeMap<String, crate::prelude::Tensor<crate::prelude::Dyn, B>>) {
                use crate::nn::module::{AutorefStateDict, AutorefStateDictFallback};
                let prefix = if prefix.is_empty() { "".to_string() } else { format!("{}.", prefix) };
                $(
                    (&&self.$field).maybe_state_dict(core::marker::PhantomData::<B>, &alloc::format!("{}{}", prefix, stringify!($field)), tensors);
                )*
            }
        }
    };
}
