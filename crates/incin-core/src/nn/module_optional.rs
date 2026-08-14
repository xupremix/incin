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
        impl<S: $shape_trait, B: crate::prelude::VariableBackend, Bias: crate::nn::optional::OptionalField> crate::nn::module::Parameters<B> for $name<S, B, Bias> {
            /// Collects named trainable parameters into `map` under the given `prefix`.
            fn named_parameters(&self, prefix: &str, map: &mut alloc::collections::BTreeMap<String, <B as crate::prelude::VariableBackend>::RawVar>) {
                let prefix = if prefix.is_empty() { "".to_string() } else { format!("{}.", prefix) };
                $(
                    crate::prelude::Parameters::named_parameters(
                        &self.$field, &alloc::format!("{}{}", prefix, stringify!($field)), map);
                )*
            }
        }

        impl<S: $shape_trait, B: crate::prelude::VariableBackend, Bias: crate::nn::optional::OptionalField> crate::nn::StateDict<B> for $name<S, B, Bias> {
            fn collect_state(&self, path: &crate::nn::StatePath, snapshot: &mut crate::nn::StateSnapshot) -> crate::prelude::Result<()> {
                $( crate::prelude::StateDict::collect_state(&self.$field, &path.child(stringify!($field)), snapshot)?; )*
                Ok(())
            }
            fn prepare_state(&self, path: &crate::nn::StatePath, snapshot: &crate::nn::StateSnapshot, plan: &mut crate::nn::StateLoadPlan) -> crate::prelude::Result<()> {
                $( crate::prelude::StateDict::prepare_state(&self.$field, &path.child(stringify!($field)), snapshot, plan)?; )*
                Ok(())
            }
            fn commit_state(&mut self, path: &crate::nn::StatePath, plan: &mut crate::nn::StateLoadPlan) -> crate::prelude::Result<()> {
                $( crate::prelude::StateDict::commit_state(&mut self.$field, &path.child(stringify!($field)), plan)?; )*
                Ok(())
            }
        }
    };
}
