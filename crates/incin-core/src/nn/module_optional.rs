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
        impl<S: $shape_trait, B: crate::prelude::VariableBackend, Bias: crate::nn::optional::OptionalField> crate::nn::module::Parameters<B, K> for $name<S, B, Bias> {
            /// Collects named trainable parameters into `map` under the given `prefix`.
            fn named_parameters(&self, prefix: &str, map: &mut alloc::collections::BTreeMap<String, <B as crate::prelude::VariableBackend>::Var<K>>) {
                let prefix = if prefix.is_empty() { "".to_string() } else { format!("{}.", prefix) };
                $(
                    crate::prelude::Parameters::named_parameters(
                        &self.$field, &alloc::format!("{}{}", prefix, stringify!($field)), map);
                )*
            }
        }

        impl<S: $shape_trait, B: crate::prelude::VariableBackend, Bias: crate::nn::optional::OptionalField> crate::nn::VisitState<B> for $name<S, B, Bias> {
            fn visit_state<V: crate::nn::StateVisitor<B>>(&self, path: &crate::nn::StatePath, visitor: &mut V) -> crate::prelude::Result<()> {
                $( crate::prelude::VisitState::visit_state(&self.$field, &path.child(stringify!($field)), visitor)?; )*
                Ok(())
            }
        }

        impl<S: $shape_trait, B: crate::prelude::VariableBackend, Bias: crate::nn::optional::OptionalField> crate::nn::VisitStateMut<B> for $name<S, B, Bias> {
            fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(&mut self, path: &crate::nn::StatePath, visitor: &mut V) -> crate::prelude::Result<()> {
                $( crate::prelude::VisitStateMut::visit_state_mut(&mut self.$field, &path.child(stringify!($field)), visitor)?; )*
                Ok(())
            }
        }

        impl<S: $shape_trait, B: crate::prelude::VariableBackend, Bias: crate::nn::optional::OptionalField> crate::nn::VisitParameters<B> for $name<S, B, Bias> {
            fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(&self, path: &crate::nn::StatePath, visitor: &mut V) -> crate::prelude::Result<()> {
                $( crate::prelude::VisitParameters::visit_parameters(&self.$field, &path.child(stringify!($field)), visitor)?; )*
                Ok(())
            }
        }
    };
}
