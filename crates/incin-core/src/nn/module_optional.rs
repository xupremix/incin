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
        impl<S: $shape_trait, B: crate::tensor::backend::VariableBackend, Bias: crate::nn::optional::OptionalField> crate::nn::VisitState<B> for $name<S, B, Bias> {
            fn visit_state<V: crate::nn::StateVisitor<B>>(&self, path: &crate::nn::StatePath, visitor: &mut V) -> crate::err::Result<()> {
                $( crate::nn::VisitState::visit_state(&self.$field, &path.try_child(stringify!($field))?, visitor)?; )*
                Ok(())
            }
        }

        impl<S: $shape_trait, B: crate::tensor::backend::VariableBackend, Bias: crate::nn::optional::OptionalField> crate::nn::VisitStateMut<B> for $name<S, B, Bias> {
            fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(&mut self, path: &crate::nn::StatePath, visitor: &mut V) -> crate::err::Result<()> {
                $( crate::nn::VisitStateMut::visit_state_mut(&mut self.$field, &path.try_child(stringify!($field))?, visitor)?; )*
                Ok(())
            }
        }

        impl<S: $shape_trait, B: crate::tensor::backend::VariableBackend, Bias: crate::nn::optional::OptionalField> crate::nn::VisitParameters<B> for $name<S, B, Bias> {
            fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(&self, path: &crate::nn::StatePath, visitor: &mut V) -> crate::err::Result<()> {
                $( crate::nn::VisitParameters::visit_parameters(&self.$field, &path.try_child(stringify!($field))?, visitor)?; )*
                Ok(())
            }
        }
    };
}
