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
            fn collect_state(&self, path: &crate::nn::StatePath, snapshot: &mut crate::nn::StateSnapshot) -> crate::prelude::Result<()> {
                use crate::nn::module::{AutorefStateDict, AutorefStateDictFallback};
                $( (&&self.$field).maybe_collect_state(core::marker::PhantomData::<B>, &path.child(stringify!($field)), snapshot)?; )*
                Ok(())
            }
            fn prepare_state(&self, path: &crate::nn::StatePath, snapshot: &crate::nn::StateSnapshot, plan: &mut crate::nn::StateLoadPlan<B>) -> crate::prelude::Result<()> {
                use crate::nn::module::{AutorefStateDict, AutorefStateDictFallback};
                $( (&&self.$field).maybe_prepare_state(core::marker::PhantomData::<B>, &path.child(stringify!($field)), snapshot, plan)?; )*
                Ok(())
            }
            fn commit_state(&mut self, path: &crate::nn::StatePath, plan: &mut crate::nn::StateLoadPlan<B>) -> crate::prelude::Result<()> {
                use crate::nn::module::{AutorefStateDict, AutorefStateDictFallback};
                $( (&mut &mut self.$field).maybe_commit_state(core::marker::PhantomData::<B>, &path.child(stringify!($field)), plan)?; )*
                Ok(())
            }
        }
    };
}
