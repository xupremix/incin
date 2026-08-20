//! `ParameterGroup`, an optimizer-owned collection of trainable variables
//! collected from a module by dtype, plus the two private types its own
//! constructor and every optimizer's update loop build on: `ParameterCollector`
//! (a `ParameterVisitor` that walks a module gathering dtype-`K` parameters,
//! used only by `ParameterGroup::from_module`) and `PreparedUpdate` (one
//! parameter's staged before/after storage, built by every optimizer's
//! `step` and read back by `super::support::commit_parameter_updates`).
//! `PreparedUpdate` and its fields are `pub(super)` rather than private for
//! that reason; `ParameterCollector` stays fully private, since nothing
//! outside this file ever names it.

use crate::err::{Error, Result};
use crate::nn::param::{Param, TrainState};
use crate::nn::{ParameterVisitor, StatePath, VisitParameters};
use crate::shapes::Shape;
use crate::tensor::backend::VariableBackend;
use crate::tensor::dtype::{ConstDType, DType};
use alloc::string::{String, ToString};

/// A homogeneous, optimizer-owned collection of trainable variables.
pub struct ParameterGroup<B: VariableBackend, K: ConstDType> {
    params: alloc::collections::BTreeMap<String, B::Var<K>>,
}

impl<B: VariableBackend, K: ConstDType> ParameterGroup<B, K> {
    /// Collects trainable parameters with dtype `K` through the canonical
    /// heterogeneous module visitor. Parameters of other dtypes are ignored;
    /// this lets one model contain optimizer-incompatible auxiliary dtypes
    /// without creating a second module traversal architecture.
    pub fn from_module<M>(module: &M) -> Result<Self>
    where
        M: VisitParameters<B>,
    {
        let mut collector = ParameterCollector::<B, K> {
            params: alloc::collections::BTreeMap::new(),
        };
        module.visit_parameters(&StatePath::root(), &mut collector)?;
        Ok(Self {
            params: collector.params,
        })
    }

    /// Creates a group from an already collected homogeneous map.
    #[must_use]
    pub fn from_map(params: alloc::collections::BTreeMap<String, B::Var<K>>) -> Self {
        Self { params }
    }

    /// Returns the number of collected variables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Returns whether the group contains no variables.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Iterates over the collected variables in canonical path order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &B::Var<K>)> {
        self.params.iter()
    }

    pub(super) fn into_map(self) -> alloc::collections::BTreeMap<String, B::Var<K>> {
        self.params
    }
}

struct ParameterCollector<B: VariableBackend, K: ConstDType> {
    params: alloc::collections::BTreeMap<String, B::Var<K>>,
}

impl<B: VariableBackend, K: ConstDType> ParameterVisitor<B> for ParameterCollector<B, K> {
    fn visit_param<S, LeafK, Train>(
        &mut self,
        path: &StatePath,
        param: &Param<S, B, LeafK, Train>,
    ) -> Result<()>
    where
        S: Shape,
        LeafK: DType,
        Train: TrainState,
    {
        if param.dtype_descriptor() != K::DESCRIPTOR {
            return Ok(());
        }
        let variable =
            param
                .variable_any()
                .downcast_ref::<B::Var<K>>()
                .ok_or(Error::InternalInvariant {
                    operation: "collect parameter group",
                    reason: "dtype matched but backend variable type did not",
                })?;
        self.params.insert(path.to_string(), variable.clone());
        Ok(())
    }
}

pub(super) struct PreparedUpdate<S> {
    pub(super) name: String,
    pub(super) before: S,
    pub(super) updated: S,
    pub(super) first_moment: Option<S>,
    pub(super) second_moment: Option<S>,
}
