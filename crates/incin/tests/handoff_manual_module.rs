//! Human-handoff acceptance fixtures.
//!
//! The manual implementation is deliberately written without `#[module]`.
//! The macro fixture below exercises the same public traversal contracts.

#![cfg(all(feature = "cpu", feature = "target-api"))]

use incin::nn::{Module, Parameters, StateDict, TrainState};
use incin::prelude::*;
use incin::state::StateLoadPlan;
use std::collections::BTreeMap;

type CpuBackend = incin_backends::cpu::CpuBackendImpl;
type Input = Tensor<Dyn, CpuBackend, f32, Grad>;
type Layer = Linear<s![4, 2], CpuBackend>;

struct ManualLinear {
    layer: Layer,
}

impl Module<Input> for ManualLinear {
    type Output = Input;
    type Error = Error;

    fn forward(&self, input: Input) -> Result<Self::Output> {
        self.layer.forward(input)
    }
}

impl<K: DType> Parameters<CpuBackend, K> for ManualLinear {
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut BTreeMap<String, <CpuBackend as VariableBackend>::Var<K>>,
    ) {
        let child_prefix = if prefix.is_empty() {
            "layer".to_owned()
        } else {
            format!("{prefix}.layer")
        };
        self.layer.named_parameters(&child_prefix, map);
    }
}

impl VisitState<CpuBackend> for ManualLinear {
    fn visit_state<V: StateVisitor<CpuBackend>>(
        &self,
        path: &StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.layer.visit_state(&path.child("layer"), visitor)
    }
}

impl StateDict<CpuBackend> for ManualLinear {
    fn collect_state(&self, prefix: &StatePath, snapshot: &mut StateSnapshot) -> Result<()> {
        self.layer.collect_state(&prefix.child("layer"), snapshot)
    }

    fn prepare_state(
        &self,
        prefix: &StatePath,
        snapshot: &StateSnapshot,
        plan: &mut StateLoadPlan,
    ) -> Result<()> {
        self.layer.prepare_state(&prefix.child("layer"), snapshot, plan)
    }

    fn commit_state(&mut self, prefix: &StatePath, plan: &mut StateLoadPlan) -> Result<()> {
        self.layer.commit_state(&prefix.child("layer"), plan)
    }
}

#[module(no_stats)]
struct MacroLinear {
    layer: Layer,
}

impl Module<Input> for MacroLinear {
    type Output = Input;
    type Error = Error;

    fn forward(&self, input: Input) -> Result<Self::Output> {
        self.layer.forward(input)
    }
}

struct ForwardOnlyField;

struct VisitedPaths(Vec<String>);

impl StateVisitor<CpuBackend> for VisitedPaths {
    fn visit_param<S, K, Train>(
        &mut self,
        path: &StatePath,
        _param: &incin::nn::param::Param<S, CpuBackend, K, Train>,
    ) -> Result<()>
    where
        S: Shape,
        K: DType,
        Train: TrainState,
    {
        self.0.push(path.to_string());
        Ok(())
    }

    fn visit_buffer<S, K>(
        &mut self,
        path: &StatePath,
        _buffer: &incin::nn::param::Buffer<S, CpuBackend, K>,
    ) -> Result<()>
    where
        S: Shape,
        K: DType,
    {
        self.0.push(path.to_string());
        Ok(())
    }
}

impl Module<Input> for ForwardOnlyField {
    type Output = Input;
    type Error = Error;

    fn forward(&self, input: Input) -> Result<Self::Output> {
        Ok(input)
    }
}

#[module(
    no_stats,
    no_parameters,
    no_state,
    no_named_layers,
    no_shape_info,
    no_train_mode,
    no_to_device,
)]
struct ForwardOnlyMacro {
    field: ForwardOnlyField,
}

impl Module<Input> for ForwardOnlyMacro {
    type Output = Input;
    type Error = Error;

    fn forward(&self, input: Input) -> Result<Self::Output> {
        self.field.forward(input)
    }
}

#[test]
fn manual_and_macro_modules_have_equivalent_state_and_forward_behavior() -> Result<()> {
    let target = Cpu;
    let manual = ManualLinear {
        layer: Linear::build(())?,
    };
    let macro_layer = MacroLinear {
        layer: Linear::build(())?,
    };

    assert_eq!(
        <ManualLinear as Parameters<CpuBackend, f32>>::parameters(&manual)
            .keys()
            .collect::<Vec<_>>(),
        <MacroLinear as Parameters<CpuBackend, f32>>::parameters(&macro_layer)
            .keys()
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        manual.state_dict()?.iter().map(|(path, _)| path).collect::<Vec<_>>(),
        macro_layer.state_dict()?.iter().map(|(path, _)| path).collect::<Vec<_>>(),
    );

    let input = target.zeros(shape![2, 4])?.into_dyn().require_grad();
    let manual_output = manual.forward(input.clone())?;
    let macro_output = macro_layer.forward(input)?;
    assert_eq!(manual_output.dims(), macro_output.dims());
    assert_eq!(manual_output.dims(), [2, 2]);

    // Exercise the training capability boundary as well as forward/state.
    let target_output = target.zeros(shape![2, 2])?.into_dyn();
    let loss = manual_output.mse_loss(&target_output)?;
    let _grads = loss.backward()?;

    let snapshot = manual.state_dict()?;
    let mut restored = manual;
    restored.load_state_dict(&snapshot)?;
    assert_eq!(
        restored.state_dict()?.iter().map(|(path, _)| path).collect::<Vec<_>>(),
        snapshot.iter().map(|(path, _)| path).collect::<Vec<_>>(),
    );

    let mut manual_paths = VisitedPaths(Vec::new());
    restored.visit_state(&StatePath::root(), &mut manual_paths)?;
    let mut macro_paths = VisitedPaths(Vec::new());
    macro_layer.visit_state(&StatePath::root(), &mut macro_paths)?;
    assert_eq!(manual_paths.0, macro_paths.0);

    Ok(())
}

#[test]
fn macro_can_opt_into_forward_only_capabilities() -> Result<()> {
    let input = Cpu.zeros(shape![1, 4])?.into_dyn().require_grad();
    let output = ForwardOnlyMacro {
        field: ForwardOnlyField,
    }
    .forward(input)?;
    assert_eq!(output.dims(), [1, 4]);
    Ok(())
}
