//! Human-handoff acceptance fixtures.
//!
//! The manual implementation is deliberately written without `#[module]`.
//! The macro fixture below exercises the same public traversal contracts.

#![cfg(all(feature = "cpu", feature = "target-api"))]

use incin::nn::{Module, ParameterVisitor, TrainState, VisitParameters};
use incin::prelude::*;

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

impl VisitState<CpuBackend> for ManualLinear {
    fn visit_state<V: StateVisitor<CpuBackend>>(
        &self,
        path: &StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.layer.visit_state(&path.child("layer"), visitor)
    }
}

impl VisitParameters<CpuBackend> for ManualLinear {
    fn visit_parameters<V: ParameterVisitor<CpuBackend>>(
        &self,
        path: &StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.layer
            .visit_parameters(&path.child("layer"), visitor)
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
        K: DType<Arg = ()>,
        CpuBackend: incin::backend_authoring::SupportsDType<K>
            + incin::backend_authoring::Capabilities
            + incin::backend_authoring::HostInterop,
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
        K: DType<Arg = ()>,
        CpuBackend: incin::backend_authoring::SupportsDType<K>
            + incin::backend_authoring::Capabilities
            + incin::backend_authoring::HostInterop,
    {
        self.0.push(path.to_string());
        Ok(())
    }
}

impl ParameterVisitor<CpuBackend> for VisitedPaths {
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

    let mut manual_parameters = VisitedPaths(Vec::new());
    manual.visit_parameters(&StatePath::root(), &mut manual_parameters)?;
    let mut macro_parameters = VisitedPaths(Vec::new());
    macro_layer.visit_parameters(&StatePath::root(), &mut macro_parameters)?;
    manual_parameters.0.sort();
    macro_parameters.0.sort();
    assert_eq!(manual_parameters.0, macro_parameters.0);
    let manual_snapshot = incin::state::collect_state::<CpuBackend, _>(&manual)?;
    assert_eq!(
        manual_snapshot.iter().map(|(path, _)| path).collect::<Vec<_>>(),
        incin::state::collect_state::<CpuBackend, _>(&macro_layer)?.iter().map(|(path, _)| path).collect::<Vec<_>>(),
    );
    assert_eq!(
        incin::state::collect_state::<CpuBackend, _>(&macro_layer)?.iter().map(|(path, _)| path).collect::<Vec<_>>(),
        incin::state::collect_state::<CpuBackend, _>(&macro_layer)?.iter().map(|(path, _)| path).collect::<Vec<_>>(),
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

    let snapshot = incin::state::collect_state::<CpuBackend, _>(&macro_layer)?;
    let mut restored = macro_layer;
    restored.load_state_dict(&snapshot)?;
    assert_eq!(
        incin::state::collect_state::<CpuBackend, _>(&restored)?.iter().map(|(path, _)| path).collect::<Vec<_>>(),
        snapshot.iter().map(|(path, _)| path).collect::<Vec<_>>(),
    );

    let mut restored_paths = VisitedPaths(Vec::new());
    restored.visit_state(&StatePath::root(), &mut restored_paths)?;
    let mut expected_paths = manual_snapshot
        .iter()
        .map(|(path, _)| path.to_string())
        .collect::<Vec<_>>();
    restored_paths.0.sort();
    expected_paths.sort();
    assert_eq!(restored_paths.0, expected_paths);

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
