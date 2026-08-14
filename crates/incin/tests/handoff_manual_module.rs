//! Human-handoff acceptance fixtures.
//!
//! The manual implementation is deliberately written without `#[module]`.
//! The macro fixture below exercises the same public traversal contracts.

#![cfg(all(feature = "cpu", feature = "target-api"))]

use incin::nn::{Module, Parameters, StateDict};
use incin::prelude::*;
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

impl Parameters<CpuBackend> for ManualLinear {
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut BTreeMap<String, <CpuBackend as VariableBackend>::RawVar>,
    ) {
        let child_prefix = if prefix.is_empty() {
            "layer".to_owned()
        } else {
            format!("{prefix}.layer")
        };
        self.layer.named_parameters(&child_prefix, map);
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

#[test]
fn manual_and_macro_modules_have_equivalent_state_and_forward_behavior() -> Result<()> {
    let target = Cpu;
    let manual = ManualLinear {
        layer: Linear::build(())?,
    };
    let macro_layer = MacroLinear {
        layer: Linear::build(())?,
    };

    assert_eq!(manual.parameters().keys().collect::<Vec<_>>(), macro_layer.parameters().keys().collect::<Vec<_>>());
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

    Ok(())
}
