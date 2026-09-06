//! Polar-to-Cartesian as a differentiable custom operation.
//!
//! A robot arm reports each joint as an angle and each link as a length, but
//! the controller plans in Cartesian space. The conversion is real, small, and
//! exactly the shape the catalog cannot express: it takes *two* inputs (radius
//! `r` and angle `theta`) and returns *two* outputs (`x` and `y`), and the
//! backward pass needs the forward inputs back (the Jacobian is written in
//! `cos theta` and `sin theta`).
//!
//! This example shows the four pieces a differentiable custom operation needs,
//! each asserted rather than printed:
//!
//! 1. `Operation` with a per-operand contract (`infer_outputs`), including the
//!    two-output inference the single-output `calibration_update` example does
//!    not exercise.
//! 2. `Execute` with a real CPU kernel, run through the validated production
//!    path (`execute_shaped_n` with one `ShapeValue` per output), not around it.
//! 3. A backward recipe per output, assembled into core `TapeNode`s and run
//!    through `incin_core::exec::tape::backward` -- the same reverse walk the
//!    CPU backend calls, with the same accumulation semantics. A tensor
//!    consumed by both outputs receives the *sum* of both contributions; the
//!    finite-difference check below would catch an overwrite.
//! 4. Utilization: gradient descent that fits `(r, theta)` to a target point
//!    through the custom backward, proving the gradients are good for
//!    something beyond their own test.
//!
//! Multi-output operations keep the explicit per-backend `tape_record` path --
//! one node per output cannot be derived from a single return type, so there
//! is no `DifferentiableOp` blanket impl for this shape. This example shows
//! the node shape directly: an in-tree backend moves the node construction of
//! section 3 into its `Execute` impl and records via its public `tape_record`
//! (CPU, WGPU, CUDA, Metal), so its custom nodes and the built-in ones share
//! one graph. The recipe and the walk
//! are identical either way, which is what this example pins down.
//!
//! Run it with:
//! `cargo run -p incin-backends --features cpu --example polar_cartesian`

use std::borrow::Cow;

extern crate incin_core as incin;

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};

/// The CPU backend the kernels below are written for.
type CpuBackend = CpuBackendImpl;
use incin_core::backend_authoring::operations::NoAttributes;
use incin_core::backend_authoring::{
    DescriptorError, Execute, ExecutionContext, ExecutionRequest, LogicalTensorMeta, Operation,
    OperationKey, ShapeBuf, execute_shaped, execute_shaped_n,
};
use incin_core::exec::TapeStorage as _;
use incin_core::exec::tape::{self, TapeNode};
use incin_core::prelude::{BackendError, DTypeId, DeviceId, Result, ShapeValue, s};
use incin_core::shapes::error::OperationKind;

/// Three points: (r, theta) = (2, 0), (1, pi/2), (2, pi/6). Every forward
/// value below is then checkable by hand: x = [2, 0, sqrt(3)], y = [0, 1, 1].
const N: usize = 3;
const RADIUS: [f64; N] = [2.0, 1.0, 2.0];
const THETA: [f64; N] = [
    0.0,
    core::f64::consts::FRAC_PI_2,
    core::f64::consts::FRAC_PI_6,
];

fn invalid(attribute: &'static str, reason: &'static str) -> DescriptorError {
    DescriptorError::InvalidAttribute {
        operation: OperationKind::Pointwise,
        attribute,
        reason,
    }
}

fn expect_f64_vector(
    meta: &LogicalTensorMeta,
    what: &'static str,
) -> core::result::Result<Vec<usize>, DescriptorError> {
    match meta.dtype {
        Some(actual) if actual == DTypeId::F64.descriptor() => {}
        Some(_) => return Err(invalid(what, "operand must be f64")),
        None => return Err(invalid(what, "operand element type is unknown")),
    }
    meta.shape
        .as_ref()
        .map(|shape| shape.dims().to_vec())
        .ok_or_else(|| invalid(what, "operand shape is unknown"))
}

/// Allocate an f64 output, preserving a construction failure inside a
/// structured backend error rather than relabelling it as an input refusal:
/// the inputs passed inference, so a failure here is the kernel's, not theirs.
fn contiguous_f64(
    values: Vec<f64>,
    dims: &[usize],
    operation: OperationKind,
) -> core::result::Result<CpuStorage, BackendError> {
    CpuStorage::try_from_contiguous(CpuBuffer::F64(values), dims).map_err(|error| {
        BackendError::Execution {
            operation,
            message: incin_core::prelude::ErrorMessage::new(error.to_string()),
        }
    })
}

// ---------------------------------------------------------------------------
// Operation 1: polar -> Cartesian.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PolarToCartesian;

impl Operation for PolarToCartesian {
    type Attributes = incin_core::exec::catalog::NoAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("example.org"),
        name: Cow::Borrowed("polar_to_cartesian"),
        version: 1,
    };

    /// Two f64 inputs of identical shape; two f64 outputs of that same shape.
    /// The catalog's per-operation rules admit one output shape per row, so a
    /// two-output inference like this one is custom-operation territory even
    /// before the backward pass enters the picture. It dispatches through
    /// `execute_shaped_n` with one `ShapeValue` per output, so the descriptor
    /// cross-checks both geometries instead of trusting the frontend.
    fn infer_outputs(
        _attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> core::result::Result<Vec<LogicalTensorMeta>, DescriptorError> {
        if inputs.len() != 2 {
            return Err(invalid(
                "inputs",
                "polar_to_cartesian takes radius and angle",
            ));
        }
        let radius_dims = expect_f64_vector(&inputs[0], "radius")?;
        let angle_dims = expect_f64_vector(&inputs[1], "angle")?;
        if radius_dims != angle_dims {
            return Err(invalid("angle", "radius and angle must share one shape"));
        }
        let output = |dims: &[usize]| LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(dims)),
            dtype: Some(DTypeId::F64.descriptor()),
            device: Some(DeviceId::cpu()),
        };
        Ok(vec![output(&radius_dims), output(&radius_dims)])
    }
}

impl Execute<PolarToCartesian> for CpuBackend {
    /// `(x, y)`, each shaped like the inputs.
    type Output = (CpuStorage, CpuStorage);

    fn execute(
        &self,
        request: ExecutionRequest<'_, PolarToCartesian, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let read = |index: usize| -> core::result::Result<&CpuStorage, BackendError> {
            request.inputs[index]
                .downcast_ref::<CpuStorage>()
                .ok_or(BackendError::InvalidInput {
                    operation: OperationKind::Pointwise,
                    reason: "polar_to_cartesian expects CPU storage",
                })
        };
        let radius = read(0)?;
        let angle = read(1)?;
        let dims = radius.metadata().shape.dims().to_vec();
        let count = dims.iter().product::<usize>().max(1);

        // `get` reads any element type back as f64, so the kernel stays
        // rank-agnostic: one flat odometer over the validated shared shape.
        let mut xs = Vec::with_capacity(count);
        let mut ys = Vec::with_capacity(count);
        let mut index = vec![0usize; dims.len().max(1)];
        for _ in 0..count {
            let r = radius.get(&index);
            let t = angle.get(&index);
            xs.push(r * t.cos());
            ys.push(r * t.sin());
            odometer(&mut index, &dims);
        }
        let x = contiguous_f64(xs, &dims, OperationKind::Pointwise)?;
        let y = contiguous_f64(ys, &dims, OperationKind::Pointwise)?;
        Ok((x, y))
    }
}

/// Odometer increment over row-major `dims`; a no-op for scalar shapes, whose
/// single element sits at `[]`.
fn odometer(index: &mut [usize], dims: &[usize]) {
    for (i, extent) in index.iter_mut().zip(dims.iter()).rev() {
        *i += 1;
        if *i < *extent {
            return;
        }
        *i = 0;
    }
}

// ---------------------------------------------------------------------------
// Operation 2: squared-error readout (scaffolding).
// ---------------------------------------------------------------------------
//
// A loss has to come from somewhere to close the graph, and the catalog's own
// `mse_loss` would do -- but routing the example's loss through a built-in
// would hide the multi-node chaining this example exists to show. This readout
// is deliberately boring: sum of squared errors against a target point, with
// the textbook backward. The interesting gradients are the polar ones above.

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct SquaredErrorAttributes {
    tx: Vec<f64>,
    ty: Vec<f64>,
}

#[derive(Debug, Clone)]
struct SquaredError;

fn invalid_loss(attribute: &'static str, reason: &'static str) -> DescriptorError {
    DescriptorError::InvalidAttribute {
        operation: OperationKind::Reduction,
        attribute,
        reason,
    }
}

impl Operation for SquaredError {
    type Attributes = SquaredErrorAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("example.org"),
        name: Cow::Borrowed("squared_error"),
        version: 1,
    };

    /// Two f64 vectors of one shared length; one scalar f64 loss.
    fn infer_outputs(
        attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> core::result::Result<Vec<LogicalTensorMeta>, DescriptorError> {
        if inputs.len() != 2 {
            return Err(invalid_loss("inputs", "squared_error takes x and y"));
        }
        let x_dims = expect_f64_vector(&inputs[0], "x")
            .map_err(|_| invalid_loss("x", "operand must be f64 with a known shape"))?;
        let y_dims = expect_f64_vector(&inputs[1], "y")
            .map_err(|_| invalid_loss("y", "operand must be f64 with a known shape"))?;
        if x_dims != y_dims {
            return Err(invalid_loss("y", "x and y must share one shape"));
        }
        if attributes.tx.len() != x_dims.iter().product::<usize>()
            || attributes.ty.len() != x_dims.iter().product::<usize>()
        {
            return Err(invalid_loss(
                "target",
                "one target coordinate is required per element",
            ));
        }
        Ok(vec![LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(&[])),
            dtype: Some(DTypeId::F64.descriptor()),
            device: Some(DeviceId::cpu()),
        }])
    }
}

impl Execute<SquaredError> for CpuBackend {
    /// The scalar loss.
    type Output = CpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, SquaredError, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let read = |index: usize| -> core::result::Result<&CpuStorage, BackendError> {
            request.inputs[index]
                .downcast_ref::<CpuStorage>()
                .ok_or(BackendError::InvalidInput {
                    operation: OperationKind::Reduction,
                    reason: "squared_error expects CPU storage",
                })
        };
        let x = read(0)?;
        let y = read(1)?;
        let attrs = request.operation.descriptor().attributes();
        let dims = x.metadata().shape.dims().to_vec();
        let count = dims.iter().product::<usize>().max(1);

        let mut total = 0.0;
        let mut index = vec![0usize; dims.len().max(1)];
        for flat in 0..count {
            let dx = x.get(&index) - attrs.tx[flat];
            let dy = y.get(&index) - attrs.ty[flat];
            total += dx * dx + dy * dy;
            odometer(&mut index, &dims);
        }
        contiguous_f64(vec![total], &[], OperationKind::Reduction)
    }
}

// ---------------------------------------------------------------------------
// The graph: forward through validated dispatch, backward through the walk.
// ---------------------------------------------------------------------------

/// Everything one training step needs: the live storages and the nodes that
/// connect them, in forward order. An in-tree backend builds these three
/// nodes inside its `Execute` impls and pushes them onto its thread-local
/// tape; the recipe closures and the walk below are byte-for-byte what it
/// would push.
struct Graph {
    radius: CpuStorage,
    theta: CpuStorage,
    x: CpuStorage,
    y: CpuStorage,
    loss: CpuStorage,
    nodes: Vec<TapeNode<CpuStorage>>,
}

fn storage(values: &[f64], dims: &[usize]) -> Result<CpuStorage> {
    CpuStorage::try_from_contiguous(CpuBuffer::F64(values.to_vec()), dims)
}

/// Forward both operations through validated dispatch, then record the three
/// backward recipes. Two nodes share the polar inputs -- one per output --
/// because a `TapeNode` names a single `output_id`; the walk sums both
/// contributions at `radius` and `theta`, which is exactly the accumulation
/// the finite-difference check holds accountable.
fn forward(
    context: &ExecutionContext<CpuBackend>,
    radius_vals: &[f64],
    theta_vals: &[f64],
    tx: &[f64],
    ty: &[f64],
) -> Result<Graph> {
    let radius = storage(radius_vals, &[N])?;
    let theta = storage(theta_vals, &[N])?;
    let handles = [
        incin_core::exec::TensorHandle::from_storage::<CpuBackend, f64, _>(&radius),
        incin_core::exec::TensorHandle::from_storage::<CpuBackend, f64, _>(&theta),
    ];
    // Typed dispatch with one proof per output: the descriptor checks both
    // inferred geometries against these instead of trusting the frontend.
    let expected = (
        ShapeValue::<s![3]>::try_new(ShapeBuf::from_slice(&[N]))?,
        ShapeValue::<s![3]>::try_new(ShapeBuf::from_slice(&[N]))?,
    );
    let (x, y) = execute_shaped_n::<PolarToCartesian, CpuBackend, _>(
        context,
        NoAttributes,
        &handles,
        &expected,
    )?;

    let loss_handles = [
        incin_core::exec::TensorHandle::from_storage::<CpuBackend, f64, _>(&x),
        incin_core::exec::TensorHandle::from_storage::<CpuBackend, f64, _>(&y),
    ];
    let loss_shape = ShapeValue::<incin_core::prelude::Dyn>::try_new(ShapeBuf::from_slice(&[]))?;
    let loss = execute_shaped::<SquaredError, CpuBackend, incin_core::prelude::Dyn>(
        context,
        SquaredErrorAttributes {
            tx: tx.to_vec(),
            ty: ty.to_vec(),
        },
        &loss_handles,
        &loss_shape,
    )?;

    // Saved values are captured by the recipe closures -- the core never names
    // a storage type, which is what keeps the node backend-neutral.
    let r_saved = radius.clone();
    let t_saved = theta.clone();
    let node_x = TapeNode {
        output_id: x.id(),
        input_ids: vec![radius.id(), theta.id()],
        backward: Box::new(move |grad_x: &CpuStorage| {
            // dx/dr = cos t, dx/dt = -r sin t.
            let mut dr = Vec::with_capacity(N);
            let mut dt = Vec::with_capacity(N);
            for i in 0..N {
                let t = t_saved.get(&[i]);
                let r = r_saved.get(&[i]);
                let g = grad_x.get(&[i]);
                dr.push(g * t.cos());
                dt.push(g * -r * t.sin());
            }
            Ok(vec![
                CpuStorage::try_from_contiguous(CpuBuffer::F64(dr), [N])?,
                CpuStorage::try_from_contiguous(CpuBuffer::F64(dt), [N])?,
            ])
        }),
    };
    let r_saved = radius.clone();
    let t_saved = theta.clone();
    let node_y = TapeNode {
        output_id: y.id(),
        input_ids: vec![radius.id(), theta.id()],
        backward: Box::new(move |grad_y: &CpuStorage| {
            // dy/dr = sin t, dy/dt = r cos t.
            let mut dr = Vec::with_capacity(N);
            let mut dt = Vec::with_capacity(N);
            for i in 0..N {
                let t = t_saved.get(&[i]);
                let r = r_saved.get(&[i]);
                let g = grad_y.get(&[i]);
                dr.push(g * t.sin());
                dt.push(g * r * t.cos());
            }
            Ok(vec![
                CpuStorage::try_from_contiguous(CpuBuffer::F64(dr), [N])?,
                CpuStorage::try_from_contiguous(CpuBuffer::F64(dt), [N])?,
            ])
        }),
    };
    let x_saved = x.clone();
    let y_saved = y.clone();
    let tx_saved = tx.to_vec();
    let ty_saved = ty.to_vec();
    let node_loss = TapeNode {
        output_id: loss.id(),
        input_ids: vec![x.id(), y.id()],
        backward: Box::new(move |seed: &CpuStorage| {
            // dL/dx = 2(x - tx), dL/dy = 2(y - ty), times the scalar seed.
            let g = seed.get(&[]);
            let mut gx = Vec::with_capacity(N);
            let mut gy = Vec::with_capacity(N);
            for i in 0..N {
                gx.push(2.0 * (x_saved.get(&[i]) - tx_saved[i]) * g);
                gy.push(2.0 * (y_saved.get(&[i]) - ty_saved[i]) * g);
            }
            Ok(vec![
                CpuStorage::try_from_contiguous(CpuBuffer::F64(gx), [N])?,
                CpuStorage::try_from_contiguous(CpuBuffer::F64(gy), [N])?,
            ])
        }),
    };

    Ok(Graph {
        radius,
        theta,
        x,
        y,
        loss,
        nodes: vec![node_x, node_y, node_loss],
    })
}

/// Analytic gradients of the loss with respect to radius and angle: the three
/// nodes through the core reverse walk, which seeds the scalar loss with one
/// and sums both polar contributions at each shared input.
fn analytic_gradients(graph: Graph) -> Result<(Vec<f64>, Vec<f64>)> {
    let Graph {
        radius,
        theta,
        loss,
        nodes,
        ..
    } = graph;
    let grads = tape::backward(nodes, &loss)?;
    let read = |storage: &CpuStorage| -> Vec<f64> {
        let id = storage.id();
        let grad = grads
            .get(id)
            .unwrap_or_else(|| panic!("backward did not reach an input"));
        (0..N).map(|i| grad.get(&[i])).collect()
    };
    Ok((read(&radius), read(&theta)))
}

fn approx_eq(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: got {actual}, want {expected}"
    );
}

fn main() -> Result<()> {
    use incin_core::exec::catalog::NoAttributes;

    let context = ExecutionContext::new(CpuBackend::default());

    // Targets with round residuals: x - tx = [1, 0, sqrt(3) - 1] and
    // y - ty = [0, 1, 0], so every gradient below is checkable by hand.
    let sqrt3 = 3.0f64.sqrt();
    let tx = [1.0, 0.0, 1.0];
    let ty = [0.0, 0.0, 1.0];

    // --- 1. Forward values, against the textbook answers. ---
    let graph = forward(&context, &RADIUS, &THETA, &tx, &ty)?;
    let xs: Vec<f64> = (0..N).map(|i| graph.x.get(&[i])).collect();
    let ys: Vec<f64> = (0..N).map(|i| graph.y.get(&[i])).collect();
    println!("x = {xs:?}\ny = {ys:?}");
    for (actual, expected) in xs.iter().zip([2.0, 0.0, sqrt3]) {
        approx_eq(*actual, expected, 1e-12, "x");
    }
    for (actual, expected) in ys.iter().zip([0.0, 1.0, 1.0]) {
        approx_eq(*actual, expected, 1e-12, "y");
    }
    approx_eq(
        graph.loss.get(&[]),
        1.0 + 0.0 + (sqrt3 - 1.0).powi(2) + 1.0,
        1e-12,
        "loss",
    );

    // --- 2. Analytic gradients, against hand differentiation. ---
    // gx = 2(x - tx) = [2, 0, 2(sqrt(3) - 1)], gy = 2(y - ty) = [0, 2, 0].
    // dr = gx cos t + gy sin t = [2, 2, (sqrt(3) - 1) sqrt(3)],
    // dt = r(-gx sin t + gy cos t) = [0, 0, -2(sqrt(3) - 1)].
    let (dr, dt) = analytic_gradients(graph)?;
    println!("dr = {dr:?}\ndt = {dt:?}");
    for (actual, expected) in dr.iter().zip([2.0, 2.0, (sqrt3 - 1.0) * sqrt3]) {
        approx_eq(*actual, expected, 1e-12, "dr");
    }
    for (actual, expected) in dt.iter().zip([0.0, 0.0, -2.0 * (sqrt3 - 1.0)]) {
        approx_eq(*actual, expected, 1e-12, "dt");
    }

    // --- 3. Finite differences over every input element. ---
    // The hand check pins one point exactly; this sweep holds the whole
    // backward accountable, including the accumulation at the shared inputs.
    let eps = 1e-5;
    let loss_of = |r: &[f64], t: &[f64]| -> f64 {
        forward(&context, r, t, &tx, &ty)
            .expect("forward must succeed for in-domain perturbations")
            .loss
            .get(&[])
    };
    let mut worst = 0.0f64;
    for (input, name) in [(&RADIUS, "r"), (&THETA, "theta")] {
        for i in 0..N {
            let mut plus = *input;
            let mut minus = *input;
            plus[i] += eps;
            minus[i] -= eps;
            let numeric = if name == "r" {
                (loss_of(&plus, &THETA) - loss_of(&minus, &THETA)) / (2.0 * eps)
            } else {
                (loss_of(&RADIUS, &plus) - loss_of(&RADIUS, &minus)) / (2.0 * eps)
            };
            let analytic = if name == "r" { dr[i] } else { dt[i] };
            let denom = analytic.abs().max(numeric.abs()).max(1e-6);
            worst = worst.max((analytic - numeric).abs() / denom);
        }
    }
    println!("finite-difference worst relative error: {worst:.2e}");
    assert!(
        worst < 1e-4,
        "analytic gradients disagree with finite differences: {worst}"
    );

    // --- 4. The per-operand contracts refuse what they must. ---
    let meta = |dims: &[usize], dtype: DTypeId| LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(dims)),
        dtype: Some(dtype.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    assert!(
        PolarToCartesian::infer_outputs(
            &NoAttributes,
            &[meta(&[N], DTypeId::F64), meta(&[N], DTypeId::F64)]
        )
        .is_ok(),
        "the well-formed polar operand list must be accepted"
    );
    assert!(
        PolarToCartesian::infer_outputs(&NoAttributes, &[meta(&[N], DTypeId::F64)]).is_err(),
        "one operand must be refused"
    );
    assert!(
        PolarToCartesian::infer_outputs(
            &NoAttributes,
            &[meta(&[N], DTypeId::F32), meta(&[N], DTypeId::F64)]
        )
        .is_err(),
        "an f32 radius must be refused"
    );
    assert!(
        PolarToCartesian::infer_outputs(
            &NoAttributes,
            &[meta(&[N], DTypeId::F64), meta(&[N + 1], DTypeId::F64)]
        )
        .is_err(),
        "mismatched shapes must be refused"
    );
    // Two outputs, not one: the inference a single-output catalog row cannot do.
    let inferred = PolarToCartesian::infer_outputs(
        &NoAttributes,
        &[meta(&[N], DTypeId::F64), meta(&[N], DTypeId::F64)],
    )
    .expect("well-formed polar operands");
    assert_eq!(inferred.len(), 2, "polar_to_cartesian infers two outputs");
    assert!(
        inferred
            .iter()
            .all(|out| out.shape.as_ref().is_some_and(|s| s.dims() == [N])),
        "both outputs carry the shared shape"
    );

    let good_tx: Vec<f64> = vec![0.0; N];
    let good = SquaredErrorAttributes {
        tx: good_tx.clone(),
        ty: good_tx,
    };
    assert!(
        SquaredError::infer_outputs(&good, &[meta(&[N], DTypeId::F64), meta(&[N], DTypeId::F64)])
            .is_ok(),
        "the well-formed loss operand list must be accepted"
    );
    assert!(
        SquaredError::infer_outputs(
            &SquaredErrorAttributes {
                tx: vec![0.0; N + 1],
                ty: vec![0.0; N],
            },
            &[meta(&[N], DTypeId::F64), meta(&[N], DTypeId::F64)]
        )
        .is_err(),
        "a target longer than the data must be refused"
    );
    println!("per-operand contracts: short, mistyped, mismatched and ragged lists all refused");

    // --- 5. Utilization: fit (r, theta) to a target point by gradient descent. ---
    // True point (2, 0, sqrt(3)) / (0, 1, 1), starting from (1, 1, 1) at
    // 0.5 rad everywhere. Only the custom backward guides the descent.
    let fit_tx = [2.0, 0.0, sqrt3];
    let fit_ty = [0.0, 1.0, 1.0];
    let mut r_fit = [1.0; N];
    let mut t_fit = [0.5; N];
    let learning_rate = 0.1;
    let steps = 400;
    for step in 0..steps {
        let graph = forward(&context, &r_fit, &t_fit, &fit_tx, &fit_ty)?;
        let (gr, gt) = analytic_gradients(graph)?;
        for i in 0..N {
            r_fit[i] -= learning_rate * gr[i];
            t_fit[i] -= learning_rate * gt[i];
        }
        if step % 100 == 0 {
            let check = forward(&context, &r_fit, &t_fit, &fit_tx, &fit_ty)?;
            println!("step {step:>3}: loss = {:.6}", check.loss.get(&[]));
        }
    }
    let fitted = forward(&context, &r_fit, &t_fit, &fit_tx, &fit_ty)?;
    let final_loss = fitted.loss.get(&[]);
    println!("step {steps}: loss = {final_loss:.2e} (r, theta) = ({r_fit:?}, {t_fit:?})");
    assert!(
        final_loss < 1e-6,
        "gradient descent through the custom backward did not converge: {final_loss}"
    );
    for i in 0..N {
        approx_eq(fitted.x.get(&[i]), fit_tx[i], 1e-3, "fitted x");
        approx_eq(fitted.y.get(&[i]), fit_ty[i], 1e-3, "fitted y");
    }
    println!("\nconverged: the custom backward fits (r, theta) to the target point");
    Ok(())
}
