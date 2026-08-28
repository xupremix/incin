//! Families whose attributes describe a shape, a window, or a filter bank.
//!
//! Split out of `fixtures.rs`, which holds the families keyed on a plain
//! operand contract. The ones here are keyed on something more: an attribute
//! that has to be recomputed for every rank the harness walks, because it
//! names extents rather than a mode. A creation row carries the output shape,
//! a pooling row carries a window, a convolution row carries a filter bank,
//! and a normalization row carries the suffix it normalizes over. All four
//! are wrong at one end of a rule's rank range if they are written as
//! constants for the other.
//!
//! The families themselves live next to the shims that compute those
//! attributes, because reading one without the other says very little. What
//! keys a fixture to an operation, and why an operation without one is
//! counted rather than failed, is in `fixtures.rs`.

use incin_core::backend_authoring::{Execute, op};
use incin_core::exec::catalog::{
    AdaptivePool2dAttributes, ArangeAttributes, AvgPool2dAttributes, BatchNormAttributes,
    Conv1dAttributes, Conv2dAttributes, ConvTranspose2dAttributes, CreationAttributes,
    FullAttributes, LayerNormAttributes, LinspaceAttributes, PixelShuffleAttributes,
    Pool2dAttributes, UnfoldAttributes,
};
use incin_core::exec::{CanonicalError, Capabilities, ExecutionContext, Operation, TensorHandle};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::DeviceId;

use crate::conformance::fixtures::{
    Fixture, Operands, Role, Route, Subject, constant_attribute_shim, derived_attribute_shim,
    family, on_axis_zero, typed_family, with_epsilon,
};
use crate::conformance::operands::materialized_extents;
use crate::conformance::plan::AdvertisedTuple;

// The creation rows infer their output from their attributes, so the shape and
// the dtype both come from the tuple rather than from an operand. That is what
// makes them reachable at all, and it is also what keeps them honest: the
// dtype the row advertises is the dtype the invocation asks for.
derived_attribute_shim!(created_as, CreationAttributes, |tuple| CreationAttributes {
    shape: materialized_extents(tuple),
    dtype: tuple.dtype,
    device: DeviceId::cpu(),
});

derived_attribute_shim!(filled_with_two, FullAttributes, |tuple| FullAttributes {
    shape: materialized_extents(tuple),
    dtype: tuple.dtype,
    device: DeviceId::cpu(),
    value: 2.0,
});

derived_attribute_shim!(counting_up, ArangeAttributes, |tuple| ArangeAttributes {
    shape: materialized_extents(tuple),
    dtype: tuple.dtype,
    device: DeviceId::cpu(),
    start: 0.0,
    step: 1.0,
});

// The endpoint is the element count rather than one, so every interpolated
// value lands on a whole number. An integer dtype refuses a fractional value,
// and a unit span over more than two points produces nothing else.
derived_attribute_shim!(spanning_the_count, LinspaceAttributes, |tuple| {
    let shape = materialized_extents(tuple);
    let count = shape.iter().product::<usize>().max(1);
    LinspaceAttributes {
        shape,
        dtype: tuple.dtype,
        device: DeviceId::cpu(),
        start: 0.0,
        end: (count - 1) as f64,
    }
});

family!(
    creating,
    Operands::Nullary,
    created_as,
    [
        Zeros,
        Ones,
        UniformRandom,
        NormalRandom,
        VariableZeros,
        VariableOnes,
        VariableUniformRandom,
        VariableNormalRandom,
    ]
);

family!(creating_full, Operands::Nullary, filled_with_two, [Full]);
family!(creating_arange, Operands::Nullary, counting_up, [Arange]);
family!(
    creating_linspace,
    Operands::Nullary,
    spanning_the_count,
    [Linspace]
);

// A unit window with unit stride and no padding is the identity pool. The
// harness asks whether the tuple runs, not whether the window is interesting,
// and an identity is the one window every admitted extent can hold.
constant_attribute_shim!(
    unit_window,
    Pool2dAttributes,
    Pool2dAttributes {
        kernel: [1, 1],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
    }
);

constant_attribute_shim!(
    unit_average_window,
    AvgPool2dAttributes,
    AvgPool2dAttributes {
        kernel: [1, 1],
        stride: [1, 1],
        padding: [0, 0],
    }
);

// Asking for the extents the input already has, which is the adaptive pool's
// identity and the only output size that works for every rank on the ladder.
derived_attribute_shim!(pooled_to_itself, AdaptivePool2dAttributes, |tuple| {
    let dims = materialized_extents(tuple);
    let last = dims.len();
    AdaptivePool2dAttributes {
        output: [dims[last - 2], dims[last - 1]],
    }
});

constant_attribute_shim!(
    unit_slide,
    UnfoldAttributes,
    UnfoldAttributes {
        axis: 0,
        size: 1,
        step: 1,
    }
);

constant_attribute_shim!(
    unscaled,
    PixelShuffleAttributes,
    PixelShuffleAttributes { upscale_factor: 1 }
);

// Every extent is one under `UnaryScalar`, so any axis the attributes name is
// a length-one axis and squeezing it is legal. The rank ladder cannot do this:
// its extents are deliberately unequal and none of them is one.
family!(
    squeezing,
    Operands::UnaryScalar,
    on_axis_zero,
    [SqueezeExact]
);

family!(pooling_max, Operands::Unary, unit_window, [MaxPool2d]);
family!(
    pooling_average,
    Operands::Unary,
    unit_average_window,
    [AvgPool2d]
);
family!(
    pooling_adaptive,
    Operands::Unary,
    pooled_to_itself,
    [AdaptiveAvgPool2dExact]
);
family!(sliding, Operands::UnaryAxis, unit_slide, [Unfold]);
family!(shuffling, Operands::Unary, unscaled, [PixelShuffle]);

// ----------------------------------------------------------------------------
// Convolutions and the weighted normalizations
// ----------------------------------------------------------------------------
//
// The four families below are the ones a row-driven operand builder could not
// reach until `Role` carried a shape as well as a dtype. Each reads operands at
// two or three different ranks, and a capability row has one rank column, so
// the row states the loosest of them and the fixture states the rest.
//
// Every kernel is unit and every stride is one, so the output extent equals the
// input's and no ladder extent can underflow a window. Groups stay at one: the
// forward inference reads the bias against `weight[0]` and the transposed one
// against `weight[1] * groups`, and one bias role serves both only at that
// factor.

constant_attribute_shim!(
    unit_kernel_1d,
    Conv1dAttributes,
    Conv1dAttributes {
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
        has_bias: true,
    }
);

constant_attribute_shim!(
    unit_kernel_2d,
    Conv2dAttributes,
    Conv2dAttributes {
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
        has_bias: true,
    }
);

constant_attribute_shim!(
    unit_kernel_transposed,
    ConvTranspose2dAttributes,
    ConvTranspose2dAttributes {
        stride: [1, 1],
        padding: [0, 0],
        output_padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
        has_bias: true,
    }
);

// A normalized shape of one axis, which is the longest suffix every rank in the
// rule's range shares. Naming more axes would be a different question at each
// end of the range, and `LayerNormAttributes::validate` requires the suffix,
// the weight and the bias to agree exactly.
derived_attribute_shim!(normalizing_the_last_axis, LayerNormAttributes, |tuple| {
    let dims = materialized_extents(tuple);
    LayerNormAttributes {
        normalized_shape: dims.last().copied().into_iter().collect(),
        epsilon: 1e-5,
        has_bias: true,
    }
});

// Running statistics rather than affine parameters. Three operands is the arity
// this builder has, and the running pair is the half both rows can carry:
// `BatchNormAttributes::validate` refuses an inference batch norm that has no
// running mean, while a training one accepts the pair either way.
derived_attribute_shim!(tracking_running_statistics, BatchNormAttributes, |tuple| {
    BatchNormAttributes {
        epsilon: 1e-5,
        momentum: 0.1,
        training: tuple.training,
        has_weight: false,
        has_bias: false,
        has_running_mean: true,
        has_running_variance: true,
    }
});

typed_family!(
    convolving_1d,
    Operands::Triple,
    &[
        Role::Tuple,
        Role::ConvWeight { spatial: 1 },
        Role::OutputVector
    ],
    unit_kernel_1d,
    [Conv1dExact]
);

typed_family!(
    convolving_2d,
    Operands::Triple,
    &[
        Role::Tuple,
        Role::ConvWeight { spatial: 2 },
        Role::OutputVector
    ],
    unit_kernel_2d,
    [Conv2dExact]
);

typed_family!(
    convolving_transposed,
    Operands::Triple,
    &[
        Role::Tuple,
        Role::ConvTransposeWeight { spatial: 2 },
        Role::OutputVector
    ],
    unit_kernel_transposed,
    [ConvTranspose2d]
);

typed_family!(
    normalizing_layer,
    Operands::Triple,
    &[Role::Tuple, Role::TrailingVector, Role::TrailingVector],
    normalizing_the_last_axis,
    [LayerNorm]
);

typed_family!(
    normalizing_rms,
    Operands::Binary,
    &[Role::Tuple, Role::TrailingVector],
    with_epsilon,
    [RmsNorm]
);

typed_family!(
    normalizing_batch,
    Operands::Triple,
    &[Role::Tuple, Role::ChannelVector, Role::ChannelVector],
    tracking_running_statistics,
    [BatchNorm]
);
