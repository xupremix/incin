//! Backend-reusable semantic conformance vectors.
//!
//! These vectors contain no backend storage. FND-005 binds them to CPU eager
//! execution; later backends consume the same cases rather than defining their
//! own semantic expectations.

use crate::shapes::error::OperationKind;

/// Semantic edge represented by a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConformanceClass {
    /// Ordinary values with no special edge.
    Normal,
    /// Rank-zero scalar input.
    ScalarRank,
    /// Differing-shape operands combining under broadcasting.
    Broadcasting,
    /// Zero-element input.
    ZeroLength,
    /// Shapes that cannot legally combine.
    InvalidShape,
    /// An axis argument at or beyond the rank.
    InvalidAxis,
    /// A value at the edge of what the destination dtype can represent.
    DTypeBoundary,
    /// A not-a-number input.
    NaN,
    /// An infinity input.
    Infinity,
    /// A derivative-contract case exercised by finite differences.
    Gradient,
}

/// Expected disposition independent of a backend implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpectedDisposition {
    /// Computes successfully with the recorded expected values.
    Succeeds,
    /// Must fail with the framework's typed error.
    TypedError,
    /// NaN/Infinity flows through unchanged in class.
    IeeePropagates,
    /// Validated against the analytic gradient via central differences.
    FiniteDifference,
}

/// One storage-free semantic case reusable by every backend suite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformanceVector {
    /// Unique case name.
    pub name: &'static str,
    /// Operation under test.
    pub operation: OperationKind,
    /// Semantic edge class exercised.
    pub class: ConformanceClass,
    /// Input value sets, row-major per tensor.
    pub inputs: &'static [&'static [f64]],
    /// Shape of each input tensor.
    pub input_shapes: &'static [&'static [usize]],
    /// Expected flat output values.
    pub expected: &'static [f64],
    /// Expected output shape.
    pub expected_shape: &'static [usize],
    /// What outcome correctness means here.
    pub disposition: ExpectedDisposition,
}

const A: &[f64] = &[1.0, -2.0, 3.0, 4.0];
const B: &[f64] = &[2.0, 0.5, -1.0, 3.0];
const NAN: &[f64] = &[f64::NAN];
const INF: &[f64] = &[f64::INFINITY];
const EMPTY: &[f64] = &[];
const SCALAR: &[f64] = &[2.0];
const ZERO: &[f64] = &[0.0];
const ONE: &[f64] = &[1.0];
const FOUR: &[f64] = &[4.0];
const MINUS_THREE: &[f64] = &[-3.0];
const PAIR: &[f64] = &[2.0, 3.0];
const ZEROS4: &[f64] = &[0.0, 0.0, 0.0, 0.0];
const B3: &[f64] = &[2.0, 0.5, -1.0];
const DIV_L: &[f64] = &[8.0, 2.0, -6.0, 12.0];
const DIV_R: &[f64] = &[2.0, 1.0, -2.0, 4.0];
const MAT_A: &[f64] = &[1.0, 2.0, 3.0, 4.0];
const MAT_B: &[f64] = &[5.0, 6.0, 7.0, 8.0];
const SHAPE_2X2: &[usize] = &[2, 2];
const SHAPE_1: &[usize] = &[1];
const SHAPE_0X2: &[usize] = &[0, 2];
const SHAPE_SCALAR: &[usize] = &[];

/// Minimum frozen vector set. Operation-specific CPU adapters select the
/// applicable cases and may add stronger cases without changing this oracle.
pub static SEMANTIC_CONFORMANCE_VECTORS: &[ConformanceVector] = &[
    ConformanceVector {
        name: "normal-values",
        operation: OperationKind::Add,
        class: ConformanceClass::Normal,
        inputs: &[A, B],
        input_shapes: &[SHAPE_2X2, SHAPE_2X2],
        expected: &[3.0, -1.5, 2.0, 7.0],
        expected_shape: SHAPE_2X2,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "rank-zero-scalar",
        operation: OperationKind::Relu,
        class: ConformanceClass::ScalarRank,
        inputs: &[SCALAR],
        input_shapes: &[SHAPE_SCALAR],
        expected: SCALAR,
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "numpy-broadcast",
        operation: OperationKind::Add,
        class: ConformanceClass::Broadcasting,
        inputs: &[A, SCALAR],
        input_shapes: &[SHAPE_2X2, SHAPE_1],
        expected: &[3.0, 0.0, 5.0, 6.0],
        expected_shape: SHAPE_2X2,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "zero-length",
        operation: OperationKind::Relu,
        class: ConformanceClass::ZeroLength,
        inputs: &[EMPTY],
        input_shapes: &[SHAPE_0X2],
        expected: EMPTY,
        expected_shape: SHAPE_0X2,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "incompatible-broadcast",
        operation: OperationKind::Add,
        class: ConformanceClass::InvalidShape,
        inputs: &[A, B3],
        input_shapes: &[&[2, 2], &[3, 1]],
        expected: EMPTY,
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::TypedError,
    },
    ConformanceVector {
        name: "axis-equals-rank",
        operation: OperationKind::Softmax,
        class: ConformanceClass::InvalidAxis,
        inputs: &[A],
        input_shapes: &[SHAPE_2X2],
        expected: EMPTY,
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::TypedError,
    },
    ConformanceVector {
        name: "log-softmax-axis-equals-rank",
        operation: OperationKind::LogSoftmax,
        class: ConformanceClass::InvalidAxis,
        inputs: &[A],
        input_shapes: &[SHAPE_2X2],
        expected: EMPTY,
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::TypedError,
    },
    // Dtype casts follow Rust's `as` semantics: fractional values truncate
    // toward zero deterministically. Exact-conversion policy boundaries
    // (scalar readback, embedding indices) are separate checked paths and
    // are covered by their own suites; this row pins the cast contract.
    ConformanceVector {
        name: "dtype-cast-truncates-fractions",
        operation: OperationKind::ToDType,
        class: ConformanceClass::DTypeBoundary,
        inputs: &[&[255.5]],
        input_shapes: &[&[1]],
        expected: &[255.0],
        expected_shape: &[1],
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "nan-propagation",
        operation: OperationKind::Exp,
        class: ConformanceClass::NaN,
        inputs: &[NAN],
        input_shapes: &[SHAPE_1],
        expected: NAN,
        expected_shape: SHAPE_1,
        disposition: ExpectedDisposition::IeeePropagates,
    },
    ConformanceVector {
        name: "infinity-propagation",
        operation: OperationKind::Exp,
        class: ConformanceClass::Infinity,
        inputs: &[INF],
        input_shapes: &[SHAPE_1],
        expected: INF,
        expected_shape: SHAPE_1,
        disposition: ExpectedDisposition::IeeePropagates,
    },
    ConformanceVector {
        name: "central-finite-difference",
        operation: OperationKind::Add,
        class: ConformanceClass::Gradient,
        inputs: &[A],
        input_shapes: &[SHAPE_2X2],
        expected: EMPTY,
        expected_shape: SHAPE_2X2,
        disposition: ExpectedDisposition::FiniteDifference,
    },
    // --- elementwise, hand-computed --------------------------------------
    ConformanceVector {
        name: "mul-normal-values",
        operation: OperationKind::Mul,
        class: ConformanceClass::Normal,
        inputs: &[A, B],
        input_shapes: &[SHAPE_2X2, SHAPE_2X2],
        expected: &[2.0, -1.0, -3.0, 12.0],
        expected_shape: SHAPE_2X2,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "sub-normal-values",
        operation: OperationKind::Sub,
        class: ConformanceClass::Normal,
        inputs: &[A, B],
        input_shapes: &[SHAPE_2X2, SHAPE_2X2],
        expected: &[-1.0, -2.5, 4.0, 1.0],
        expected_shape: SHAPE_2X2,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "div-exact-quotients",
        operation: OperationKind::Div,
        class: ConformanceClass::Normal,
        inputs: &[DIV_L, DIV_R],
        input_shapes: &[SHAPE_2X2, SHAPE_2X2],
        expected: &[4.0, 2.0, 3.0, 3.0],
        expected_shape: SHAPE_2X2,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "neg-flips-sign",
        operation: OperationKind::Neg,
        class: ConformanceClass::Normal,
        inputs: &[A],
        input_shapes: &[SHAPE_2X2],
        expected: &[-1.0, 2.0, -3.0, -4.0],
        expected_shape: SHAPE_2X2,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "sigmoid-at-zero-is-one-half",
        operation: OperationKind::Sigmoid,
        class: ConformanceClass::ScalarRank,
        inputs: &[ZERO],
        input_shapes: &[SHAPE_SCALAR],
        expected: &[0.5],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "tanh-at-zero-is-zero",
        operation: OperationKind::Tanh,
        class: ConformanceClass::ScalarRank,
        inputs: &[ZERO],
        input_shapes: &[SHAPE_SCALAR],
        expected: &[0.0],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "log-of-one-is-zero",
        operation: OperationKind::Log,
        class: ConformanceClass::ScalarRank,
        inputs: &[ONE],
        input_shapes: &[SHAPE_SCALAR],
        expected: &[0.0],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "sqrt-of-four-is-two",
        operation: OperationKind::Sqrt,
        class: ConformanceClass::ScalarRank,
        inputs: &[FOUR],
        input_shapes: &[SHAPE_SCALAR],
        expected: &[2.0],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "abs-of-minus-three-is-three",
        operation: OperationKind::Abs,
        class: ConformanceClass::ScalarRank,
        inputs: &[MINUS_THREE],
        input_shapes: &[SHAPE_SCALAR],
        expected: &[3.0],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    // --- reductions -------------------------------------------------------
    ConformanceVector {
        name: "sum-all-elements",
        operation: OperationKind::SumAll,
        class: ConformanceClass::Normal,
        inputs: &[A],
        input_shapes: &[SHAPE_2X2],
        expected: &[6.0],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "mean-all-elements",
        operation: OperationKind::MeanAll,
        class: ConformanceClass::Normal,
        inputs: &[A],
        input_shapes: &[SHAPE_2X2],
        expected: &[1.5],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    // A pair two thousand apart along the reduced axis. The smaller entry
    // contributes `e^-2000`, which is zero in f32 and would be zero in f64 too,
    // so the answer is the larger entry exactly, and the row can be read without
    // knowing the accumulation order.
    //
    // The same pair rules out the naive spelling. `log(sum(exp(x)))` reaches
    // `exp(1000)` first, which is infinity in f32, and the logarithm of infinity
    // is infinity rather than 1000. Shifting by the axis maximum is what this
    // row exists to require.
    ConformanceVector {
        name: "logsumexp-is-the-maximum-when-the-rest-underflows",
        operation: OperationKind::LogSumExpDim,
        class: ConformanceClass::Normal,
        inputs: &[&[1000.0, -1000.0]],
        input_shapes: &[&[2]],
        expected: &[1000.0],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "max-all-elements",
        operation: OperationKind::MaxAll,
        class: ConformanceClass::Normal,
        inputs: &[A],
        input_shapes: &[SHAPE_2X2],
        expected: &[4.0],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "min-all-elements",
        operation: OperationKind::MinAll,
        class: ConformanceClass::Normal,
        inputs: &[A],
        input_shapes: &[SHAPE_2X2],
        expected: &[-2.0],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    ConformanceVector {
        name: "prod-all-elements",
        operation: OperationKind::ProdAll,
        class: ConformanceClass::Normal,
        inputs: &[PAIR],
        input_shapes: &[&[2]],
        expected: &[6.0],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
    // --- matmul -----------------------------------------------------------
    ConformanceVector {
        name: "matmul-hand-computed",
        operation: OperationKind::MatMulExact,
        class: ConformanceClass::Normal,
        inputs: &[MAT_A, MAT_B],
        input_shapes: &[SHAPE_2X2, SHAPE_2X2],
        expected: &[19.0, 22.0, 43.0, 50.0],
        expected_shape: SHAPE_2X2,
        disposition: ExpectedDisposition::Succeeds,
    },
    // --- normalization ----------------------------------------------------
    ConformanceVector {
        name: "softmax-uniform-pair-is-half-half",
        operation: OperationKind::Softmax,
        class: ConformanceClass::Normal,
        inputs: &[&[0.0, 0.0]],
        input_shapes: &[&[2]],
        expected: &[0.5, 0.5],
        expected_shape: SHAPE_1,
        disposition: ExpectedDisposition::Succeeds,
    },
    // A pair a thousand apart, chosen so the answer is exact and so that the
    // composition cannot produce it. The larger logit takes essentially all the
    // mass, so its log-probability is zero and the other's is the gap itself,
    // and both are representable without rounding. The operation is still
    // transcendental and consumers still grant it that tolerance; the point of
    // choosing these inputs is that the row does not need the allowance, so a
    // backend cannot hide a wrong answer inside it.
    //
    // The same row rules out answering `log_softmax` with `log(softmax(x))`.
    // That path exponentiates the gap, which underflows to zero in f32, and the
    // logarithm of zero is negative infinity rather than `-1000`.
    ConformanceVector {
        name: "log-softmax-keeps-a-logit-a-thousand-below-the-maximum",
        operation: OperationKind::LogSoftmax,
        class: ConformanceClass::Normal,
        inputs: &[&[0.0, -1000.0]],
        input_shapes: &[&[2]],
        expected: &[0.0, -1000.0],
        expected_shape: SHAPE_1,
        disposition: ExpectedDisposition::Succeeds,
    },
    // --- loss -------------------------------------------------------------
    ConformanceVector {
        name: "mse-loss-of-ones-versus-a",
        operation: OperationKind::MseLoss,
        class: ConformanceClass::Normal,
        inputs: &[ZEROS4, A],
        input_shapes: &[SHAPE_2X2, SHAPE_2X2],
        expected: &[7.5],
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::Succeeds,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    #[test]
    fn required_semantic_edges_are_present_once_or_more() {
        let classes: BTreeSet<_> = SEMANTIC_CONFORMANCE_VECTORS
            .iter()
            .map(|case| case.class)
            .collect();
        for required in [
            ConformanceClass::Normal,
            ConformanceClass::ScalarRank,
            ConformanceClass::Broadcasting,
            ConformanceClass::ZeroLength,
            ConformanceClass::InvalidShape,
            ConformanceClass::InvalidAxis,
            ConformanceClass::DTypeBoundary,
            ConformanceClass::NaN,
            ConformanceClass::Infinity,
            ConformanceClass::Gradient,
        ] {
            assert!(
                classes.contains(&required),
                "missing {required:?} conformance class"
            );
        }
        for case in SEMANTIC_CONFORMANCE_VECTORS {
            assert!(case.operation.is_exact());
            assert!(crate::exec::catalog_entry(case.operation).is_some());
        }
    }
}
