//! Backend-reusable semantic conformance vectors.
//!
//! These vectors contain no backend storage. FND-005 binds them to CPU eager
//! execution; later backends consume the same cases rather than defining their
//! own semantic expectations.

use crate::shapes::error::OperationKind;

/// Semantic edge represented by a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConformanceClass {
    Normal,
    ScalarRank,
    Broadcasting,
    ZeroLength,
    InvalidShape,
    InvalidAxis,
    DTypeBoundary,
    NaN,
    Infinity,
    Gradient,
}

/// Expected disposition independent of a backend implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpectedDisposition {
    Succeeds,
    TypedError,
    IeeePropagates,
    FiniteDifference,
}

/// One storage-free semantic case reusable by every backend suite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformanceVector {
    pub name: &'static str,
    pub operation: OperationKind,
    pub class: ConformanceClass,
    pub inputs: &'static [&'static [f64]],
    pub input_shapes: &'static [&'static [usize]],
    pub expected: &'static [f64],
    pub expected_shape: &'static [usize],
    pub disposition: ExpectedDisposition,
}

const A: &[f64] = &[1.0, -2.0, 3.0, 4.0];
const B: &[f64] = &[2.0, 0.5, -1.0, 3.0];
const NAN: &[f64] = &[f64::NAN];
const INF: &[f64] = &[f64::INFINITY];
const EMPTY: &[f64] = &[];
const SCALAR: &[f64] = &[2.0];
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
        inputs: &[A, B],
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
        name: "checked-dtype-boundary",
        operation: OperationKind::ToDType,
        class: ConformanceClass::DTypeBoundary,
        inputs: &[&[255.0, 256.0]],
        input_shapes: &[&[2]],
        expected: EMPTY,
        expected_shape: SHAPE_SCALAR,
        disposition: ExpectedDisposition::TypedError,
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
