//! Backend-neutral normalization of tensor iteration metadata.
//!
//! Operation implementations consume normalized, output-rank strides. A
//! broadcast dimension has stride zero, so kernels do not need separate
//! input-shape branches in their inner loops.

use alloc::vec::Vec;
use kindle_core::prelude::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OperandIteration {
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}

impl OperandIteration {
    #[inline]
    pub(crate) fn physical_index(&self, mut flat_index: usize, output_shape: &[usize]) -> usize {
        debug_assert_eq!(self.strides.len(), output_shape.len());
        let mut physical = self.offset;
        for axis in (0..output_shape.len()).rev() {
            let dimension = output_shape[axis];
            if dimension != 0 {
                let coordinate = flat_index % dimension;
                flat_index /= dimension;
                physical += coordinate * self.strides[axis];
            }
        }
        physical
    }

    pub(crate) fn max_physical_index(&self, output_shape: &[usize]) -> Result<Option<usize>> {
        if self.strides.len() != output_shape.len() {
            return Err(iteration_shape_error(output_shape, &self.strides));
        }
        if output_shape.contains(&0) {
            return Ok(None);
        }
        let max_offset = output_shape.iter().zip(&self.strides).try_fold(
            0usize,
            |offset, (&dimension, &stride)| {
                let axis_offset =
                    dimension
                        .saturating_sub(1)
                        .checked_mul(stride)
                        .ok_or_else(|| {
                            Error::Msg(format!(
                                "stride overflow building iteration bounds for shape \
                                 {output_shape:?} and strides {:?}",
                                self.strides
                            ))
                        })?;
                offset.checked_add(axis_offset).ok_or_else(|| {
                    Error::Msg(format!(
                        "offset overflow building iteration bounds for shape \
                             {output_shape:?} and strides {:?}",
                        self.strides
                    ))
                })
            },
        )?;
        self.offset
            .checked_add(max_offset)
            .map(Some)
            .ok_or_else(|| Error::Msg("storage offset overflow in iteration plan".into()))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OperandLayout<'a> {
    pub(crate) shape: &'a [usize],
    pub(crate) strides: &'a [usize],
    pub(crate) offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IterationPlan {
    pub(crate) output_shape: Vec<usize>,
    pub(crate) numel: usize,
    pub(crate) operands: [OperandIteration; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnaryIterationPlan {
    pub(crate) output_shape: Vec<usize>,
    pub(crate) numel: usize,
    pub(crate) operand: OperandIteration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "cuda")]
pub(crate) enum UnaryLayoutClass {
    Contiguous,
    Strided,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "cuda")]
pub(crate) enum BinaryLayoutClass {
    Contiguous,
    ScalarLeft,
    ScalarRight,
    Strided,
}

#[cfg(feature = "cuda")]
fn is_contiguous(operand: &OperandIteration, output_shape: &[usize]) -> bool {
    let mut expected_stride = 1usize;
    for (&dimension, &stride) in output_shape.iter().zip(&operand.strides).rev() {
        if stride != expected_stride {
            return false;
        }
        let Some(next_stride) = expected_stride.checked_mul(dimension) else {
            return false;
        };
        expected_stride = next_stride;
    }
    true
}

#[cfg(feature = "cuda")]
fn is_scalar_broadcast(operand: &OperandIteration) -> bool {
    operand.strides.iter().all(|&stride| stride == 0)
}

impl UnaryIterationPlan {
    pub(crate) fn new(input: OperandLayout<'_>) -> Result<Self> {
        let mut operand = normalize_operand(input.shape, input.strides, input.offset, input.shape)?;
        let numel = checked_numel(input.shape)?;
        let output_shape = coalesce_dimensions(input.shape, core::slice::from_mut(&mut operand))?;
        Ok(Self {
            output_shape,
            numel,
            operand,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn layout_class(&self) -> UnaryLayoutClass {
        if is_contiguous(&self.operand, &self.output_shape) {
            UnaryLayoutClass::Contiguous
        } else {
            UnaryLayoutClass::Strided
        }
    }
}

impl IterationPlan {
    pub(crate) fn binary(
        lhs: OperandLayout<'_>,
        rhs: OperandLayout<'_>,
        output_shape: &[usize],
    ) -> Result<Self> {
        let lhs = normalize_operand(lhs.shape, lhs.strides, lhs.offset, output_shape)?;
        let rhs = normalize_operand(rhs.shape, rhs.strides, rhs.offset, output_shape)?;
        let numel = checked_numel(output_shape)?;
        let mut operands = [lhs, rhs];
        let output_shape = coalesce_dimensions(output_shape, &mut operands)?;

        Ok(Self {
            output_shape,
            numel,
            operands,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn binary_layout_class(&self) -> BinaryLayoutClass {
        let lhs_contiguous = is_contiguous(&self.operands[0], &self.output_shape);
        let rhs_contiguous = is_contiguous(&self.operands[1], &self.output_shape);
        if lhs_contiguous && rhs_contiguous {
            BinaryLayoutClass::Contiguous
        } else if is_scalar_broadcast(&self.operands[0]) && rhs_contiguous {
            BinaryLayoutClass::ScalarLeft
        } else if lhs_contiguous && is_scalar_broadcast(&self.operands[1]) {
            BinaryLayoutClass::ScalarRight
        } else {
            BinaryLayoutClass::Strided
        }
    }
}

fn coalesce_dimensions(
    output_shape: &[usize],
    operands: &mut [OperandIteration],
) -> Result<Vec<usize>> {
    let mut coalesced_shape = Vec::<usize>::with_capacity(output_shape.len());
    let mut coalesced_strides = vec![Vec::with_capacity(output_shape.len()); operands.len()];

    for (axis, &dimension) in output_shape.iter().enumerate() {
        // A unit dimension never changes a physical index, so its stride is
        // irrelevant and the axis can always be removed.
        if dimension == 1 {
            continue;
        }

        let merge = coalesced_shape.last().is_some_and(|_| {
            operands
                .iter()
                .zip(&coalesced_strides)
                .all(|(operand, strides)| {
                    strides.last().is_some_and(|&outer_stride| {
                        operand.strides[axis]
                            .checked_mul(dimension)
                            .is_some_and(|expected| outer_stride == expected)
                    })
                })
        });

        if merge {
            let outer_dimension = coalesced_shape
                .last_mut()
                .expect("merge requires an existing dimension");
            *outer_dimension = outer_dimension.checked_mul(dimension).ok_or_else(|| {
                Error::Msg(format!(
                    "shape overflow coalescing iteration plan for {output_shape:?}"
                ))
            })?;
            for (operand, strides) in operands.iter().zip(&mut coalesced_strides) {
                *strides
                    .last_mut()
                    .expect("merge requires an existing stride") = operand.strides[axis];
            }
        } else {
            coalesced_shape.push(dimension);
            for (operand, strides) in operands.iter().zip(&mut coalesced_strides) {
                strides.push(operand.strides[axis]);
            }
        }
    }

    for (operand, strides) in operands.iter_mut().zip(coalesced_strides) {
        operand.strides = strides;
    }
    Ok(coalesced_shape)
}

fn normalize_operand(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    output_shape: &[usize],
) -> Result<OperandIteration> {
    if shape.len() != strides.len() || shape.len() > output_shape.len() {
        return Err(iteration_shape_error(output_shape, shape));
    }

    let leading = output_shape.len() - shape.len();
    let mut normalized = vec![0; output_shape.len()];
    for (input_axis, (&input_dim, &stride)) in shape.iter().zip(strides).enumerate() {
        let output_axis = leading + input_axis;
        let output_dim = output_shape[output_axis];
        if input_dim != output_dim && input_dim != 1 {
            return Err(iteration_shape_error(output_shape, shape));
        }
        normalized[output_axis] = if input_dim == 1 && output_dim != 1 {
            0
        } else {
            stride
        };
    }

    Ok(OperandIteration {
        strides: normalized,
        offset,
    })
}

fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |numel, &dimension| {
        numel.checked_mul(dimension).ok_or_else(|| {
            Error::Msg(format!(
                "shape overflow building iteration plan for {shape:?}"
            ))
        })
    })
}

fn iteration_shape_error(output_shape: &[usize], input_shape: &[usize]) -> Error {
    Error::ShapeMismatch {
        op: "iteration_plan",
        expected: output_shape.to_vec(),
        got: input_shape.to_vec(),
        msg: format!(
            "input shape {input_shape:?} cannot be normalized to output shape {output_shape:?}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_operands_keep_their_strides() {
        let plan = IterationPlan::binary(
            OperandLayout {
                shape: &[2, 3],
                strides: &[3, 1],
                offset: 0,
            },
            OperandLayout {
                shape: &[2, 3],
                strides: &[3, 1],
                offset: 4,
            },
            &[2, 3],
        )
        .unwrap();

        assert_eq!(plan.numel, 6);
        assert_eq!(plan.output_shape, vec![6]);
        assert_eq!(plan.operands[0].strides, vec![1]);
        assert_eq!(plan.operands[1].strides, vec![1]);
        assert_eq!(plan.operands[1].offset, 4);
        #[cfg(feature = "cuda")]
        assert_eq!(plan.binary_layout_class(), BinaryLayoutClass::Contiguous);
    }

    #[test]
    fn broadcast_axes_become_zero_stride() {
        let plan = IterationPlan::binary(
            OperandLayout {
                shape: &[3, 1],
                strides: &[1, 1],
                offset: 2,
            },
            OperandLayout {
                shape: &[4],
                strides: &[1],
                offset: 0,
            },
            &[3, 4],
        )
        .unwrap();

        assert_eq!(plan.operands[0].strides, vec![1, 0]);
        assert_eq!(plan.operands[1].strides, vec![0, 1]);
        assert_eq!(plan.operands[0].physical_index(5, &[3, 4]), 3);
        assert_eq!(plan.operands[1].physical_index(5, &[3, 4]), 1);
        #[cfg(feature = "cuda")]
        assert_eq!(plan.binary_layout_class(), BinaryLayoutClass::Strided);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn classifies_whole_operand_scalar_broadcasts_without_confusing_dense_broadcasts() {
        let scalar_left = IterationPlan::binary(
            OperandLayout {
                shape: &[],
                strides: &[],
                offset: 3,
            },
            OperandLayout {
                shape: &[2, 4],
                strides: &[4, 1],
                offset: 7,
            },
            &[2, 4],
        )
        .unwrap();
        assert_eq!(
            scalar_left.binary_layout_class(),
            BinaryLayoutClass::ScalarLeft
        );

        let scalar_right = IterationPlan::binary(
            OperandLayout {
                shape: &[8],
                strides: &[1],
                offset: 2,
            },
            OperandLayout {
                shape: &[1],
                strides: &[1],
                offset: 5,
            },
            &[8],
        )
        .unwrap();
        assert_eq!(
            scalar_right.binary_layout_class(),
            BinaryLayoutClass::ScalarRight
        );
    }

    #[test]
    fn non_contiguous_strides_and_offsets_are_preserved() {
        let plan = IterationPlan::binary(
            OperandLayout {
                shape: &[3, 2],
                strides: &[1, 3],
                offset: 7,
            },
            OperandLayout {
                shape: &[3, 2],
                strides: &[2, 1],
                offset: 0,
            },
            &[3, 2],
        )
        .unwrap();

        assert_eq!(plan.operands[0].strides, vec![1, 3]);
        assert_eq!(plan.operands[0].offset, 7);
    }

    #[test]
    fn coalesces_only_axes_contiguous_for_every_operand() {
        let plan = IterationPlan::binary(
            OperandLayout {
                shape: &[2, 3, 4],
                strides: &[12, 4, 1],
                offset: 0,
            },
            OperandLayout {
                shape: &[1, 3, 4],
                strides: &[12, 4, 1],
                offset: 0,
            },
            &[2, 3, 4],
        )
        .unwrap();

        assert_eq!(plan.output_shape, vec![2, 12]);
        assert_eq!(plan.operands[0].strides, vec![12, 1]);
        assert_eq!(plan.operands[1].strides, vec![0, 1]);
        assert_eq!(plan.operands[0].physical_index(23, &plan.output_shape), 23);
        assert_eq!(plan.operands[1].physical_index(23, &plan.output_shape), 11);
    }

    #[test]
    fn unit_axes_are_removed_regardless_of_stride() {
        let plan = UnaryIterationPlan::new(OperandLayout {
            shape: &[1, 3, 1, 4],
            strides: &[999, 4, 777, 1],
            offset: 5,
        })
        .unwrap();

        assert_eq!(plan.output_shape, vec![12]);
        assert_eq!(plan.operand.strides, vec![1]);
        assert_eq!(plan.operand.physical_index(11, &plan.output_shape), 16);
        #[cfg(feature = "cuda")]
        assert_eq!(plan.layout_class(), UnaryLayoutClass::Contiguous);
    }

    #[test]
    fn incompatible_shapes_are_rejected() {
        assert!(matches!(
            IterationPlan::binary(
                OperandLayout {
                    shape: &[2, 4],
                    strides: &[4, 1],
                    offset: 0,
                },
                OperandLayout {
                    shape: &[3, 4],
                    strides: &[4, 1],
                    offset: 0,
                },
                &[3, 4],
            ),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn unary_plan_preserves_view_metadata_and_checks_bounds() {
        let plan = UnaryIterationPlan::new(OperandLayout {
            shape: &[3, 2],
            strides: &[1, 3],
            offset: 4,
        })
        .unwrap();

        assert_eq!(plan.numel, 6);
        assert_eq!(plan.operand.physical_index(5, &plan.output_shape), 9);
        assert_eq!(
            plan.operand.max_physical_index(&plan.output_shape).unwrap(),
            Some(9)
        );
    }

    #[test]
    fn empty_iteration_has_no_physical_index() {
        let plan = UnaryIterationPlan::new(OperandLayout {
            shape: &[0, 3],
            strides: &[3, 1],
            offset: usize::MAX,
        })
        .unwrap();

        assert_eq!(plan.numel, 0);
        assert_eq!(
            plan.operand.max_physical_index(&plan.output_shape).unwrap(),
            None
        );
    }
}
