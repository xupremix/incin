//! Backend-neutral normalization of tensor iteration metadata.
//!
//! Operation implementations consume normalized, output-rank strides. A
//! broadcast dimension has stride zero, so kernels do not need separate
//! input-shape branches in their inner loops.
//!
//! Nothing here allocates for a shape the typed frontend can express. A plan
//! is built once per operation and read once per element, so its own cost is
//! pure overhead against the kernel it precedes, and a `[64, 8] + [64, 1]` add
//! used to spend six heap allocations describing two operands of rank two.
//! The dimension and stride lists are [`ShapeBuf`] and [`StrideBuf`], which
//! `SHP-003` already made inline up to `INLINE_RANK`, and the per-operand working
//! storage in `coalesce_dimensions` is a fixed-size array rather than a vector
//! of vectors, because an iteration plan has exactly one or two operands and
//! the count is known to the type system.

use crate::bytes::checked_numel;
use incin_core::error::{Error, Result};
#[cfg(feature = "cuda")]
use incin_core::exec::LayoutClass;
use incin_core::shapes::{ShapeBuf, StrideBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OperandIteration {
    pub(crate) strides: StrideBuf,
    pub(crate) offset: usize,
}

impl OperandIteration {
    #[inline]
    #[cfg(any(feature = "cpu", test))]
    pub(crate) fn physical_index(&self, mut flat_index: usize, output_shape: &[usize]) -> usize {
        let strides = self.strides.strides();
        debug_assert_eq!(strides.len(), output_shape.len());
        let mut physical = self.offset;
        for axis in (0..output_shape.len()).rev() {
            let dimension = output_shape[axis];
            if dimension != 0 {
                let coordinate = flat_index % dimension;
                flat_index /= dimension;
                physical += coordinate * strides[axis];
            }
        }
        physical
    }

    pub(crate) fn max_physical_index(&self, output_shape: &[usize]) -> Result<Option<usize>> {
        if self.strides.len() != output_shape.len() {
            return Err(iteration_shape_error(output_shape, self.strides.strides()));
        }
        if output_shape.contains(&0) {
            return Ok(None);
        }
        let max_offset = output_shape.iter().zip(self.strides.strides()).try_fold(
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
                                self.strides.strides()
                            ))
                        })?;
                offset.checked_add(axis_offset).ok_or_else(|| {
                    Error::Msg(format!(
                        "offset overflow building iteration bounds for shape \
                             {output_shape:?} and strides {:?}",
                        self.strides.strides()
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
    pub(crate) output_shape: ShapeBuf,
    pub(crate) numel: usize,
    pub(crate) operands: [OperandIteration; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnaryIterationPlan {
    pub(crate) output_shape: ShapeBuf,
    pub(crate) numel: usize,
    pub(crate) operand: OperandIteration,
}

#[cfg(feature = "cuda")]
fn is_contiguous(operand: &OperandIteration, output_shape: &[usize]) -> bool {
    let mut expected_stride = 1usize;
    for (&dimension, &stride) in output_shape.iter().zip(operand.strides.strides()).rev() {
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
    operand.strides.strides().iter().all(|&stride| stride == 0)
}

impl UnaryIterationPlan {
    pub(crate) fn new(input: OperandLayout<'_>) -> Result<Self> {
        let mut operands = [normalize_operand(
            input.shape,
            input.strides,
            input.offset,
            input.shape,
        )?];
        let numel = checked_numel(input.shape)?;
        let output_shape = coalesce_dimensions(input.shape, &mut operands)?;
        let [operand] = operands;
        Ok(Self {
            output_shape,
            numel,
            operand,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn layout_class(&self) -> LayoutClass {
        if is_contiguous(&self.operand, self.output_shape.dims()) {
            LayoutClass::Contiguous
        } else {
            LayoutClass::Strided
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
    pub(crate) fn binary_layout_class(&self) -> LayoutClass {
        let lhs_contiguous = is_contiguous(&self.operands[0], self.output_shape.dims());
        let rhs_contiguous = is_contiguous(&self.operands[1], self.output_shape.dims());
        if lhs_contiguous && rhs_contiguous {
            LayoutClass::Contiguous
        } else if is_scalar_broadcast(&self.operands[0]) && rhs_contiguous {
            LayoutClass::ScalarLeft
        } else if lhs_contiguous && is_scalar_broadcast(&self.operands[1]) {
            LayoutClass::ScalarRight
        } else {
            LayoutClass::Strided
        }
    }
}

/// Merge axes that are contiguous for *every* operand, and drop unit axes.
///
/// `N` is the operand count, one or two, and it is a const parameter so the
/// per-operand working strides are a fixed-size array. A vector of vectors here
/// cost one allocation for the outer list plus one per operand, on a structure
/// whose length the caller always knew.
fn coalesce_dimensions<const N: usize>(
    output_shape: &[usize],
    operands: &mut [OperandIteration; N],
) -> Result<ShapeBuf> {
    let mut coalesced_shape = ShapeBuf::SCALAR;
    let mut coalesced_strides = [const { StrideBuf::EMPTY }; N];

    for (axis, &dimension) in output_shape.iter().enumerate() {
        // A unit dimension never changes a physical index, so its stride is
        // irrelevant and the axis can always be removed.
        if dimension == 1 {
            continue;
        }

        let merge = coalesced_shape.rank() != 0
            && operands
                .iter()
                .zip(&coalesced_strides)
                .all(|(operand, strides)| {
                    strides.strides().last().is_some_and(|&outer_stride| {
                        operand.strides.strides()[axis]
                            .checked_mul(dimension)
                            .is_some_and(|expected| outer_stride == expected)
                    })
                });

        if merge {
            let outer = coalesced_shape
                .dims_mut()
                .last_mut()
                .expect("merge requires an existing dimension");
            *outer = outer.checked_mul(dimension).ok_or_else(|| {
                Error::Msg(format!(
                    "shape overflow coalescing iteration plan for {output_shape:?}"
                ))
            })?;
            for (operand, strides) in operands.iter().zip(&mut coalesced_strides) {
                strides.pop();
                strides.push(operand.strides.strides()[axis]);
            }
        } else {
            coalesced_shape.push(dimension);
            for (operand, strides) in operands.iter().zip(&mut coalesced_strides) {
                strides.push(operand.strides.strides()[axis]);
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

    // Built by pushing rather than by indexing into a zero-filled buffer: the
    // leading axes an operand does not have are exactly the broadcast ones, and
    // their stride is the zero this pushes.
    let leading = output_shape.len() - shape.len();
    let mut normalized = StrideBuf::EMPTY;
    for _ in 0..leading {
        normalized.push(0);
    }
    for (input_axis, (&input_dim, &stride)) in shape.iter().zip(strides).enumerate() {
        let output_dim = output_shape[leading + input_axis];
        if input_dim != output_dim && input_dim != 1 {
            return Err(iteration_shape_error(output_shape, shape));
        }
        normalized.push(if input_dim == 1 && output_dim != 1 {
            0
        } else {
            stride
        });
    }

    Ok(OperandIteration {
        strides: normalized,
        offset,
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

/// Decomposes a 2D iteration space `(rows × cols)` into 2D tiles of size `(TY × TX)`.
///
/// Calls `f(r0, r1, c0, c1)` for each tile, where `[r0..r1)` is the row range
/// (step size `TY`) and `[c0..c1)` is the column range (step size `TX`).
pub fn tile_2d<const TX: usize, const TY: usize>(
    rows: usize,
    cols: usize,
    mut f: impl FnMut(usize, usize, usize, usize),
) {
    if rows == 0 || cols == 0 {
        return;
    }
    let mut r0 = 0;
    while r0 < rows {
        let r1 = (r0 + TY).min(rows);
        let mut c0 = 0;
        while c0 < cols {
            let c1 = (c0 + TX).min(cols);
            f(r0, r1, c0, c1);
            c0 += TX;
        }
        r0 += TY;
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
        assert_eq!(plan.binary_layout_class(), LayoutClass::Contiguous);
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
        assert_eq!(plan.binary_layout_class(), LayoutClass::Strided);
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
        assert_eq!(scalar_left.binary_layout_class(), LayoutClass::ScalarLeft);

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
        assert_eq!(scalar_right.binary_layout_class(), LayoutClass::ScalarRight);
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
        assert_eq!(plan.layout_class(), LayoutClass::Contiguous);
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
