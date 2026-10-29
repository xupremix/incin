//! The CPU backend's answer to the core gradient-check contract.
//!
//! Separate from `cpu::gradcheck`, which is `#[cfg(test)]` and holds this
//! crate's own sweep. This module is not test-gated, and cannot be: the
//! public `incin_core::exec::gradcheck` is useless to a custom-operation
//! author without an implementation of its storage trait, and an
//! implementation that exists only in this crate's unit tests is not one.

use incin_core::error::{Error, Result};
use incin_core::exec::{GradCheckStorage, GradientMap};

use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::{stride, tape};

/// Three reads and a walk.
///
/// `backward_from` is this backend's own thread-local walk, the one
/// `Tensor::backward` reaches, so a check written against the core trait
/// exercises the real path rather than a reconstruction of it.
impl GradCheckStorage for CpuStorage {
    fn backward_from(loss: &Self) -> Result<GradientMap<Self>> {
        tape::backward(loss).map(|grads| grads.grads)
    }

    fn element_count(&self) -> usize {
        stride::validated_numel(&self.shape).max(1)
    }

    fn element(&self, index: usize) -> Result<f64> {
        Ok(self.get(&self.multi_index(index)))
    }

    fn with_element_perturbed(&self, index: usize, delta: f64) -> Result<Self> {
        let multi = self.multi_index(index);
        let mut flat = self.offset_elements;
        for (i, s) in multi.iter().zip(self.strides.iter()) {
            flat += i * s;
        }

        // A fresh contiguous buffer, never a mutation. The original is still
        // on the graph and a recipe may have saved it, so perturbing in place
        // would move the value the backward rule is about to read.
        let buffer = match &*self.buffer {
            CpuBuffer::F32(values) => {
                let mut values = values.clone();
                values[flat] = (f64::from(values[flat]) + delta) as f32;
                CpuBuffer::F32(values)
            }
            CpuBuffer::F64(values) => {
                let mut values = values.clone();
                values[flat] += delta;
                CpuBuffer::F64(values)
            }
            // An error rather than a panic. Checking the gradient of an
            // integer or block-quantized operand is a mistake in the caller's
            // test, and a mistake in a test should come back as a message
            // rather than an aborted process.
            _ => {
                return Err(Error::UnsupportedBackendOperation {
                    op: "gradcheck perturbation (only f32 and f64 operands)",
                    backend: "Cpu",
                });
            }
        };
        Ok(Self::from_contiguous(buffer, &self.shape))
    }
}

impl CpuStorage {
    /// Resolve a logical row-major position into a multi-index.
    ///
    /// Logical rather than physical, so a non-contiguous view is swept in the
    /// order its shape describes rather than the order its buffer happens to
    /// hold.
    fn multi_index(&self, index: usize) -> alloc::vec::Vec<usize> {
        let mut multi = alloc::vec![0usize; self.shape.len()];
        for _ in 0..index {
            crate::cpu::ops::elementwise::increment_index(&mut multi, &self.shape);
        }
        multi
    }
}
