//! Evaluation metrics module.
//!
//! Provides traits and implementations for tracking model evaluation metrics
//! during training and validation loops.

use alloc::vec;
use alloc::vec::Vec;

/// Core trait for evaluation metrics.
pub trait Metric: Send + Sync {
    /// Resets the metric counter.
    fn reset(&mut self);
    /// Returns the current computed scalar metric value.
    fn value(&self) -> f64;
}

/// Classification accuracy metric (fraction of correct predictions).
#[derive(Debug, Clone, Default)]
pub struct Accuracy {
    correct: usize,
    total: usize,
}

impl Accuracy {
    /// Creates a new empty `Accuracy` metric.
    pub fn new() -> Self {
        Self {
            correct: 0,
            total: 0,
        }
    }

    /// Updates the metric with prediction and target class index slices.
    pub fn update(&mut self, preds: &[usize], targets: &[usize]) {
        let count = preds.len().min(targets.len());
        for i in 0..count {
            if preds[i] == targets[i] {
                self.correct += 1;
            }
        }
        self.total += count;
    }
}

impl Metric for Accuracy {
    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn value(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }
}

/// Binary precision metric: `TP / (TP + FP)`.
#[derive(Debug, Clone, Default)]
pub struct Precision {
    tp: usize,
    fp: usize,
    positive_class: usize,
}

impl Precision {
    /// Creates a new `Precision` metric for target positive class (default 1).
    pub fn new(positive_class: usize) -> Self {
        Self {
            tp: 0,
            fp: 0,
            positive_class,
        }
    }

    /// Updates the metric with predictions and targets.
    pub fn update(&mut self, preds: &[usize], targets: &[usize]) {
        let count = preds.len().min(targets.len());
        for i in 0..count {
            if preds[i] == self.positive_class {
                if targets[i] == self.positive_class {
                    self.tp += 1;
                } else {
                    self.fp += 1;
                }
            }
        }
    }
}

impl Metric for Precision {
    fn reset(&mut self) {
        self.tp = 0;
        self.fp = 0;
    }

    fn value(&self) -> f64 {
        if self.tp + self.fp == 0 {
            0.0
        } else {
            self.tp as f64 / (self.tp + self.fp) as f64
        }
    }
}

/// Binary recall metric: `TP / (TP + FN)`.
#[derive(Debug, Clone, Default)]
pub struct Recall {
    tp: usize,
    fn_count: usize,
    positive_class: usize,
}

impl Recall {
    /// Creates a new `Recall` metric for target positive class (default 1).
    pub fn new(positive_class: usize) -> Self {
        Self {
            tp: 0,
            fn_count: 0,
            positive_class,
        }
    }

    /// Updates the metric with predictions and targets.
    pub fn update(&mut self, preds: &[usize], targets: &[usize]) {
        let count = preds.len().min(targets.len());
        for i in 0..count {
            if targets[i] == self.positive_class {
                if preds[i] == self.positive_class {
                    self.tp += 1;
                } else {
                    self.fn_count += 1;
                }
            }
        }
    }
}

impl Metric for Recall {
    fn reset(&mut self) {
        self.tp = 0;
        self.fn_count = 0;
    }

    fn value(&self) -> f64 {
        if self.tp + self.fn_count == 0 {
            0.0
        } else {
            self.tp as f64 / (self.tp + self.fn_count) as f64
        }
    }
}

/// Binary F1-score metric: `2 * P * R / (P + R)`.
#[derive(Debug, Clone, Default)]
pub struct F1Score {
    precision: Precision,
    recall: Recall,
}

impl F1Score {
    /// Creates a new `F1Score` metric for positive class.
    pub fn new(positive_class: usize) -> Self {
        Self {
            precision: Precision::new(positive_class),
            recall: Recall::new(positive_class),
        }
    }

    /// Updates the metric with predictions and targets.
    pub fn update(&mut self, preds: &[usize], targets: &[usize]) {
        self.precision.update(preds, targets);
        self.recall.update(preds, targets);
    }
}

impl Metric for F1Score {
    fn reset(&mut self) {
        self.precision.reset();
        self.recall.reset();
    }

    fn value(&self) -> f64 {
        let p = self.precision.value();
        let r = self.recall.value();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

/// Mean Squared Error (MSE) regression metric.
#[derive(Debug, Clone, Default)]
pub struct MSE {
    sum_sq_err: f64,
    count: usize,
}

impl MSE {
    /// Creates a new empty `MSE` metric.
    pub fn new() -> Self {
        Self {
            sum_sq_err: 0.0,
            count: 0,
        }
    }

    /// Updates the metric with predicted and target float slices.
    pub fn update(&mut self, preds: &[f32], targets: &[f32]) {
        let len = preds.len().min(targets.len());
        for i in 0..len {
            let diff = (preds[i] - targets[i]) as f64;
            self.sum_sq_err += diff * diff;
        }
        self.count += len;
    }
}

impl Metric for MSE {
    fn reset(&mut self) {
        self.sum_sq_err = 0.0;
        self.count = 0;
    }

    fn value(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_sq_err / self.count as f64
        }
    }
}

/// Confusion matrix tracking per-class counts.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    num_classes: usize,
    matrix: Vec<Vec<usize>>,
}

impl ConfusionMatrix {
    /// Creates a new `ConfusionMatrix` for `num_classes`.
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            matrix: vec![vec![0; num_classes]; num_classes],
        }
    }

    /// Updates the matrix with target and prediction indices (matrix[target][pred]).
    pub fn update(&mut self, preds: &[usize], targets: &[usize]) {
        let count = preds.len().min(targets.len());
        for i in 0..count {
            let t = targets[i];
            let p = preds[i];
            if t < self.num_classes && p < self.num_classes {
                self.matrix[t][p] += 1;
            }
        }
    }

    /// Returns a slice of rows representing the matrix (`matrix[target][pred]`).
    pub fn matrix(&self) -> &[Vec<usize>] {
        &self.matrix
    }

    /// Resets all counts to zero.
    pub fn reset(&mut self) {
        for row in &mut self.matrix {
            for val in row {
                *val = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        let mut acc = Accuracy::new();
        acc.update(&[0, 1, 2, 3], &[0, 1, 2, 0]);
        assert_eq!(acc.value(), 0.75); // 3 out of 4 correct
        acc.reset();
        assert_eq!(acc.value(), 0.0);
    }

    #[test]
    fn test_precision_recall_f1() {
        let mut f1 = F1Score::new(1); // class 1
        // preds:  [1, 1, 0, 0]
        // targets:[1, 0, 1, 0]
        // TP=1, FP=1, FN=1, TN=1 -> P=0.5, R=0.5, F1=0.5
        f1.update(&[1, 1, 0, 0], &[1, 0, 1, 0]);
        assert_eq!(f1.precision.value(), 0.5);
        assert_eq!(f1.recall.value(), 0.5);
        assert_eq!(f1.value(), 0.5);
    }

    #[test]
    fn test_mse() {
        let mut mse = MSE::new();
        mse.update(&[1.0, 2.0, 3.0], &[1.0, 4.0, 3.0]);
        // diffs: 0, -2, 0 -> sq diffs: 0, 4, 0 -> avg = 4/3
        assert!((mse.value() - (4.0 / 3.0)).abs() < 1e-6);
    }
}
