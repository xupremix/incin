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

/// Running mean accumulator (e.g. epoch loss): `sum / count`.
///
/// Unlike [`MSE`], which fixes the accumulation to squared errors, this
/// observes caller-supplied scalars one at a time, so the same type tracks a
/// training loss, a validation loss, or any other per-batch scalar the loop
/// reports. The trainer's validation loop accumulates its per-batch losses
/// here and monitors the mean.
#[derive(Debug, Clone, Default)]
pub struct Mean {
    sum: f64,
    count: usize,
}

impl Mean {
    /// Creates a new empty `Mean` accumulator.
    pub fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }

    /// Observes one scalar value (e.g. a batch loss).
    pub fn update(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
    }

    /// Observes a slice of scalar values (e.g. per-sample losses).
    pub fn update_slice(&mut self, values: &[f32]) {
        for &value in values {
            self.update(f64::from(value));
        }
    }

    /// How many values have been observed.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Metric for Mean {
    fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }

    fn value(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

/// Top-k classification accuracy with configurable `k`: a sample counts as
/// correct when its target class is among the `k` highest-scored classes.
///
/// At `k == 1` this agrees with [`Accuracy`]; larger `k` credits a model
/// that ranks the right answer highly without putting it first.
#[derive(Debug, Clone)]
pub struct TopKAccuracy {
    k: usize,
    correct: usize,
    total: usize,
}

impl TopKAccuracy {
    /// Creates a new empty `TopKAccuracy` for the given `k`.
    ///
    /// A `k` of zero behaves as `k == 1`: an empty top-k set could never
    /// contain the target, so it would report zero forever rather than
    /// measure anything.
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            correct: 0,
            total: 0,
        }
    }

    /// The configured `k` (at least one).
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Updates the metric with flat row-major scores and one target per sample.
    ///
    /// `scores` holds `num_classes` consecutive scores per sample
    /// (`scores[sample * num_classes + class]`). Only whole samples are
    /// observed - a trailing partial row is ignored, matching the truncation
    /// convention of the other host-slice metrics - and a zero
    /// `num_classes` observes nothing. Ties break toward the lower class
    /// index, so the result is deterministic for equal scores.
    pub fn update(&mut self, scores: &[f32], targets: &[usize], num_classes: usize) {
        if num_classes == 0 {
            return;
        }
        let samples = targets.len().min(scores.len() / num_classes);
        for sample in 0..samples {
            let row = &scores[sample * num_classes..(sample + 1) * num_classes];
            // Class indices of the top-k scores, best first. A linear
            // insertion keeps the order deterministic on ties; rows are
            // class-count wide, so the quadratic scan stays proportional
            // to work the caller already did to produce the scores.
            let mut top: Vec<usize> = Vec::new();
            for (class, &score) in row.iter().enumerate() {
                let mut position = top.len();
                for (index, &placed) in top.iter().enumerate() {
                    if score > row[placed] || (score == row[placed] && class < placed) {
                        position = index;
                        break;
                    }
                }
                if position < self.k {
                    top.insert(position, class);
                    top.truncate(self.k);
                }
            }
            if top.contains(&targets[sample]) {
                self.correct += 1;
            }
        }
        self.total += samples;
    }
}

impl Metric for TopKAccuracy {
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

    /// Updates the matrix with target and prediction indices (`matrix[target][pred]`).
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

    #[test]
    fn test_mean() {
        let mut mean = Mean::new();
        assert_eq!(mean.value(), 0.0);
        assert_eq!(mean.count(), 0);
        mean.update(1.0);
        mean.update(2.0);
        mean.update_slice(&[3.0, 4.0]);
        assert_eq!(mean.count(), 4);
        assert!((mean.value() - 2.5).abs() < 1e-12);
        mean.reset();
        assert_eq!(mean.value(), 0.0);
        assert_eq!(mean.count(), 0);
    }

    #[test]
    fn test_top_k_accuracy() {
        // Two samples over three classes.
        // Sample 0: scores [0.1, 0.8, 0.1], target 0. Top-1 is class 1
        // (wrong); top-2 is {1, 0} (right).
        // Sample 1: scores [0.7, 0.2, 0.1], target 2. Top-1 is class 0
        // (wrong); top-2 is {0, 1} (still wrong).
        let scores = [0.1, 0.8, 0.1, 0.7, 0.2, 0.1];
        let targets = [0, 2];

        let mut top1 = TopKAccuracy::new(1);
        top1.update(&scores, &targets, 3);
        assert_eq!(top1.k(), 1);
        assert_eq!(top1.value(), 0.0);

        let mut top2 = TopKAccuracy::new(2);
        top2.update(&scores, &targets, 3);
        assert_eq!(top2.value(), 0.5);
        top2.reset();
        assert_eq!(top2.value(), 0.0);
    }

    #[test]
    fn top_k_at_one_agrees_with_accuracy_on_argmax() {
        // Every sample's argmax is its target: both metrics report 1.0.
        let scores = [0.9, 0.1, 0.2, 0.8];
        let targets = [0, 1];
        let mut top1 = TopKAccuracy::new(1);
        top1.update(&scores, &targets, 2);
        assert_eq!(top1.value(), 1.0);

        let mut accuracy = Accuracy::new();
        accuracy.update(&[0, 1], &targets);
        assert_eq!(accuracy.value(), top1.value());
    }

    #[test]
    fn top_k_ignores_partial_rows_and_zero_classes() {
        let mut metric = TopKAccuracy::new(2);
        // Five scores cannot form two whole 3-class rows: one sample observed.
        metric.update(&[0.1, 0.8, 0.1, 0.7, 0.2], &[0, 2], 3);
        assert_eq!(metric.value(), 1.0);

        metric.reset();
        metric.update(&[0.5], &[0], 0);
        assert_eq!(metric.value(), 0.0);

        // A zero k behaves as top-1 rather than an always-empty set.
        let mut zero_k = TopKAccuracy::new(0);
        assert_eq!(zero_k.k(), 1);
        zero_k.update(&[0.9, 0.1], &[0], 2);
        assert_eq!(zero_k.value(), 1.0);
    }
}
