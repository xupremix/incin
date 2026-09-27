//! Learning Rate Schedulers
//!
//! Provides strategies for adjusting the learning rate during training.

#[cfg(feature = "std")]
use core::f64::consts::PI;

/// Trait for learning rate schedulers.
pub trait LRScheduler {
    /// Returns the current learning rate.
    fn get_lr(&self) -> f64;
    /// Advances the scheduler by one step.
    fn step(&mut self);
}

/// Constant learning rate scheduler.
pub struct ConstantLR {
    lr: f64,
}

impl ConstantLR {
    /// Starts from one constant learning rate.
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self) -> f64 {
        self.lr
    }

    fn step(&mut self) {}
}

/// Linear learning rate decay.
pub struct LinearLR {
    initial_lr: f64,
    final_lr: f64,
    total_steps: usize,
    current_step: usize,
}

impl LinearLR {
    /// Linearly decays to final_lr over total_steps.
    pub fn new(initial_lr: f64, final_lr: f64, total_steps: usize) -> Self {
        Self {
            initial_lr,
            final_lr,
            total_steps,
            current_step: 0,
        }
    }
}

impl LRScheduler for LinearLR {
    fn get_lr(&self) -> f64 {
        if self.current_step >= self.total_steps {
            return self.final_lr;
        }
        let progress = self.current_step as f64 / self.total_steps as f64;
        self.initial_lr + (self.final_lr - self.initial_lr) * progress
    }

    fn step(&mut self) {
        self.current_step = self.current_step.saturating_add(1);
    }
}

/// Cosine annealing learning rate scheduler.
#[cfg(feature = "std")]
pub struct CosineAnnealingLR {
    initial_lr: f64,
    min_lr: f64,
    t_max: usize,
    current_step: usize,
}

#[cfg(feature = "std")]
impl CosineAnnealingLR {
    /// Cosine-anneals to min_lr over t_max steps.
    pub fn new(initial_lr: f64, min_lr: f64, t_max: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_max,
            current_step: 0,
        }
    }
}

#[cfg(feature = "std")]
impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self) -> f64 {
        let progress = (self.current_step as f64) / (self.t_max as f64);
        let progress = progress.min(1.0);

        self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + f64::cos(progress * PI))
    }

    fn step(&mut self) {
        self.current_step = self.current_step.saturating_add(1);
    }
}

/// Step learning rate decay.
#[cfg(feature = "std")]
pub struct StepLR {
    initial_lr: f64,
    step_size: usize,
    gamma: f64,
    current_step: usize,
}

#[cfg(feature = "std")]
impl StepLR {
    /// Steps gamma decay every step_size epochs.
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
}

#[cfg(feature = "std")]
impl LRScheduler for StepLR {
    fn get_lr(&self) -> f64 {
        let num_steps = (self.current_step / self.step_size) as i32;
        self.initial_lr * self.gamma.powi(num_steps)
    }

    fn step(&mut self) {
        self.current_step = self.current_step.saturating_add(1);
    }
}

/// Linear warmup from zero to a base learning rate.
///
/// Returns `base_lr * current_step / warmup_steps` while warming up and
/// `base_lr` once `current_step >= warmup_steps`, so the very first rate
/// (before any `step` call) is exactly `0.0`. A `warmup_steps` of zero skips
/// the ramp and returns `base_lr` immediately, which keeps the constructor
/// infallible like every other scheduler here.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// use incin_core::optim::{LRScheduler, LinearWarmup};
///
/// let mut scheduler = LinearWarmup::new(0.1, 4);
/// assert_eq!(scheduler.get_lr(), 0.0);
/// scheduler.step();
/// scheduler.step();
/// let lr = scheduler.get_lr();
/// assert!((lr - 0.05).abs() < 1e-12, "unexpected lr {lr}");
/// ```
pub struct LinearWarmup {
    base_lr: f64,
    warmup_steps: usize,
    current_step: usize,
}

impl LinearWarmup {
    /// Ramps linearly from `0.0` to `base_lr` over `warmup_steps` steps.
    pub fn new(base_lr: f64, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            current_step: 0,
        }
    }
}

impl LRScheduler for LinearWarmup {
    fn get_lr(&self) -> f64 {
        if self.warmup_steps == 0 || self.current_step >= self.warmup_steps {
            return self.base_lr;
        }
        self.base_lr * (self.current_step as f64) / (self.warmup_steps as f64)
    }

    fn step(&mut self) {
        self.current_step = self.current_step.saturating_add(1);
    }
}

/// Exponential learning rate decay.
///
/// Returns `initial_lr * gamma.powi(current_step)`: with the usual
/// `gamma < 1.0` the rate shrinks geometrically every step, with
/// `gamma == 1.0` this is a constant schedule, and `gamma > 1.0` grows it
/// (valid but rarely what a training loop wants).
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// use incin_core::optim::{ExponentialLR, LRScheduler};
///
/// let mut scheduler = ExponentialLR::new(0.1, 0.9);
/// scheduler.step();
/// scheduler.step();
/// let lr = scheduler.get_lr();
/// assert!((lr - 0.081).abs() < 1e-12, "unexpected lr {lr}");
/// ```
pub struct ExponentialLR {
    initial_lr: f64,
    gamma: f64,
    current_step: usize,
}

impl ExponentialLR {
    /// Decays `initial_lr` by `gamma` every step.
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self {
            initial_lr,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self) -> f64 {
        let step = i32::try_from(self.current_step).unwrap_or(i32::MAX);
        self.initial_lr * self.gamma.powi(step)
    }

    fn step(&mut self) {
        self.current_step = self.current_step.saturating_add(1);
    }
}

/// Linear warmup followed by cosine annealing.
///
/// The first `warmup_steps` rates ramp linearly from `0.0` to `base_lr`
/// (exactly like [`LinearWarmup`]); steps `warmup_steps..=total_steps` then
/// cosine-anneal from `base_lr` down to `min_lr`, and anything past
/// `total_steps` holds `min_lr`. When `total_steps <= warmup_steps` there is
/// no cosine phase and every step at or past warmup returns `min_lr`.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// use incin_core::optim::{CosineWithWarmup, LRScheduler};
///
/// let mut scheduler = CosineWithWarmup::new(0.1, 0.0, 2, 6);
/// assert_eq!(scheduler.get_lr(), 0.0);
/// scheduler.step();
/// scheduler.step();
/// let lr = scheduler.get_lr();
/// assert!((lr - 0.1).abs() < 1e-12, "unexpected lr {lr}");
/// ```
#[cfg(feature = "std")]
pub struct CosineWithWarmup {
    base_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

#[cfg(feature = "std")]
impl CosineWithWarmup {
    /// Warms up from `0.0` to `base_lr` over `warmup_steps`, then
    /// cosine-anneals to `min_lr` at `total_steps`.
    pub fn new(base_lr: f64, min_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }
}

#[cfg(feature = "std")]
impl LRScheduler for CosineWithWarmup {
    fn get_lr(&self) -> f64 {
        if self.warmup_steps > 0 && self.current_step < self.warmup_steps {
            return self.base_lr * (self.current_step as f64) / (self.warmup_steps as f64);
        }
        if self.total_steps <= self.warmup_steps {
            return self.min_lr;
        }
        let elapsed = self.current_step.saturating_sub(self.warmup_steps) as f64;
        let span = (self.total_steps - self.warmup_steps) as f64;
        let progress = (elapsed / span).min(1.0);
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + f64::cos(progress * PI))
    }

    fn step(&mut self) {
        self.current_step = self.current_step.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_lr(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
            "lr {actual} vs expected {expected}"
        );
    }

    #[test]
    fn linear_warmup_ramps_exactly_then_holds() {
        let mut scheduler = LinearWarmup::new(1.0, 4);
        for expected in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_lr(scheduler.get_lr(), expected);
            scheduler.step();
        }
        // Past warmup the base rate holds, however many steps follow.
        scheduler.step();
        scheduler.step();
        assert_lr(scheduler.get_lr(), 1.0);
    }

    #[test]
    fn linear_warmup_without_steps_is_constant() {
        let mut scheduler = LinearWarmup::new(0.5, 0);
        assert_lr(scheduler.get_lr(), 0.5);
        scheduler.step();
        assert_lr(scheduler.get_lr(), 0.5);
    }

    #[test]
    fn exponential_decay_follows_the_geometric_sequence() {
        let mut scheduler = ExponentialLR::new(2.0, 0.5);
        for expected in [2.0, 1.0, 0.5, 0.25, 0.125] {
            assert_lr(scheduler.get_lr(), expected);
            scheduler.step();
        }
    }

    #[test]
    fn exponential_decay_with_unit_gamma_is_constant() {
        let mut scheduler = ExponentialLR::new(0.3, 1.0);
        for _ in 0..5 {
            assert_lr(scheduler.get_lr(), 0.3);
            scheduler.step();
        }
    }

    #[test]
    fn exponential_decay_matches_powi_on_a_typical_gamma() {
        let mut scheduler = ExponentialLR::new(1.0, 0.9);
        for step in 0..6 {
            assert_lr(scheduler.get_lr(), 0.9f64.powi(step));
            scheduler.step();
        }
        // 0.9^5 spot value, anchoring the sequence to a literal.
        let mut scheduler = ExponentialLR::new(1.0, 0.9);
        for _ in 0..5 {
            scheduler.step();
        }
        assert_lr(scheduler.get_lr(), 0.59049);
    }

    #[cfg(feature = "std")]
    #[test]
    fn cosine_with_warmup_ramps_then_anneals_to_the_floor() {
        let mut scheduler = CosineWithWarmup::new(1.0, 0.0, 2, 6);
        // Warmup: exact ramp from zero.
        assert_lr(scheduler.get_lr(), 0.0);
        scheduler.step();
        assert_lr(scheduler.get_lr(), 0.5);
        scheduler.step();
        // Cosine phase opens exactly at the base rate (cos 0 == 1).
        assert_lr(scheduler.get_lr(), 1.0);
        scheduler.step();
        // Progress 1/4: 0.5 * (1 + cos(pi/4)).
        assert_lr(scheduler.get_lr(), 0.8535533905932737);
        scheduler.step();
        // Progress 1/2: 0.5 * (1 + cos(pi/2)) == 0.5 up to fp trig error.
        assert_lr(scheduler.get_lr(), 0.5);
        scheduler.step();
        scheduler.step();
        // Total reached: exactly the floor (cos pi == -1).
        assert_lr(scheduler.get_lr(), 0.0);
        scheduler.step();
        scheduler.step();
        assert_lr(scheduler.get_lr(), 0.0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn cosine_with_warmup_without_a_cosine_phase_holds_the_floor() {
        let mut scheduler = CosineWithWarmup::new(1.0, 0.1, 4, 4);
        for expected in [0.0, 0.25, 0.5, 0.75] {
            assert_lr(scheduler.get_lr(), expected);
            scheduler.step();
        }
        assert_lr(scheduler.get_lr(), 0.1);
        scheduler.step();
        assert_lr(scheduler.get_lr(), 0.1);
    }

    #[cfg(feature = "std")]
    #[test]
    fn cosine_with_warmup_without_warmup_starts_at_the_base_rate() {
        let mut scheduler = CosineWithWarmup::new(1.0, 0.0, 0, 4);
        assert_lr(scheduler.get_lr(), 1.0);
        for _ in 0..4 {
            scheduler.step();
        }
        assert_lr(scheduler.get_lr(), 0.0);
    }
}
