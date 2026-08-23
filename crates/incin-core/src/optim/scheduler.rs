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
