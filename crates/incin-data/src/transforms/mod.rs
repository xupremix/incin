//! Data transformation and augmentation pipeline.
//!
//! Provides traits and implementations for data preprocessing, batch normalization,
//! image transformations, and pipeline composition.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use anyhow::{Result, bail};
use rand::Rng;

/// Core trait for a data transformation step.
pub trait Transform: Send + Sync {
    /// Input data type.
    type Input;
    /// Output data type.
    type Output;

    /// Applies the transformation to `input`.
    fn transform(&self, input: Self::Input) -> Result<Self::Output>;
}

/// Pipeline composing multiple sequential transformations on the same data type.
pub struct Compose<T> {
    transforms: Vec<Box<dyn Transform<Input = T, Output = T>>>,
}

impl<T> Compose<T> {
    /// Creates a new empty transform pipeline.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Appends a transform step to the pipeline.
    pub fn push<TR>(mut self, transform: TR) -> Self
    where
        TR: Transform<Input = T, Output = T> + 'static,
    {
        self.transforms.push(Box::new(transform));
        self
    }
}

impl<T> Default for Compose<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform for Compose<T> {
    type Input = T;
    type Output = T;

    fn transform(&self, mut input: Self::Input) -> Result<Self::Output> {
        for t in &self.transforms {
            input = t.transform(input)?;
        }
        Ok(input)
    }
}

/// Normalizes floating point slice values across channels: `out[c] = (in[c] - mean[c]) / std[c]`.
#[derive(Debug, Clone)]
pub struct Normalize {
    /// Mean value per channel.
    pub mean: Vec<f32>,
    /// Standard deviation value per channel.
    pub std: Vec<f32>,
}

impl Normalize {
    /// Creates a new `Normalize` transform for 1 or more channels.
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self { mean, std }
    }

    /// Standard ImageNet normalization (3 channels).
    pub fn imagenet() -> Self {
        Self {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }
}

impl Transform for Normalize {
    type Input = (Vec<f32>, Vec<usize>); // (data, shape [C, H, W] or [C, L])
    type Output = (Vec<f32>, Vec<usize>);

    fn transform(&self, (mut data, shape): Self::Input) -> Result<Self::Output> {
        if shape.is_empty() {
            bail!("Normalize transform requires a non-empty shape");
        }
        let channels = shape[0];
        if channels != self.mean.len() || channels != self.std.len() {
            bail!(
                "Normalize channel count mismatch: input has {} channels, mean has {}, std has {}",
                channels,
                self.mean.len(),
                self.std.len()
            );
        }

        let channel_stride = data.len() / channels;
        for c in 0..channels {
            let mean_c = self.mean[c];
            let std_c = self.std[c];
            if std_c == 0.0 {
                bail!("Normalize std dev cannot be zero for channel {}", c);
            }
            let start = c * channel_stride;
            let end = start + channel_stride;
            for val in &mut data[start..end] {
                *val = (*val - mean_c) / std_c;
            }
        }
        Ok((data, shape))
    }
}

/// Multiplies all elements in a flat float buffer by a scaling factor (e.g. `1.0 / 255.0`).
#[derive(Debug, Clone, Copy)]
pub struct Scale {
    /// Scaling factor.
    pub factor: f32,
}

impl Scale {
    /// Creates a new `Scale` transform with the given factor.
    pub fn new(factor: f32) -> Self {
        Self { factor }
    }
}

impl Transform for Scale {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    fn transform(&self, mut input: Self::Input) -> Result<Self::Output> {
        for x in &mut input {
            *x *= self.factor;
        }
        Ok(input)
    }
}

/// Randomly flips 2D or 3D image data ([C, H, W]) horizontally along the width axis with probability `p`.
#[derive(Debug, Clone, Copy)]
pub struct RandomHorizontalFlip {
    /// Probability of flipping (default 0.5).
    pub p: f64,
}

impl RandomHorizontalFlip {
    /// Creates a new `RandomHorizontalFlip` with probability `p`.
    pub fn new(p: f64) -> Self {
        Self { p }
    }
}

impl Default for RandomHorizontalFlip {
    fn default() -> Self {
        Self { p: 0.5 }
    }
}

impl Transform for RandomHorizontalFlip {
    type Input = (Vec<f32>, Vec<usize>); // (data, shape [C, H, W])
    type Output = (Vec<f32>, Vec<usize>);

    fn transform(&self, (data, shape): Self::Input) -> Result<Self::Output> {
        if shape.len() != 3 {
            bail!("RandomHorizontalFlip requires 3D shape [C, H, W]");
        }
        let mut rng = rand::thread_rng();
        if rng.gen_bool(self.p) {
            let channels = shape[0];
            let height = shape[1];
            let width = shape[2];

            let mut flipped = vec![0.0f32; data.len()];
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let src_idx = c * height * width + h * width + w;
                        let dst_idx = c * height * width + h * width + (width - 1 - w);
                        flipped[dst_idx] = data[src_idx];
                    }
                }
            }
            Ok((flipped, shape))
        } else {
            Ok((data, shape))
        }
    }
}

/// Center-crops a 3D tensor ([C, H, W]) to target height `crop_h` and width `crop_w`.
#[derive(Debug, Clone, Copy)]
pub struct CenterCrop {
    /// Target height.
    pub crop_h: usize,
    /// Target width.
    pub crop_w: usize,
}

impl CenterCrop {
    /// Creates a new `CenterCrop` with target dimensions.
    pub fn new(crop_h: usize, crop_w: usize) -> Self {
        Self { crop_h, crop_w }
    }
}

impl Transform for CenterCrop {
    type Input = (Vec<f32>, Vec<usize>); // (data, shape [C, H, W])
    type Output = (Vec<f32>, Vec<usize>);

    fn transform(&self, (data, shape): Self::Input) -> Result<Self::Output> {
        if shape.len() != 3 {
            bail!("CenterCrop requires 3D shape [C, H, W]");
        }
        let c = shape[0];
        let h = shape[1];
        let w = shape[2];

        if self.crop_h > h || self.crop_w > w {
            bail!(
                "Crop dimensions [{}, {}] exceed image dimensions [{}, {}]",
                self.crop_h,
                self.crop_w,
                h,
                w
            );
        }

        let start_h = (h - self.crop_h) / 2;
        let start_w = (w - self.crop_w) / 2;

        let mut cropped = Vec::with_capacity(c * self.crop_h * self.crop_w);
        for ch in 0..c {
            for row in start_h..(start_h + self.crop_h) {
                let row_start = ch * h * w + row * w + start_w;
                let row_end = row_start + self.crop_w;
                cropped.extend_from_slice(&data[row_start..row_end]);
            }
        }
        Ok((cropped, vec![c, self.crop_h, self.crop_w]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_transform() {
        let scale = Scale::new(0.5);
        let data = vec![1.0, 2.0, 4.0, 8.0];
        let out = scale.transform(data).unwrap();
        assert_eq!(out, vec![0.5, 1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_normalize_transform() {
        let norm = Normalize::new(vec![0.5, 1.0], vec![0.5, 2.0]);
        let data = vec![1.0, 0.5, 5.0, 1.0]; // 2 channels, 2 elements each
        let (out, shape) = norm.transform((data, vec![2, 2])).unwrap();
        assert_eq!(shape, vec![2, 2]);
        // ch0: (1.0-0.5)/0.5 = 1.0, (0.5-0.5)/0.5 = 0.0
        // ch1: (5.0-1.0)/2.0 = 2.0, (1.0-1.0)/2.0 = 0.0
        assert_eq!(out, vec![1.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_center_crop_transform() {
        let crop = CenterCrop::new(2, 2);
        // 1 channel, 4x4 image
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let (out, shape) = crop.transform((data, vec![1, 4, 4])).unwrap();
        assert_eq!(shape, vec![1, 2, 2]);
        // Center 2x2 of 4x4: rows 1..3, cols 1..3 -> [5, 6, 9, 10]
        assert_eq!(out, vec![5.0, 6.0, 9.0, 10.0]);
    }
}
