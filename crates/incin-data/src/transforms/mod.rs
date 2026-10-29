//! Data transformation and augmentation pipeline.
//!
//! Provides traits and implementations for data preprocessing, batch normalization,
//! image transformations, and pipeline composition.

use crate::loader::{BatchResult as Result, DataError};
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

/// Rejects a transform input that violates its declared contract.
fn invalid_input(message: impl Into<String>) -> DataError {
    DataError::InvalidInput(message.into())
}
use rand::RngExt as _;

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
            return Err(invalid_input(
                "Normalize transform requires a non-empty shape",
            ));
        }
        let channels = shape[0];
        if channels != self.mean.len() || channels != self.std.len() {
            return Err(invalid_input(format!(
                "Normalize channel count mismatch: input has {} channels, mean has {}, std has {}",
                channels,
                self.mean.len(),
                self.std.len()
            )));
        }

        let channel_stride = data.len() / channels;
        for c in 0..channels {
            let mean_c = self.mean[c];
            let std_c = self.std[c];
            if std_c == 0.0 {
                return Err(invalid_input(format!(
                    "Normalize std dev cannot be zero for channel {c}"
                )));
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
            return Err(invalid_input(
                "RandomHorizontalFlip requires 3D shape [C, H, W]",
            ));
        }
        let mut rng = rand::rng();
        if rng.random_bool(self.p) {
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
            return Err(invalid_input("CenterCrop requires 3D shape [C, H, W]"));
        }
        let c = shape[0];
        let h = shape[1];
        let w = shape[2];

        if self.crop_h > h || self.crop_w > w {
            return Err(invalid_input(format!(
                "Crop dimensions [{}, {}] exceed image dimensions [{}, {}]",
                self.crop_h, self.crop_w, h, w
            )));
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

/// Converts unsigned-byte samples to `f32` without rescaling.
///
/// This is the `ToTensor` half that [`Normalize`] and [`Scale`] assume but do
/// not provide: [`Normalize`] expects float data on a `[C, ..]` layout and
/// [`Scale`] multiplies a bare float buffer by a factor (e.g. `1.0 / 255.0`
/// for a `[0, 1]` range). Neither converts `u8` samples, so a pipeline that
/// starts from bytes would otherwise have nowhere to cross the dtype
/// boundary. This transform only casts (`0` to `255.0`); chain it with
/// `Scale::new(1.0 / 255.0)` when the downstream stages expect `[0, 1]`
/// floats, keeping the scaling factor in the one place that already owns it.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToTensor;

impl Transform for ToTensor {
    type Input = (Vec<u8>, Vec<usize>); // (data, shape)
    type Output = (Vec<f32>, Vec<usize>);

    fn transform(&self, (data, shape): Self::Input) -> Result<Self::Output> {
        if shape.is_empty() {
            return Err(invalid_input(
                "ToTensor transform requires a non-empty shape",
            ));
        }
        let mut numel = 1usize;
        for dim in &shape {
            numel = numel.checked_mul(*dim).ok_or_else(|| {
                invalid_input(format!(
                    "ToTensor shape {shape:?} overflows the element count"
                ))
            })?;
        }
        if numel != data.len() {
            return Err(invalid_input(format!(
                "ToTensor data length {} does not match shape {shape:?} ({} elements)",
                data.len(),
                numel
            )));
        }
        Ok((data.into_iter().map(|byte| byte as f32).collect(), shape))
    }
}

/// Resizes 3D image data ([C, H, W]) to a target height and width with nearest
/// neighbor sampling.
///
/// Each output pixel copies the nearest input pixel (`src = dst * in / out`
/// in integer arithmetic), so label maps and masks survive the resize without
/// the blended values a linear filter would invent. A target dimension of
/// zero has no output pixels and is refused.
#[derive(Debug, Clone, Copy)]
pub struct Resize {
    /// Target height.
    pub new_h: usize,
    /// Target width.
    pub new_w: usize,
}

impl Resize {
    /// Creates a new `Resize` with target dimensions.
    pub fn new(new_h: usize, new_w: usize) -> Self {
        Self { new_h, new_w }
    }
}

impl Transform for Resize {
    type Input = (Vec<f32>, Vec<usize>); // (data, shape [C, H, W])
    type Output = (Vec<f32>, Vec<usize>);

    fn transform(&self, (data, shape): Self::Input) -> Result<Self::Output> {
        if shape.len() != 3 {
            return Err(invalid_input("Resize requires 3D shape [C, H, W]"));
        }
        if self.new_h == 0 || self.new_w == 0 {
            return Err(invalid_input(format!(
                "Resize target dimensions [{}, {}] must both be non-zero",
                self.new_h, self.new_w
            )));
        }
        let channels = shape[0];
        let height = shape[1];
        let width = shape[2];
        if data.len() != channels * height * width {
            return Err(invalid_input(format!(
                "Resize data length {} does not match shape {shape:?}",
                data.len()
            )));
        }

        let mut resized = Vec::with_capacity(channels * self.new_h * self.new_w);
        for c in 0..channels {
            for out_h in 0..self.new_h {
                let in_h = out_h * height / self.new_h;
                for out_w in 0..self.new_w {
                    let in_w = out_w * width / self.new_w;
                    resized.push(data[c * height * width + in_h * width + in_w]);
                }
            }
        }
        Ok((resized, vec![channels, self.new_h, self.new_w]))
    }
}

/// Randomly crops a 3D tensor ([C, H, W]) to target height `crop_h` and width
/// `crop_w`, after optionally zero-padding every side by `padding` pixels.
///
/// [`CenterCrop`] covers the deterministic crop; this is the augmentation
/// counterpart, with a uniformly sampled top-left corner. The padding exists
/// for the translate-then-crop augmentation (e.g. padding 4 on CIFAR): the
/// image grows by `2 * padding` in each spatial dimension before the crop is
/// sampled. A crop larger than the padded image is refused rather than
/// sampled from an empty range.
#[derive(Debug, Clone, Copy)]
pub struct RandomCrop {
    /// Target height.
    pub crop_h: usize,
    /// Target width.
    pub crop_w: usize,
    /// Zero-padding applied to every side before cropping.
    pub padding: usize,
}

impl RandomCrop {
    /// Creates a new `RandomCrop` with target dimensions and no padding.
    pub fn new(crop_h: usize, crop_w: usize) -> Self {
        Self {
            crop_h,
            crop_w,
            padding: 0,
        }
    }

    /// Sets the zero-padding applied to every side before cropping.
    #[must_use]
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }
}

impl Transform for RandomCrop {
    type Input = (Vec<f32>, Vec<usize>); // (data, shape [C, H, W])
    type Output = (Vec<f32>, Vec<usize>);

    fn transform(&self, (data, shape): Self::Input) -> Result<Self::Output> {
        if shape.len() != 3 {
            return Err(invalid_input("RandomCrop requires 3D shape [C, H, W]"));
        }
        if self.crop_h == 0 || self.crop_w == 0 {
            return Err(invalid_input(format!(
                "RandomCrop dimensions [{}, {}] must both be non-zero",
                self.crop_h, self.crop_w
            )));
        }
        let channels = shape[0];
        let height = shape[1];
        let width = shape[2];
        if data.len() != channels * height * width {
            return Err(invalid_input(format!(
                "RandomCrop data length {} does not match shape {shape:?}",
                data.len()
            )));
        }

        let padded_h = height + 2 * self.padding;
        let padded_w = width + 2 * self.padding;
        if self.crop_h > padded_h || self.crop_w > padded_w {
            return Err(invalid_input(format!(
                "Crop dimensions [{}, {}] exceed padded image dimensions [{}, {}]",
                self.crop_h, self.crop_w, padded_h, padded_w
            )));
        }

        let mut rng = rand::rng();
        let top = rng.random_range(0..=(padded_h - self.crop_h));
        let left = rng.random_range(0..=(padded_w - self.crop_w));

        // Reads through the padding without materializing it: coordinates
        // inside the pad band are zeros, the rest index the input.
        let at = |channel: usize, row: usize, col: usize| -> f32 {
            if row < self.padding
                || row >= self.padding + height
                || col < self.padding
                || col >= self.padding + width
            {
                0.0
            } else {
                data[channel * height * width + (row - self.padding) * width + (col - self.padding)]
            }
        };
        let mut cropped = Vec::with_capacity(channels * self.crop_h * self.crop_w);
        for c in 0..channels {
            for row in top..(top + self.crop_h) {
                for col in left..(left + self.crop_w) {
                    cropped.push(at(c, row, col));
                }
            }
        }
        Ok((cropped, vec![channels, self.crop_h, self.crop_w]))
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

    #[test]
    fn normalize_rejects_empty_shape_as_invalid_input() {
        let error = Normalize::new(vec![1.0], vec![1.0])
            .transform((vec![1.0], Vec::new()))
            .expect_err("an empty shape has no channel axis");
        assert!(matches!(error, DataError::InvalidInput(_)));
    }

    #[test]
    fn normalize_rejects_channel_count_mismatch_as_invalid_input() {
        let error = Normalize::new(vec![1.0], vec![1.0])
            .transform((vec![1.0, 2.0], vec![3]))
            .expect_err("three channels cannot be normalized by one mean and std");
        assert!(matches!(error, DataError::InvalidInput(_)));
    }

    #[test]
    fn normalize_rejects_zero_std_as_invalid_input() {
        let error = Normalize::new(vec![1.0], vec![0.0])
            .transform((vec![1.0], vec![1]))
            .expect_err("a zero standard deviation divides every value by zero");
        assert!(matches!(error, DataError::InvalidInput(_)));
    }

    #[test]
    fn random_horizontal_flip_rejects_non_3d_shape_as_invalid_input() {
        let error = RandomHorizontalFlip::default()
            .transform((vec![1.0, 2.0], vec![2]))
            .expect_err("the flip kernel indexes channels, height, and width");
        assert!(matches!(error, DataError::InvalidInput(_)));
    }

    #[test]
    fn center_crop_rejects_crop_larger_than_image_as_invalid_input() {
        let error = CenterCrop::new(8, 8)
            .transform((vec![0.0; 16], vec![1, 4, 4]))
            .expect_err("a crop larger than the image has no center");
        assert!(matches!(error, DataError::InvalidInput(_)));
    }

    #[test]
    fn test_to_tensor_casts_bytes_without_rescaling() {
        let (out, shape) = ToTensor.transform((vec![0u8, 128, 255], vec![3])).unwrap();
        assert_eq!(shape, vec![3]);
        assert_eq!(out, vec![0.0, 128.0, 255.0]);
    }

    #[test]
    fn to_tensor_pairs_with_scale_for_unit_range() {
        // `ToTensor` owns the dtype crossing, `Scale` owns the range: the two
        // stages compose to `[0, 1]` without either duplicating the other.
        let (floats, shape) = ToTensor.transform((vec![0u8, 255], vec![2])).unwrap();
        let out = Scale::new(1.0 / 255.0).transform(floats).unwrap();
        assert_eq!(shape, vec![2]);
        assert_eq!(out, vec![0.0, 1.0]);
    }

    #[test]
    fn to_tensor_rejects_data_that_disagrees_with_its_shape() {
        let error = ToTensor
            .transform((vec![1u8, 2], vec![3]))
            .expect_err("two bytes cannot fill three slots");
        assert!(matches!(error, DataError::InvalidInput(_)));
    }

    #[test]
    fn test_resize_nearest_neighbor_replicates() {
        let resize = Resize::new(4, 4);
        // 1 channel, 2x2 image.
        let (out, shape) = resize
            .transform((vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]))
            .unwrap();
        assert_eq!(shape, vec![1, 4, 4]);
        // Each input pixel covers a 2x2 output block.
        assert_eq!(
            out,
            vec![
                1.0, 1.0, 2.0, 2.0, //
                1.0, 1.0, 2.0, 2.0, //
                3.0, 3.0, 4.0, 4.0, //
                3.0, 3.0, 4.0, 4.0,
            ]
        );
    }

    #[test]
    fn test_resize_identity_leaves_values_untouched() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let (out, shape) = Resize::new(3, 4)
            .transform((data.clone(), vec![1, 3, 4]))
            .unwrap();
        assert_eq!(shape, vec![1, 3, 4]);
        assert_eq!(out, data);
    }

    #[test]
    fn resize_rejects_zero_targets_and_non_3d_shapes() {
        let error = Resize::new(0, 4)
            .transform((vec![1.0; 4], vec![1, 2, 2]))
            .expect_err("a zero target height has no output pixels");
        assert!(matches!(error, DataError::InvalidInput(_)));
        let error = Resize::new(2, 2)
            .transform((vec![1.0, 2.0], vec![2]))
            .expect_err("resize indexes channels, height, and width");
        assert!(matches!(error, DataError::InvalidInput(_)));
    }

    #[test]
    fn random_crop_covering_the_whole_image_is_deterministic() {
        // Exactly one valid position, so no sampling happens: values pass
        // through on every channel.
        let data: Vec<f32> = (0..18).map(|x| x as f32).collect();
        let (out, shape) = RandomCrop::new(2, 3)
            .transform((data.clone(), vec![3, 2, 3]))
            .unwrap();
        assert_eq!(shape, vec![3, 2, 3]);
        assert_eq!(out, data);
    }

    #[test]
    fn random_crop_samples_inside_the_image() {
        // A 1x4 crop of a 1-channel 4x4 image: 16 valid positions, every one
        // of them a single row of the input.
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        for _ in 0..32 {
            let (out, shape) = RandomCrop::new(1, 4)
                .transform((data.clone(), vec![1, 4, 4]))
                .unwrap();
            assert_eq!(shape, vec![1, 1, 4]);
            let row = out[0] as usize / 4;
            assert_eq!(
                out,
                vec![
                    row as f32 * 4.0,
                    row as f32 * 4.0 + 1.0,
                    row as f32 * 4.0 + 2.0,
                    row as f32 * 4.0 + 3.0
                ]
            );
        }
    }

    #[test]
    fn random_crop_with_padding_keeps_shapes_and_refuses_oversize_crops() {
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let (out, shape) = RandomCrop::new(4, 4)
            .with_padding(1)
            .transform((data, vec![1, 4, 4]))
            .unwrap();
        assert_eq!(shape, vec![1, 4, 4]);
        assert_eq!(out.len(), 16);
        // Every output value is either a pad zero or an input value.
        assert!(
            out.iter()
                .all(|value| *value == 0.0 || (0.0..16.0).contains(value))
        );

        let error = RandomCrop::new(8, 8)
            .with_padding(1)
            .transform((vec![0.0; 16], vec![1, 4, 4]))
            .expect_err("a 6x6 padded image has no 8x8 crop");
        assert!(matches!(error, DataError::InvalidInput(_)));
    }
}
