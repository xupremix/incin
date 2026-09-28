//! Example: classify synthetic galaxies with a small CNN, live.
//!
//! Elliptical vs spiral galaxies are generated procedurally (seeded, no
//! downloads): exponential blobs with random ellipticity against
//! two-armed logarithmic spirals on an exponential disk, plus noise. A
//! Conv-ReLU-Pool x2 + Linear head trains with Adam and cross-entropy;
//! each epoch prints loss, a text bar, and test accuracy, and the end
//! shows ASCII thumbnails with predicted vs true labels.
//!
//! Run it with `cargo run -p incin --example galaxy_classifier`.

use incin::prelude::*;

// Tiny deterministic RNG (xorshift64* + Box-Muller-free uniform noise).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
}

const W: usize = 28;

/// Elliptical galaxy: exponential profile, random flattening + rotation.
fn elliptical(rng: &mut Rng) -> Vec<f32> {
    let sigma = rng.range(4.0, 7.0);
    let flat = rng.range(0.3, 0.7);
    let angle = rng.range(0.0, std::f64::consts::PI);
    let (ca, sa) = (angle.cos(), angle.sin());
    let mut img = Vec::with_capacity(W * W);
    for y in 0..W {
        for x in 0..W {
            let dx = x as f64 - 13.5;
            let dy = y as f64 - 13.5;
            let rx = dx * ca + dy * sa;
            let ry = (-dx * sa + dy * ca) / flat;
            let r2 = rx * rx + ry * ry;
            let v = (-r2 / (2.0 * sigma * sigma)).exp() + rng.range(0.0, 0.08);
            img.push(v.min(1.0) as f32);
        }
    }
    img
}

/// Spiral galaxy: exponential disk plus two logarithmic arms.
fn spiral(rng: &mut Rng) -> Vec<f32> {
    let arms = 2.0;
    let twist = rng.range(0.35, 0.6);
    let phase = rng.range(0.0, std::f64::consts::TAU);
    let width = rng.range(0.35, 0.55);
    let mut img = Vec::with_capacity(W * W);
    for y in 0..W {
        for x in 0..W {
            let dx = x as f64 - 13.5;
            let dy = y as f64 - 13.5;
            let r = (dx * dx + dy * dy).sqrt().max(0.5);
            let theta = dy.atan2(dx);
            let disk = (-r / 8.0).exp() * 0.7;
            // Distance from the nearest arm center, wrapped to [-pi, pi].
            let mut d = (theta - (twist * r.ln() + phase)) % (std::f64::consts::TAU / arms);
            if d > std::f64::consts::PI / arms {
                d -= std::f64::consts::TAU / arms;
            }
            if d < -std::f64::consts::PI / arms {
                d += std::f64::consts::TAU / arms;
            }
            let arm = (-(d * d) / (2.0 * width * width)).exp() * (-r / 10.0).exp();
            let v = disk + arm + rng.range(0.0, 0.08);
            img.push(v.min(1.0) as f32);
        }
    }
    img
}

fn thumbnail(img: &[f32]) -> String {
    const SHADES: &[u8] = b" .:-=+*#%@";
    let mut out = String::new();
    for y in (0..W).step_by(2) {
        for x in (0..W).step_by(2) {
            // 2x2 average down to 14x14.
            let mut sum = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    sum += img[(y + dy) * W + x + dx];
                }
            }
            let level = ((sum / 4.0).clamp(0.0, 1.0) * 9.0) as usize;
            out.push(SHADES[level] as char);
        }
        out.push('\n');
    }
    out
}

fn bar(loss: f64, ref_loss: f64) -> String {
    let width = ((loss / ref_loss).clamp(0.0, 1.0) * 30.0) as usize;
    format!(
        "{}{}",
        "#".repeat(30 - width.min(30)),
        "-".repeat(width.min(30))
    )
}

fn main() -> Result<()> {
    println!("Generating synthetic galaxies (seeded)...");
    let mut rng = Rng(0x12345678);
    let mut train_images = Vec::new();
    let mut train_labels: Vec<i64> = Vec::new();
    for i in 0..240 {
        let is_elliptical = i % 2 == 0;
        let img = if is_elliptical {
            elliptical(&mut rng)
        } else {
            spiral(&mut rng)
        };
        train_images.extend_from_slice(&img);
        train_labels.push(if is_elliptical { 0 } else { 1 });
    }
    let mut test_images = Vec::new();
    let mut test_labels: Vec<i64> = Vec::new();
    for i in 0..48 {
        let is_elliptical = i % 2 == 0;
        let img = if is_elliptical {
            elliptical(&mut rng)
        } else {
            spiral(&mut rng)
        };
        test_images.extend_from_slice(&img);
        test_labels.push(if is_elliptical { 0 } else { 1 });
    }
    println!("240 train + 48 test images (28x28, 0=elliptical 1=spiral).");
    println!(
        "Sample spiral:\n{}",
        thumbnail(&test_images[28 * 28..2 * 28 * 28])
    );

    type B = DefaultBackend;
    let model = seq![
        Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((8, 1))?,
        ReLU,
        MaxPool2d::<typenum::U2, typenum::U2>::new()?,
        Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((16, 8))?,
        ReLU,
        MaxPool2d::<typenum::U2, typenum::U2>::new()?,
        Flatten::new(1isize, -1isize),
        Linear::<Dyn, B>::build((16 * 7 * 7, 32))?,
        ReLU,
        Linear::<Dyn, B>::build((32, 2))?,
    ];
    let mut optim = Adam::<B>::from_module(&model, 1e-3)?;

    let batch = 16;
    let mut first_loss = 0.0;
    for epoch in 0..8 {
        // Deterministic rotation instead of shuffling: offset the start.
        let offset = (epoch * 37) % train_labels.len();
        let mut epoch_loss = 0.0;
        let mut steps = 0;
        for start in (0..train_labels.len()).step_by(batch) {
            let idx: Vec<usize> = (0..batch)
                .map(|k| (offset + start + k) % train_labels.len())
                .collect();
            let mut bx = Vec::with_capacity(batch * 784);
            let mut by = Vec::with_capacity(batch);
            for &i in &idx {
                bx.extend_from_slice(&train_images[i * 784..(i + 1) * 784]);
                by.push(train_labels[i]);
            }
            let images = Tensor::<Dyn, B>::from_slice(&bx, vec![batch, 1, 28, 28])?;
            let labels = Tensor::<Dyn, B, i64>::from_slice(&by, vec![batch])?;
            let out = model.forward(images)?;
            let loss = out.cross_entropy_loss(&labels)?;
            let value = loss.to_scalar::<f32>()? as f64;
            epoch_loss += value;
            steps += 1;
            let grads = loss.backward()?;
            optim.step(&grads)?;
        }
        epoch_loss /= steps as f64;
        if epoch == 0 {
            first_loss = epoch_loss;
        }
        if epoch == 7 {
            assert!(
                epoch_loss < first_loss,
                "8 epochs must reduce the loss: {epoch_loss} vs first {first_loss}"
            );
        }
        // Test accuracy.
        let test_x = Tensor::<Dyn, B>::from_slice(&test_images, vec![48, 1, 28, 28])?;
        let logits = model.forward(test_x)?.to_vec1::<f32>()?;
        let mut correct = 0;
        for (i, &true_label) in test_labels.iter().enumerate() {
            let (mut best, mut best_j) = (logits[i * 2], 0);
            if logits[i * 2 + 1] > best {
                best = logits[i * 2 + 1];
                best_j = 1;
            }
            let _ = best;
            if best_j as i64 == true_label {
                correct += 1;
            }
        }
        let acc = correct as f64 / test_labels.len() as f64;
        println!(
            "epoch {epoch}: loss {epoch_loss:.4} {} acc {acc:.2}",
            bar(epoch_loss, first_loss.max(1e-9))
        );
    }

    // Gallery: first 4 test samples with predictions.
    println!("\nGallery (pred/true, E=elliptical S=spiral):");
    let test_x = Tensor::<Dyn, B>::from_slice(&test_images[..4 * 784], vec![4, 1, 28, 28])?;
    let logits = model.forward(test_x)?.to_vec1::<f32>()?;
    for i in 0..4 {
        let pred = if logits[i * 2 + 1] > logits[i * 2] {
            'S'
        } else {
            'E'
        };
        let actual = if test_labels[i] == 1 { 'S' } else { 'E' };
        let mark = if pred == actual { "ok" } else { "WRONG" };
        println!("sample {i}: pred {pred} true {actual} [{mark}]");
        println!("{}", thumbnail(&test_images[i * 784..(i + 1) * 784]));
    }
    println!("PASS: galaxy CNN trained and picturing its answers");
    Ok(())
}
