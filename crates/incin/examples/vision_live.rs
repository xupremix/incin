//! Example: live vision-training dashboard in your browser.
//!
//! Trains a small CNN on a 10-way image dataset while serving a dashboard
//! on http://127.0.0.1:8000 — loss curve, accuracy, and a test gallery
//! with predicted vs true labels, auto-refreshed every few seconds. The
//! page is rendered per request from the live training state, so what you
//! see is what the model just did.
//!
//! ```text
//! cargo run -p incin --features cpu-blas --example vision_live -- --dataset fashion
//! cargo run -p incin --features cpu-blas --example vision_live -- --dataset cifar
//! ```
//!
//! The `cpu-blas` feature is not required - a default build trains the
//! same model to the same accuracy - but it is worth knowing about. It
//! hands large f32 GEMMs to a blocked, register-tiled kernel, and because
//! the CPU `conv2d` is im2col plus a batched matmul, that is exactly the
//! hot loop here: measured on a 4-core CPU, the identical four-step
//! CIFAR-10 batch took 2m15s on a default build and 31s with the feature
//! (4.4x). Without it this example is a long coffee break rather than a
//! demo.
//!
//! `fashion` is Fashion-MNIST (1x28x28), `cifar` is CIFAR-10 (3x32x32).
//! Both are ten-way, so the model, the training loop and the page below
//! are one code path; only the geometry, the class names and the
//! training budget differ. Each trains a fast subset on CPU — swap the
//! fields in its `Corpus` for the full splits.
//!
//! The recipe is the ordinary small-image one, and each piece is here
//! because it earned its place on the measured curve rather than because
//! it is conventional:
//!
//! - per-channel standardization, from the corpus's own statistics;
//! - train-only random crop with a 4-pixel pad plus a horizontal flip,
//!   through `incin::transforms` — on 4000 images this is worth more
//!   than any single architecture change;
//! - a VGG-style stack: every convolution followed by batch norm and
//!   ReLU, widths 32/64/128, two 2x2 pools;
//! - AdamW with decoupled weight decay and a warmup + cosine schedule
//!   rather than a flat step size.
//!
//! Regularization is augmentation plus weight decay, not dropout: the
//! test pass cannot switch the model to evaluation mode without also
//! switching batch norm over to running statistics it has never written,
//! and half a mechanism is worse than none.

use incin::prelude::*;
use incin::transforms::{Compose, Normalize, RandomCrop, RandomHorizontalFlip, Transform};
use incin_core::exec::GradMode;
use incin_data::vision::cifar::Cifar10Dataset;
use incin_data::vision::fashion_mnist::FashionMnistDataset;
use incin_data::{DataError, Dataset};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Batch size. 32 is small enough that batch-norm statistics stay noisy
/// enough to regularize, and large enough that the per-op overhead this
/// example spends most of its time on stays amortized.
const BATCH: usize = 32;

/// One convolution per resolution, and its channel width: 16 channels at
/// 32x32, 32 at 16x16, 64 at 8x8, then a 4x4 classifier head. Three
/// pooling stages keep the arithmetic where it is cheap, and the whole
/// stack is about 2.9M multiply-accumulates per image - which is a
/// budget, not an accident. This CPU sustains roughly 120 MFLOP/s
/// through its own im2col + GEMM conv path and 540 MFLOP/s with the
/// `cpu-blas` feature, which is 4.4x on this shape and the difference
/// between a demo that finishes and one that does not.
const WIDTHS: [usize; 3] = [16, 32, 64];

/// One materialised split: flat channel-major pixels in `[0, 1]` plus
/// integer labels, already cut down to the demo subset.
struct Split {
    pixels: Vec<f32>,
    labels: Vec<i64>,
    total: usize,
}

/// Everything about a corpus that is not the data itself.
struct Corpus {
    /// Directory the dataset is loaded from (relative to the workspace root).
    dir: &'static str,
    /// Display name, also the page title.
    name: &'static str,
    classes: [&'static str; 10],
    channels: usize,
    side: usize,
    /// Fast demo subset sizes, not caps: the same code trains the full split.
    train: usize,
    test: usize,
    epochs: usize,
    /// AdamW peak step size, reached after the warmup ramp.
    lr: f64,
    /// Decoupled weight decay, applied by AdamW and not by the loss.
    weight_decay: f64,
    /// Steps spent ramping from zero to `lr`.
    warmup: usize,
    /// Per-channel mean and standard deviation, in `[0, 1]` pixel units.
    mean: Vec<f32>,
    std: Vec<f32>,
    /// Pixels of zero padding around each image before the random crop.
    crop_pad: usize,
}

/// Flat `[batch, channels, side, side]` shape for a batch of `n`.
fn batch_shape(spec: &Corpus, n: usize) -> Vec<usize> {
    vec![n, spec.channels, spec.side, spec.side]
}

/// Pixels per image, channel-major.
fn pixels_per_image(spec: &Corpus) -> usize {
    spec.channels * spec.side * spec.side
}

/// The pipeline every image passes through: augmentation for training
/// only, then standardization for both splits. Augmentation draws from
/// the process RNG inside `incin-data`, so a training epoch sees a
/// different crop of every image every time it comes round.
struct Prep {
    normalize: Normalize,
    augment: Compose<(Vec<f32>, Vec<usize>)>,
}

impl Prep {
    fn new(spec: &Corpus) -> Self {
        Self {
            normalize: Normalize::new(spec.mean.clone(), spec.std.clone()),
            augment: Compose::new()
                .push(RandomCrop::new(spec.side, spec.side).with_padding(spec.crop_pad))
                .push(RandomHorizontalFlip::new(0.5)),
        }
    }

    /// Standardizes one `[channels, side, side]` image.
    fn standardize(&self, pixels: Vec<f32>, spec: &Corpus) -> Result<Vec<f32>> {
        let shape = vec![spec.channels, spec.side, spec.side];
        self.normalize
            .transform((pixels, shape))
            .map(|(out, _)| out)
            .map_err(data_err)
    }

    /// Augments one image, then standardizes it.
    fn augmented(&self, pixels: Vec<f32>, spec: &Corpus) -> Result<Vec<f32>> {
        let shape = vec![spec.channels, spec.side, spec.side];
        self.augment
            .transform((pixels, shape))
            .map_err(data_err)
            .and_then(|(out, _)| self.standardize(out, spec))
    }
}

/// Copies `indices` out of `data` into a `Split`, refusing to guess when
/// the corpus is smaller than the requested subset.
fn subset<D>(data: &D, indices: impl Iterator<Item = usize>, total: usize) -> Result<Split>
where
    D: Dataset<Item = (Vec<f32>, u8)>,
{
    let mut pixels = Vec::new();
    let mut labels = Vec::new();
    for i in indices {
        let (img, label) = data.get(i).map_err(data_err)?.ok_or_else(|| {
            incin::Error::Msg(format!(
                "subset index {i} is past the {total}-image split; raise the subset size or \
                 lower the stride"
            ))
        })?;
        pixels.extend_from_slice(&img);
        labels.push(i64::from(label));
    }
    Ok(Split {
        pixels,
        labels,
        total,
    })
}

/// `n` evenly spaced indices over a split of `len` images: the test
/// subset should not be the first `n` images.
fn spread(len: usize, n: usize) -> impl Iterator<Item = usize> {
    let stride = (len / n).max(1);
    (0..len).step_by(stride).take(n)
}

/// A tiny LCG for the per-epoch shuffle. `incin-data` draws augmentation
/// randomness internally, so a repeated image order would be the one
/// source of repetition left in the epoch; this keeps the example free of
/// a `rand` dependency of its own.
struct Shuffle(u64);

impl Shuffle {
    fn shuffled(&mut self, n: usize) -> Vec<usize> {
        // Fisher-Yates, one pass down.
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            order.swap(i, (self.0 >> 33) as usize % (i + 1));
        }
        order
    }
}

fn load_fashion() -> Result<(Corpus, Split, Split)> {
    let spec = Corpus {
        dir: "./data/fashion-mnist",
        name: "fashion-mnist",
        classes: [
            "T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
            "Ankle",
        ],
        channels: 1,
        side: 28,
        train: 3000,
        test: 1000,
        epochs: 8,
        lr: 1e-3,
        weight_decay: 5e-4,
        warmup: 200,
        // Fashion-MNIST's own per-pixel mean and standard deviation.
        mean: vec![0.2860],
        std: vec![0.3530],
        crop_pad: 2,
    };
    let dir = PathBuf::from(spec.dir);
    println!("Loading {} into {:?}...", spec.name, dir);
    let train = FashionMnistDataset::new(&dir, true)?;
    let test = FashionMnistDataset::new(&dir, false)?;
    let tr = subset(&train, 0..spec.train.min(train.len()), train.len())?;
    let te = subset(&test, spread(test.len(), spec.test), test.len())?;
    Ok((spec, tr, te))
}

fn load_cifar() -> Result<(Corpus, Split, Split)> {
    let spec = Corpus {
        dir: "./data/cifar10",
        name: "cifar-10",
        classes: [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        channels: 3,
        side: 32,
        // A 3x32x32 image is several times the work per step and CIFAR is
        // the harder ten-way problem, so this is the longer of the two
        // runs: about 33 minutes on this box with `cpu-blas` on, for 1250
        // optimizer steps. More images beat more passes - but only once
        // the recipe was right: 1500 images for 6 epochs, then 4000 for 8,
        // both plateaued around 0.26-0.29 without normalization or
        // augmentation, and the same 4000 images reach 0.46 with them.
        train: 4000,
        test: 1000,
        epochs: 10,
        lr: 2e-3,
        weight_decay: 5e-4,
        warmup: 300,
        // The CIFAR-10 training-set channel statistics, the ones every
        // published baseline normalizes with.
        mean: vec![0.4914, 0.4822, 0.4465],
        std: vec![0.2470, 0.2435, 0.2616],
        crop_pad: 4,
    };
    let dir = PathBuf::from(spec.dir);
    println!("Loading {} into {:?}...", spec.name, dir);
    let train = Cifar10Dataset::new(&dir, true)?;
    let test = Cifar10Dataset::new(&dir, false)?;
    let tr = subset(&train, 0..spec.train.min(train.len()), train.len())?;
    let te = subset(&test, spread(test.len(), spec.test), test.len())?;
    Ok((spec, tr, te))
}

/// Prints the facts a reader should not have to take on trust: the split
/// sizes behind the subsets, the label range actually read, and the
/// channel-major layout of the first image.
fn smoke(spec: &Corpus, train: &Split, test: &Split) {
    let plane = spec.side * spec.side;
    let (lo, hi) = train
        .labels
        .iter()
        .chain(test.labels.iter())
        .fold((i64::MAX, i64::MIN), |(lo, hi), &l| (lo.min(l), hi.max(l)));
    println!(
        "{}: {} train + {} test images available, using {} train + {} test.",
        spec.name,
        train.total,
        test.total,
        train.labels.len(),
        test.labels.len()
    );
    println!(
        "  labels {lo}..{hi} against 0..{}, {} pixels per image",
        spec.classes.len() - 1,
        pixels_per_image(spec)
    );
    // Channel-major: channel c of image 0 starts at c * side * side, so the
    // first pixel of every plane is one screen apart, not adjacent.
    let planes: Vec<String> = (0..spec.channels)
        .map(|c| {
            format!(
                "c{c}@{}={:.3}",
                c * plane,
                train.pixels[c * plane] / spec.std[c] - spec.mean[c] / spec.std[c]
            )
        })
        .collect();
    println!(
        "  first image channel planes, standardized: {}",
        planes.join(" ")
    );
    println!("  batch shape {:?}", batch_shape(spec, BATCH));
}

/// What the dashboard renders, updated by the training thread.
struct Snapshot {
    title: String,
    losses: Vec<f64>,
    accs: Vec<f64>,
    gallery: Vec<(Vec<f32>, usize, usize)>,
    status: String,
    done: bool,
}

fn svg_curve(losses: &[f64], w: usize, h: usize) -> String {
    if losses.is_empty() {
        return format!("<svg width='{w}' height='{h}'></svg>");
    }
    let max = losses.iter().cloned().fold(0.0f64, f64::max).max(1e-9);
    let pts: Vec<String> = losses
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = if losses.len() == 1 {
                0.0
            } else {
                i as f64 / (losses.len() - 1) as f64 * (w as f64 - 8.0) + 4.0
            };
            let y = h as f64 - 4.0 - (v / max).clamp(0.0, 1.0) * (h as f64 - 8.0);
            format!("{x:.1},{y:.1}")
        })
        .collect();
    format!(
        "<svg width='{w}' height='{h}' style='background:#111'><polyline points='{}' \
         fill='none' stroke='#4af' stroke-width='2'/></svg>",
        pts.join(" ")
    )
}

fn gallery_html(spec: &Corpus, gallery: &[(Vec<f32>, usize, usize)]) -> String {
    let names: Vec<String> = spec.classes.iter().map(|s| format!("\"{s}\"")).collect();
    let mut out = format!(
        "<div id='gal'></div><script>\nconst SIDE={};\nconst CH={};\nconst NAMES=[",
        spec.side, spec.channels
    );
    out.push_str(&names.join(","));
    out.push_str("];\nconst GAL=[");
    for (pixels, pred, actual) in gallery {
        out.push_str(&format!(
            "{{p:[{}],pred:{},actual:{}}},",
            pixels
                .iter()
                .map(|v| format!("{:.3}", v.clamp(0.0, 1.0)))
                .collect::<Vec<_>>()
                .join(","),
            pred,
            actual
        ));
    }
    out.push_str(
        "];\nconst gal=document.getElementById('gal');\n\
         GAL.forEach((g,i)=>{const d=document.createElement('div');d.style.display='inline-block';d.style.margin='6px';d.style.textAlign='center';\n\
         const c=document.createElement('canvas');c.width=SIDE;c.height=SIDE;c.style.width=(SIDE*3)+'px';c.style.imageRendering='pixelated';\n\
         const x=c.getContext('2d');const im=x.createImageData(SIDE,SIDE);\n\
         for(let k=0;k<SIDE*SIDE;k++){if(CH===1){const v=Math.round(g.p[k]*255);im.data[4*k]=v;im.data[4*k+1]=v;im.data[4*k+2]=v;}\n\
         else{for(let c2=0;c2<3;c2++){im.data[4*k+c2]=Math.round(g.p[c2*SIDE*SIDE+k]*255);}}\n\
         im.data[4*k+3]=255;}\n\
         x.putImageData(im,0,0);d.appendChild(c);\n\
         const ok=g.pred===g.actual;const l=document.createElement('div');\n\
         l.style.color=ok?'#4f4':'#f44';l.textContent=(ok?'ok ':'WRONG ')+NAMES[g.pred]+'/'+NAMES[g.actual];\n\
         d.appendChild(l);gal.appendChild(d);});\n\
         </script>\n",
    );
    out
}

fn render(state: &Snapshot, spec: &Corpus) -> String {
    let last_loss = state.losses.last().copied().unwrap_or(0.0);
    let last_acc = state.accs.last().copied().unwrap_or(0.0);
    format!(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>\
         <meta http-equiv='refresh' content='5'>\
         <title>{title} live</title></head>\
         <body style='background:#000;color:#ddd;font-family:monospace'>\
         <h2>{title} CNN — live</h2>\
         <p>{} (epoch {}/{})</p>\
         <p>loss {last_loss:.4} &nbsp; test-acc {last_acc:.2}</p>\
         <h3>loss</h3>{} <h3>test gallery (pred/true)</h3>{}\
         </body></html>",
        state.status,
        state.losses.len(),
        if state.done { "done" } else { "training" },
        svg_curve(&state.losses, 480, 140),
        gallery_html(spec, &state.gallery),
        title = state.title,
    )
}

fn serve(listener: TcpListener, state: Arc<Mutex<Snapshot>>, spec: Arc<Corpus>) {
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut buf = [0u8; 4096];
        let _ = stream.read(&mut buf);
        let body = {
            let state = state.lock().unwrap();
            render(&state, &spec)
        };
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
    }
}

fn data_err(error: DataError) -> incin::Error {
    incin::Error::Msg(error.to_string())
}

fn main() -> Result<()> {
    // `--dataset <fashion|cifar>`, default `fashion`. An unrecognised name
    // is a refusal, never a silent default to the other corpus.
    let mut args = std::env::args().skip(1);
    let mut choice = String::from("fashion");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => {
                choice = args.next().ok_or_else(|| {
                    incin::Error::Msg("--dataset needs a value: fashion or cifar".to_string())
                })?;
            }
            "-h" | "--help" => {
                println!("usage: vision_live [--dataset fashion|cifar]");
                std::process::exit(0);
            }
            other => {
                return Err(incin::Error::Msg(format!(
                    "unrecognised argument {other:?}: usage is vision_live \
                     [--dataset fashion|cifar]"
                )));
            }
        }
    }
    let (spec, train, mut test) = match choice.as_str() {
        "fashion" => load_fashion()?,
        "cifar" => load_cifar()?,
        other => {
            return Err(incin::Error::Msg(format!(
                "unknown dataset {other:?}: the only options are fashion and cifar"
            )));
        }
    };
    smoke(&spec, &train, &test);

    // Standardize the test split once, up front: the test pass is a
    // forward over the whole subset and it should not pay for the
    // transform 1000 times per epoch.
    let prep = Prep::new(&spec);
    let per_image = pixels_per_image(&spec);
    for i in 0..test.labels.len() {
        let raw = test.pixels[i * per_image..(i + 1) * per_image].to_vec();
        test.pixels[i * per_image..(i + 1) * per_image]
            .copy_from_slice(&prep.standardize(raw, &spec)?);
    }

    type B = DefaultBackend;
    // Three 2x2 pools leave side/8 spatial, and the last convolution ends
    // at WIDTHS[2] channels whatever the corpus started with, so the
    // classifier sees WIDTHS[2] * (side/8)^2 features. (The input channel
    // count does *not* survive into this width; multiplying by it was a
    // bug the 1-channel Fashion-MNIST path could not see.)
    let pooled = (spec.side / 8) * (spec.side / 8);
    let model = seq![
        // Full resolution.
        Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((WIDTHS[0], spec.channels))?,
        BatchNorm2d::<s![dyn], B>::build((WIDTHS[0], 1e-5, 0.1))?,
        ReLU,
        MaxPool2d::<typenum::U2, typenum::U2>::new()?,
        // Half resolution.
        Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((WIDTHS[1], WIDTHS[0]))?,
        BatchNorm2d::<s![dyn], B>::build((WIDTHS[1], 1e-5, 0.1))?,
        ReLU,
        MaxPool2d::<typenum::U2, typenum::U2>::new()?,
        // Quarter resolution, then the classifier.
        Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((WIDTHS[2], WIDTHS[1]))?,
        BatchNorm2d::<s![dyn], B>::build((WIDTHS[2], 1e-5, 0.1))?,
        ReLU,
        MaxPool2d::<typenum::U2, typenum::U2>::new()?,
        Flatten::new(1isize, -1isize),
        Linear::<Dyn, B>::build((WIDTHS[2] * pooled, 128))?,
        ReLU,
        Linear::<Dyn, B>::build((128, spec.classes.len()))?,
    ];
    let mut optim = AdamW::<B>::from_module(&model, spec.lr)?;
    // Decoupled weight decay, set on the optimizer rather than folded into
    // the loss: the two only agree at `weight_decay == 0.0`.
    optim.weight_decay = spec.weight_decay;

    // Warm up from zero, then anneal to a hundredth of the peak. A flat
    // step size is the single cheapest thing to give up: the early steps
    // need the ramp, and the late ones need to settle.
    let steps_per_epoch = train.labels.len().div_ceil(BATCH);
    let total_steps = steps_per_epoch * spec.epochs;
    let mut schedule = CosineWithWarmup::new(spec.lr, spec.lr / 100.0, spec.warmup, total_steps);
    let mut shuffle = Shuffle(0x5eed_1234_9abc_def0);

    let state = Arc::new(Mutex::new(Snapshot {
        title: spec.name.to_string(),
        losses: Vec::new(),
        accs: Vec::new(),
        gallery: Vec::new(),
        status: "starting".to_string(),
        done: false,
    }));
    let listener = [8000u16, 8091, 8092, 8093]
        .into_iter()
        .find_map(|port| TcpListener::bind(("127.0.0.1", port)).ok())
        .expect("dashboard port should bind");
    let port = listener.local_addr().expect("bound port").port();
    println!(
        "dashboard: http://127.0.0.1:{port} (auto-refreshes every 5s)\n\
         training {} epochs over {} images, {steps_per_epoch} steps each, \
         AdamW peak lr {} weight decay {}",
        spec.epochs,
        train.labels.len(),
        spec.lr,
        spec.weight_decay
    );
    let server_state = state.clone();
    let server_spec = Arc::new(spec);
    std::thread::spawn({
        let server_spec = server_spec.clone();
        move || serve(listener, server_state, server_spec)
    });
    let spec = server_spec;

    let n_train = train.labels.len();
    let n_test = test.labels.len();
    let classes = spec.classes.len();
    for epoch in 0..spec.epochs {
        let order = shuffle.shuffled(n_train);
        let mut epoch_loss = 0.0;
        let mut steps = 0;
        for start in (0..n_train).step_by(BATCH) {
            let mut bx = Vec::with_capacity(BATCH * per_image);
            let mut by = Vec::with_capacity(BATCH);
            for k in 0..BATCH {
                // The order is a permutation, so wrapping only reuses a
                // handful of images to keep the last batch full.
                let i = order[(start + k) % n_train];
                let raw = train.pixels[i * per_image..(i + 1) * per_image].to_vec();
                bx.extend_from_slice(&prep.augmented(raw, &spec)?);
                by.push(train.labels[i]);
            }
            let images = Tensor::<Dyn, B>::from_slice(&bx, batch_shape(&spec, BATCH))?;
            let labels = Tensor::<Dyn, B, i64>::from_slice(&by, vec![BATCH])?;
            let out = model.forward(images)?;
            let loss = out.cross_entropy_loss(&labels)?;
            epoch_loss += loss.to_scalar::<f32>()? as f64;
            steps += 1;
            let grads = loss.backward()?;
            optim.set_lr(schedule.get_lr());
            optim.step(&grads)?;
            schedule.step();
        }
        epoch_loss /= steps as f64;

        // Test pass + gallery refresh. The layer stays in batch-statistics
        // mode here: `forward` never writes BatchNorm2d's running buffers
        // (they arrive as shared references the execution contract does not
        // carry mutations through), so an evaluation-mode pass would
        // normalize by the initial `mean 0 / var 1` and be meaningless. The
        // test batch is 1000 images, so its own statistics are a close
        // stand-in for the population's.
        let test_x = Tensor::<Dyn, B>::from_slice(&test.pixels, batch_shape(&spec, n_test))?;
        let logits = GradMode::Disabled
            .scope(|| model.forward(test_x))
            .and_then(|out| out.to_vec1::<f32>())?;
        let mut correct = 0;
        let mut gallery = Vec::new();
        for i in 0..n_test {
            let mut best_j = 0;
            for j in 1..classes {
                if logits[i * classes + j] > logits[i * classes + best_j] {
                    best_j = j;
                }
            }
            if best_j as i64 == test.labels[i] {
                correct += 1;
            }
            if gallery.len() < 12 {
                // The gallery draws the standardized image, so undo the
                // standardization before handing the page pixels.
                gallery.push((
                    undo_standardize(&test.pixels[i * per_image..(i + 1) * per_image], &spec),
                    best_j,
                    test.labels[i] as usize,
                ));
            }
        }
        let acc = correct as f64 / n_test as f64;
        println!(
            "epoch {epoch}: loss {epoch_loss:.4} acc {acc:.3} lr {:.2e}",
            schedule.get_lr()
        );
        let mut state = state.lock().unwrap();
        state.losses.push(epoch_loss);
        state.accs.push(acc);
        state.gallery = gallery;
        state.status = format!("epoch {epoch}");
    }
    let final_acc = state.lock().unwrap().accs.last().copied().unwrap_or(0.0);
    state.lock().unwrap().done = true;
    state.lock().unwrap().status = "finished".to_string();
    // The point of the run: chance is 1/10. Anything at or below it means
    // the subset is too small to have taught the model anything.
    let chance = 1.0 / classes as f64;
    println!("final test accuracy {final_acc:.3} against {chance:.2} chance on {n_test} images");
    if final_acc <= chance {
        return Err(incin::Error::Msg(format!(
            "final test accuracy {final_acc:.3} did not beat {chance:.2} chance"
        )));
    }
    println!("PASS: browse http://127.0.0.1:{port} for loss, accuracy and predictions");
    println!("Leaving the dashboard up; stop with Ctrl-C.");
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// `(x - mean) / std` again, so the page draws real pixel values.
fn undo_standardize(pixels: &[f32], spec: &Corpus) -> Vec<f32> {
    let plane = spec.side * spec.side;
    pixels
        .iter()
        .enumerate()
        .map(|(k, v)| {
            let c = k / plane;
            v * spec.std[c] + spec.mean[c]
        })
        .collect()
}
