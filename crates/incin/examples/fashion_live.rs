//! Example: live fashion-MNIST dashboard in your browser.
//!
//! Trains a small CNN on 3000 Fashion-MNIST images (a fast subset; the
//! same code trains the full 60k) while serving a dashboard on
//! http://127.0.0.1:8000 — loss curve, accuracy, and a test gallery with
//! predicted vs true labels, auto-refreshed every few seconds. The page
//! is rendered per request from the live training state, so what you see
//! is what the model just did.
//!
//! Run it with `cargo run -p incin --example fashion_live`
//! (downloads ~30MB on first run), then open the URL.

use incin::prelude::*;
use incin_data::vision::fashion_mnist::FashionMnistDataset;
use incin_data::{DataError, Dataset};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

const CLASSES: [&str; 10] = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle",
];

/// What the dashboard renders, updated by the training thread.
struct Snapshot {
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

fn gallery_html(gallery: &[(Vec<f32>, usize, usize)]) -> String {
    let names: Vec<String> = CLASSES.iter().map(|s| format!("\"{s}\"")).collect();
    let mut out = String::from("<div id='gal'></div><script>\nconst NAMES=[");
    out.push_str(&names.join(","));
    out.push_str("];\nconst GAL=[");
    for (pixels, pred, actual) in gallery {
        out.push_str(&format!(
            "{{p:[{}],pred:{},actual:{}}},",
            pixels
                .iter()
                .map(|v| format!("{:.2}", v.clamp(0.0, 1.0)))
                .collect::<Vec<_>>()
                .join(","),
            pred,
            actual
        ));
    }
    out.push_str(
        "];\nconst gal=document.getElementById('gal');\n\
         GAL.forEach((g,i)=>{const d=document.createElement('div');d.style.display='inline-block';d.style.margin='6px';d.style.textAlign='center';\n\
         const c=document.createElement('canvas');c.width=28;c.height=28;c.style.width='84px';c.style.imageRendering='pixelated';\n\
         const x=c.getContext('2d');const im=x.createImageData(28,28);\n\
         for(let k=0;k<784;k++){const v=Math.round(g.p[k]*255);im.data[4*k]=v;im.data[4*k+1]=v;im.data[4*k+2]=v;im.data[4*k+3]=255;}\n\
         x.putImageData(im,0,0);d.appendChild(c);\n\
         const ok=g.pred===g.actual;const l=document.createElement('div');\n\
         l.style.color=ok?'#4f4':'#f44';l.textContent=(ok?'ok ':'WRONG ')+NAMES[g.pred]+'/'+NAMES[g.actual];\n\
         d.appendChild(l);gal.appendChild(d);});\n</script>\n",
    );
    out
}

fn render(state: &Snapshot) -> String {
    let last_loss = state.losses.last().copied().unwrap_or(0.0);
    let last_acc = state.accs.last().copied().unwrap_or(0.0);
    format!(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>\
         <meta http-equiv='refresh' content='5'>\
         <title>fashion-mnist live</title></head>\
         <body style='background:#000;color:#ddd;font-family:monospace'>\
         <h2>fashion-mnist CNN — live</h2>\
         <p>{} (epoch {}/{})</p>\
         <p>loss {last_loss:.4} &nbsp; test-acc {last_acc:.2}</p>\
         <h3>loss</h3>{} <h3>test gallery (pred/true)</h3>{}\
         </body></html>",
        state.status,
        state.losses.len(),
        if state.done { "done" } else { "training" },
        svg_curve(&state.losses, 480, 140),
        gallery_html(&state.gallery),
    )
}

fn serve(listener: TcpListener, state: Arc<Mutex<Snapshot>>) {
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut buf = [0u8; 4096];
        let _ = stream.read(&mut buf);
        let body = {
            let state = state.lock().unwrap();
            render(&state)
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
    let dir = std::path::PathBuf::from("./data/fashion-mnist");
    println!("Loading Fashion-MNIST into {:?}...", dir);
    let train = FashionMnistDataset::new(&dir, true)?;
    let test = FashionMnistDataset::new(&dir, false)?;
    println!("{} train + {} test images.", train.len(), test.len());

    // Fast subset for the demo; swap the takes for the full 60k.
    const N_TRAIN: usize = 3000;
    const N_TEST: usize = 500;
    let mut train_images = Vec::with_capacity(N_TRAIN * 784);
    let mut train_labels: Vec<i64> = Vec::with_capacity(N_TRAIN);
    for i in 0..N_TRAIN {
        let (img, label) = train.get(i).map_err(data_err)?.expect("subset in range");
        train_images.extend_from_slice(&img);
        train_labels.push(label as i64);
    }
    let mut test_images = Vec::with_capacity(N_TEST * 784);
    let mut test_labels: Vec<i64> = Vec::with_capacity(N_TEST);
    for i in (0..test.len()).step_by(test.len() / N_TEST).take(N_TEST) {
        let (img, label) = test.get(i).map_err(data_err)?.expect("subset in range");
        test_images.extend_from_slice(&img);
        test_labels.push(label as i64);
    }

    type B = DefaultBackend;
    let model = seq![
        Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((8, 1))?,
        ReLU,
        MaxPool2d::<typenum::U2, typenum::U2>::new()?,
        Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((16, 8))?,
        ReLU,
        MaxPool2d::<typenum::U2, typenum::U2>::new()?,
        Flatten::new(1isize, -1isize),
        Linear::<Dyn, B>::build((16 * 7 * 7, 64))?,
        ReLU,
        Linear::<Dyn, B>::build((64, 10))?,
    ];
    let mut optim = Adam::<B>::from_module(&model, 1e-3)?;

    let state = Arc::new(Mutex::new(Snapshot {
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
    println!("dashboard: http://127.0.0.1:{port} (auto-refreshes every 5s)");
    let server_state = state.clone();
    std::thread::spawn(move || serve(listener, server_state));

    let batch = 32;
    for epoch in 0..6 {
        let offset = (epoch * 137) % N_TRAIN;
        let mut epoch_loss = 0.0;
        let mut steps = 0;
        for start in (0..N_TRAIN).step_by(batch) {
            let idx: Vec<usize> = (0..batch).map(|k| (offset + start + k) % N_TRAIN).collect();
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
            epoch_loss += loss.to_scalar::<f32>()? as f64;
            steps += 1;
            let grads = loss.backward()?;
            optim.step(&grads)?;
        }
        epoch_loss /= steps as f64;

        // Test pass + gallery refresh.
        let test_x = Tensor::<Dyn, B>::from_slice(&test_images, vec![N_TEST, 1, 28, 28])?;
        let logits = model.forward(test_x)?.to_vec1::<f32>()?;
        let mut correct = 0;
        let mut gallery = Vec::new();
        for i in 0..N_TEST {
            let mut best_j = 0;
            for j in 1..10 {
                if logits[i * 10 + j] > logits[i * 10 + best_j] {
                    best_j = j;
                }
            }
            if best_j as i64 == test_labels[i] {
                correct += 1;
            }
            if gallery.len() < 12 {
                gallery.push((
                    test_images[i * 784..(i + 1) * 784].to_vec(),
                    best_j,
                    test_labels[i] as usize,
                ));
            }
        }
        let acc = correct as f64 / N_TEST as f64;
        println!("epoch {epoch}: loss {epoch_loss:.4} acc {acc:.2}");
        let mut state = state.lock().unwrap();
        state.losses.push(epoch_loss);
        state.accs.push(acc);
        state.gallery = gallery;
        state.status = format!("epoch {epoch}");
    }
    state.lock().unwrap().done = true;
    state.lock().unwrap().status = "finished".to_string();
    println!("PASS: browse http://127.0.0.1:{port} for loss, accuracy and predictions");
    println!("Leaving the dashboard up; stop with Ctrl-C.");
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}
