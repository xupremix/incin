extern crate alloc;
use alloc::collections::BTreeMap;
use incin::backend_authoring::SupportsDType;
use incin::prelude::*;
use incin::{Linear, Module};
use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::{TracingBackend, extract_graph, tracing_mark_input, tracing_mark_output};

/// Nb.
type NB = CpuBackendImpl;
/// Tb.
type TB = TracingBackend<NB>;

#[module]
/// Simple mlp.
pub struct SimpleMlp<B: Backend> {
    /// Fc1.
    pub fc1: Linear<Dyn, B>,
    /// Fc2.
    pub fc2: Linear<Dyn, B>,
    /// Fc3.
    pub fc3: Linear<Dyn, B>,
}

impl<B: Backend> SimpleMlp<B>
where
    B: SupportsDType<B::FloatElem>,
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    /// New.
    pub fn new() -> Result<Self> {
        Ok(Self {
            fc1: Linear::<Dyn, B>::build((10, 32))?,
            fc2: Linear::<Dyn, B>::build((32, 16))?,
            fc3: Linear::<Dyn, B>::build((16, 2))?,
        })
    }

    /// Forward.
    pub fn forward(&self, x: Tensor<Dyn, B>) -> Result<Tensor<Dyn, B>> {
        let x = self.fc1.forward(x)?.relu()?;
        let x = self.fc2.forward(x)?.relu()?;
        self.fc3.forward(x)
    }
}

/// Project.
fn project(
    x: f64,
    y: f64,
    z: f64,
    yaw: f64,
    pitch: f64,
    zoom: f64,
    pan_x: f64,
    pan_y: f64,
) -> (f64, f64) {
    let x1 = x * yaw.cos() - z * yaw.sin();
    let z1 = x * yaw.sin() + z * yaw.cos();

    let y2 = y * pitch.cos() - z1 * pitch.sin();
    let _z2 = y * pitch.sin() + z1 * pitch.cos();

    ((x1 * zoom) + pan_x, (y2 * zoom) + pan_y)
}

fn main() -> anyhow::Result<()> {
    let model = SimpleMlp::<TB>::new()?;
    let input = Tensor::<Dyn, TB>::zeros([1, 10])?;
    tracing_mark_input(input.inner().value_id);
    let output = model.forward(input)?;
    tracing_mark_output(output.inner().value_id);
    let graph = extract_graph();

    println!("Graph has {} nodes.", graph.nodes.len());

    let mut value_depths: BTreeMap<usize, usize> = BTreeMap::new();
    let mut node_depths: BTreeMap<usize, usize> = BTreeMap::new();
    let mut max_depth = 0;

    for node in &graph.nodes {
        let d = node
            .inputs
            .iter()
            .map(|i| value_depths.get(i).copied().unwrap_or(0) + 1)
            .max()
            .unwrap_or(0);
        node_depths.insert(node.id, d);
        for &out in &node.outputs {
            value_depths.insert(out, d);
        }
        if d > max_depth {
            max_depth = d;
        }
    }

    println!("Max depth: {}", max_depth);

    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); max_depth + 1];
    for node in &graph.nodes {
        if let Some(&d) = node_depths.get(&node.id) {
            layers[d].push(node.id);
        }
    }

    let mut positions = BTreeMap::new();
    let spacing_z = 20.0;
    let spacing_x = 20.0;
    let spacing_y = 15.0;

    for (d, layer) in layers.iter().enumerate() {
        let z = (d as f64 - max_depth as f64 / 2.0) * spacing_z;
        let n = layer.len();
        for (i, &id) in layer.iter().enumerate() {
            let x = (i as f64 - (n as f64 - 1.0) / 2.0) * spacing_x;
            let y = if i % 2 == 0 {
                spacing_y / 2.0
            } else {
                -spacing_y / 2.0
            };
            positions.insert(id, (x, y, z));
        }
    }

    let pitch = std::f64::consts::PI / 6.0;
    let yaw = -std::f64::consts::PI / 4.0;
    let zoom = 1.0;
    let pan_x = 0.0;
    let pan_y = 0.0;

    let mut points_inside = 0;
    for node in &graph.nodes {
        if let Some(&(x, y, z)) = positions.get(&node.id) {
            let (px, py) = project(x, y, z, yaw, pitch, zoom, pan_x, pan_y);
            println!(
                "Node {} ({}): x={:.1}, y={:.1}, z={:.1} -> px={:.1}, py={:.1}",
                node.id,
                node.op.as_str(),
                x,
                y,
                z,
                px,
                py
            );
            if (-100.0..=100.0).contains(&px) && (-100.0..=100.0).contains(&py) {
                points_inside += 1;
            }
        }
    }
    println!("Points inside canvas [-100, 100]: {}", points_inside);

    Ok(())
}
