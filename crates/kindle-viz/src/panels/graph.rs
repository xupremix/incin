use alloc::collections::BTreeMap;
use kindle_telemetry::events::{Event, GraphSnapshotEvent};
use kindle_viz_plugin_api::event::{KeyCode, PanelEvent, PanelMouseEvent};
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;
use ratatui::{
    layout::{Alignment, Constraint},
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{
        Cell, Paragraph, Row, Table,
        canvas::{Canvas, Line, Points},
    },
};

#[derive(PartialEq, Eq)]
/// Auto-generated documentation for ViewMode.
pub enum ViewMode {
    /// Auto-generated documentation for List.
    List,
    /// Auto-generated documentation for Canvas3D.
    Canvas3D,
    /// Auto-generated documentation for Canvas2D.
    Canvas2D,
}

/// Auto-generated documentation for GraphModuleListPanel.
pub struct GraphModuleListPanel {
    snapshot: Option<GraphSnapshotEvent>,
    scroll_offset: usize,
    view_mode: ViewMode,

    // 3D Camera State
    pitch: f64,
    yaw: f64,
    zoom: f64,
    pan_x: f64,
    pan_y: f64,

    last_mouse: Option<(u16, u16)>,
}

impl Default for GraphModuleListPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphModuleListPanel {
    /// Auto-generated documentation for new.
    pub fn new() -> Self {
        Self {
            snapshot: None,
            scroll_offset: 0,
            view_mode: ViewMode::Canvas2D,
            pitch: std::f64::consts::PI / 6.0,
            yaw: -std::f64::consts::PI / 4.0,
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            last_mouse: None,
        }
    }

    /// Auto-generated documentation for calculate_3d_layout.
    fn calculate_3d_layout(
        &self,
        snapshot: &GraphSnapshotEvent,
    ) -> BTreeMap<usize, (f64, f64, f64)> {
        let mut value_depths: BTreeMap<usize, usize> = BTreeMap::new();
        let mut node_depths: BTreeMap<usize, usize> = BTreeMap::new();
        let mut max_depth = 0;

        // Compute depth
        for node in &snapshot.graph.nodes {
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

        // Group by depth
        let mut layers: Vec<Vec<usize>> = vec![Vec::new(); max_depth + 1];
        for node in &snapshot.graph.nodes {
            if let Some(&d) = node_depths.get(&node.id) {
                layers[d].push(node.id);
            }
        }

        let mut positions = BTreeMap::new();
        let spacing_z = 30.0;
        let spacing_x = 25.0;

        for (d, layer) in layers.iter().enumerate() {
            let z = (d as f64 - max_depth as f64 / 2.0) * spacing_z;
            let n = layer.len();
            for (i, &id) in layer.iter().enumerate() {
                // Arrange nodes in a slightly staggered circle/ellipse if many, or simple line if few
                let (x, y) = if n == 1 {
                    (0.0, 0.0)
                } else if n < 4 {
                    ((i as f64 - (n as f64 - 1.0) / 2.0) * spacing_x, 0.0)
                } else {
                    let angle = (i as f64 / n as f64) * std::f64::consts::PI * 2.0;
                    let radius = (n as f64).sqrt() * spacing_x / 2.0;
                    (angle.cos() * radius, angle.sin() * radius * 0.5)
                };

                positions.insert(id, (x, y, z));
            }
        }

        positions
    }
}

impl Panel for GraphModuleListPanel {
    /// Auto-generated documentation for id.
    fn id(&self) -> &'static str {
        "graph_modules"
    }

    /// Auto-generated documentation for title.
    fn title(&self) -> &str {
        match self.view_mode {
            ViewMode::List => "Model Structure (List | Press 'v' for 2D)",
            ViewMode::Canvas2D => "Model Structure (2D | Drag: Pan | Scroll: Zoom | 'v' for 3D)",
            ViewMode::Canvas3D => {
                "Model Structure (3D | Drag: Pan | Shift+Drag: Rotate | Scroll: Zoom | 'v' for List)"
            }
        }
    }

    /// Auto-generated documentation for update.
    fn update(&mut self, event: &Event) {
        if let Event::GraphSnapshot(m) = event {
            self.snapshot = Some(m.clone());
        }
    }

    /// Auto-generated documentation for render.
    fn render(&mut self, ctx: &mut RenderCtx<'_, '_>) {
        let area = ctx.area();
        let frame = ctx.frame_mut();

        if self.snapshot.is_none() {
            let placeholder = Paragraph::new("waiting for graph snapshot event…")
                .style(Style::default().add_modifier(Modifier::DIM))
                .alignment(Alignment::Center);
            frame.render_widget(placeholder, area);
            return;
        }

        let snapshot = self.snapshot.as_ref().unwrap();

        if self.view_mode == ViewMode::List {
            let header = Row::new(vec![
                Cell::from("ID").style(Style::default().add_modifier(Modifier::BOLD)),
                Cell::from("Op").style(Style::default().add_modifier(Modifier::BOLD)),
                Cell::from("Inputs").style(Style::default().add_modifier(Modifier::BOLD)),
            ]);

            let rows: Vec<Row> = snapshot
                .graph
                .nodes
                .iter()
                .skip(self.scroll_offset)
                .map(|m| {
                    Row::new(vec![
                        Cell::from(m.id.to_string()),
                        Cell::from(m.op.as_str().to_string()),
                        Cell::from(format!("{:?}", m.inputs)),
                    ])
                })
                .collect();

            let table = Table::new(
                rows,
                [
                    Constraint::Percentage(50),
                    Constraint::Percentage(30),
                    Constraint::Percentage(20),
                ],
            )
            .header(header);

            frame.render_widget(table, area);
        } else {
            let positions = self.calculate_3d_layout(snapshot);

            let pitch = self.pitch;
            let yaw = self.yaw;
            let zoom = self.zoom;
            let pan_x = self.pan_x;
            let pan_y = self.pan_y;

            let project = |x: f64, y: f64, z: f64| -> (f64, f64) {
                if self.view_mode == ViewMode::Canvas2D {
                    // In 2D mode, we ignore z and rotate, using an orthographic mapping.
                    // y should go down as depth increases.
                    // Since z represents depth, we map -z to y.
                    let px = x;
                    let py = -z;
                    ((px * zoom) + pan_x, (py * zoom) + pan_y)
                } else {
                    // Rotate Yaw (around Y axis)
                    let x1 = x * yaw.cos() - z * yaw.sin();
                    let z1 = x * yaw.sin() + z * yaw.cos();

                    // Rotate Pitch (around X axis)
                    let y2 = y * pitch.cos() - z1 * pitch.sin();
                    let _z2 = y * pitch.sin() + z1 * pitch.cos();

                    // Orthographic projection + pan + scale
                    ((x1 * zoom) + pan_x, (y2 * zoom) + pan_y)
                }
            };

            let mut value_to_node: BTreeMap<usize, usize> = BTreeMap::new();
            for node in &snapshot.graph.nodes {
                for &out in &node.outputs {
                    value_to_node.insert(out, node.id);
                }
            }
            let mut max_abs_x = 10.0_f64;
            let mut max_abs_y = 10.0_f64;
            for &(x, y, z) in positions.values() {
                let (px, py) = project(x, y, z);
                if px.abs() > max_abs_x {
                    max_abs_x = px.abs();
                }
                if py.abs() > max_abs_y {
                    max_abs_y = py.abs();
                }
            }
            // Add a 20% margin
            max_abs_x *= 1.2;
            max_abs_y *= 1.2;

            // In 3D mode, keeping a square bound prevents rotation from stretching weirdly.
            // In 2D mode, we want to use the full screen independently.
            if self.view_mode == ViewMode::Canvas3D {
                let max_abs = max_abs_x.max(max_abs_y);
                max_abs_x = max_abs;
                max_abs_y = max_abs;
            }

            let canvas = Canvas::default()
                .marker(ratatui::symbols::Marker::Braille)
                .x_bounds([-max_abs_x, max_abs_x])
                .y_bounds([-max_abs_y, max_abs_y])
                .paint(|ctx| {
                    // Draw edges
                    for node in &snapshot.graph.nodes {
                        if let Some(&(x1, y1, z1)) = positions.get(&node.id) {
                            let (px1, py1) = project(x1, y1, z1);
                            for input in &node.inputs {
                                if let Some(&source_node_id) = value_to_node.get(input)
                                    && let Some(&(x2, y2, z2)) = positions.get(&source_node_id) {
                                        let (px2, py2) = project(x2, y2, z2);
                                        ctx.draw(&Line {
                                            x1: px1,
                                            y1: py1,
                                            x2: px2,
                                            y2: py2,
                                            color: Color::DarkGray,
                                        });
                                    }
                            }
                        }
                    }

                    // Draw nodes
                    for node in &snapshot.graph.nodes {
                        if let Some(&(x, y, z)) = positions.get(&node.id) {
                            let (px, py) = project(x, y, z);
                            let color = match node.op.as_str() {
                                "MatMul" | "Gemm" => Color::Yellow,
                                "Relu" | "Gelu" => Color::Green,
                                "Add" | "Sub" | "Mul" => Color::Blue,
                                "Transpose" | "Reshape" => Color::Magenta,
                                _ => Color::Cyan,
                            };
                            ctx.draw(&Points {
                                coords: &[(px, py)],
                                color,
                            });

                            // Add labels in 2D mode
                            if self.view_mode == ViewMode::Canvas2D {
                                let mut in_shapes = Vec::new();
                                for &in_id in &node.inputs {
                                    if let Some(val) = snapshot.graph.values.get(&in_id) {
                                        in_shapes.push(format!("{:?}", val.shape));
                                    }
                                }
                                let in_str = if in_shapes.is_empty() {
                                    String::new()
                                } else {
                                    in_shapes.join(", ")
                                };

                                let out_shape_str = if let Some(&out_id) = node.outputs.first() {
                                    snapshot
                                        .graph
                                        .values
                                        .get(&out_id)
                                        .map(|v| format!("{:?}", v.shape))
                                        .unwrap_or_default()
                                } else {
                                    String::new()
                                };

                                let op_label = format!("[ {} ]", node.op.as_str());
                                let label_y_offset = max_abs_y * 0.06;
                                let label_x_offset = -(max_abs_x * 0.01 * op_label.len() as f64);

                                if !in_str.is_empty() {
                                    ctx.print(
                                        px + label_x_offset,
                                        py + label_y_offset,
                                        Span::styled(in_str, Style::default().fg(Color::DarkGray)),
                                    );
                                }
                                ctx.print(
                                    px + label_x_offset,
                                    py,
                                    Span::styled(
                                        op_label,
                                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                                    ),
                                );
                                if !out_shape_str.is_empty() {
                                    ctx.print(
                                        px + label_x_offset,
                                        py - label_y_offset,
                                        Span::styled(
                                            out_shape_str,
                                            Style::default().fg(Color::White),
                                        ),
                                    );
                                }
                            }
                        }
                    }
                });

            frame.render_widget(canvas, area);
        }
    }

    /// Auto-generated documentation for handle_event.
    fn handle_event(&mut self, event: &PanelEvent) -> bool {
        match event {
            PanelEvent::Key(k) => match k.code {
                KeyCode::Char('v') => {
                    self.view_mode = match self.view_mode {
                        ViewMode::List => ViewMode::Canvas2D,
                        ViewMode::Canvas2D => ViewMode::Canvas3D,
                        ViewMode::Canvas3D => ViewMode::List,
                    };
                    true
                }
                KeyCode::Up => {
                    if self.view_mode == ViewMode::List {
                        self.scroll_offset = self.scroll_offset.saturating_sub(1);
                        true
                    } else {
                        false
                    }
                }
                KeyCode::Down => {
                    if self.view_mode == ViewMode::List {
                        self.scroll_offset = self.scroll_offset.saturating_add(1);
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            },
            PanelEvent::Mouse(m) => {
                if self.view_mode == ViewMode::List {
                    match m {
                        PanelMouseEvent::ScrollUp { .. } => {
                            self.scroll_offset = self.scroll_offset.saturating_sub(1);
                            true
                        }
                        PanelMouseEvent::ScrollDown { .. } => {
                            self.scroll_offset = self.scroll_offset.saturating_add(1);
                            true
                        }
                        _ => false,
                    }
                } else {
                    match *m {
                        PanelMouseEvent::ScrollUp { .. } => {
                            self.zoom *= 1.1;
                            true
                        }
                        PanelMouseEvent::ScrollDown { .. } => {
                            self.zoom /= 1.1;
                            true
                        }
                        PanelMouseEvent::Down { x, y, .. } => {
                            self.last_mouse = Some((x, y));
                            true
                        }
                        PanelMouseEvent::Drag { x, y, modifiers } => {
                            if let Some((lx, ly)) = self.last_mouse {
                                let dx = (x as f64) - (lx as f64);
                                let dy = (y as f64) - (ly as f64);

                                if modifiers.shift {
                                    // Rotate
                                    self.yaw += dx * 0.02;
                                    self.pitch -= dy * 0.02;
                                } else {
                                    // Pan
                                    self.pan_x += dx * 2.0;
                                    self.pan_y -= dy * 2.0; // Y axis is inverted in ratatui canvas vs screen space? Ratatui canvas Y is up. Screen Y is down.
                                }
                            }
                            self.last_mouse = Some((x, y));
                            true
                        }
                        PanelMouseEvent::Up { .. } => {
                            self.last_mouse = None;
                            true
                        }
                    }
                }
            }
        }
    }

    /// Auto-generated documentation for reset.
    fn reset(&mut self) {
        self.snapshot = None;
        self.scroll_offset = 0;
        self.view_mode = ViewMode::Canvas2D;
        self.pitch = std::f64::consts::PI / 6.0;
        self.yaw = -std::f64::consts::PI / 4.0;
        self.zoom = 1.0;
        self.pan_x = 0.0;
        self.pan_y = 0.0;
        self.last_mouse = None;
    }
}
