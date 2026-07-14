# Phase 11: Pseudo-3D Model-Graph Visualization

## Phase Goals
A user can explore a run's model graph as an interactive, mouse-driven pseudo-3D structure instead of only the flat text/tree view.

## Architecture & Implementation Choices
1. **Target Architecture**: We will use `ratatui`'s `Canvas` widget for drawing the graph.
2. **Projection**: We'll implement a simple 3D to 2D projection mathematics (e.g. isometric or perspective projection) to map 3D coordinate space to the 2D terminal canvas.
3. **Graph Layout**:
   - `Z-axis`: Network layers / execution sequence depth.
   - `X/Y-plane`: Individual nodes spread out to minimize overlap.
4. **Interactivity**: We will hook into the `PanelMouseEvent::Drag` and `PanelMouseEvent::Scroll*` events already supported by the plugin API to manipulate a virtual "camera" (pan, zoom, rotate).

## Tasks
- [x] Determine a basic graph layout algorithm that maps the `GraphSnapshotEvent` node topological order to 3D `(x, y, z)` coordinates.
- [x] Write 3D to 2D projection math for the camera.
- [x] Implement `ratatui` `Canvas` drawing logic inside `graph.rs` to render nodes as braille dots or shapes, and edges as lines between them.
- [x] Hook up mouse `Drag` events to adjust camera angles (rotation) and translation (pan).
- [x] Hook up mouse `ScrollUp` and `ScrollDown` events to adjust camera zoom.
- [x] Integrate into the main TUI loop and ensure performance remains acceptable (no blocking rendering).
- [x] Ensure the panel remains reachable and dismissible via standard keybindings.

## Verification
- Unit test the 3D-to-2D projection math with known coordinates.
- Attach `kindle-viz` to a running `native_training_demo` with telemetry enabled.
- Verify the graph visually represents the underlying CNN architecture.
- Verify dragging rotates the structure, and scrolling zooms in/out without breaking other panels.
