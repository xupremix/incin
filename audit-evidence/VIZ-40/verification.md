# incin-viz functional verification - 2026-08-22T23:44:39Z
- Stream: 305 wire-format events (scalar loss/throughput/lr,
  custom_metric; gradient norms; memory; epoch summaries) written by
  crates/incin-viz/examples/stream_fixture.rs through Emitter +
  FileTransport, i.e. the production serialization path.
- TUI: target/debug/incin-viz --run-dir <stream> under tmux 200x50;
  all seven registered panels asserted present in the captured pane
  (tui-capture.txt); 'q' quit accepted.
- Plugin: crates/incin-viz/examples/plugin_stream_check.rs registers a
  Panel implementing id/title/update/handle_event/reset/render against
  App + FileTransportReader, drains the same stream, renders through a
  TestBackend, and asserts title and body text plus default-keymap Quit
  resolution (plugin-stream-check.txt).
- Reproduce with: tools/viz-smoke.sh
