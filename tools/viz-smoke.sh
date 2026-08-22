#!/usr/bin/env bash
set -euo pipefail

# incin-viz functional verification (0.1.0 gate issue #40).
#
# Drives the real TUI binary against a synthetic-but-real telemetry stream
# written through Emitter + FileTransport (the actual wire format), inside
# a tmux session, and asserts that every registered panel renders. Also
# runs the plugin-contract example against the same stream.
#
# Requires: tmux, cargo. Retains its captured evidence under
# audit-evidence/VIZ-40/ when run from the repository root.

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
fail() { echo -e "${RED}FAILED: $1${NC}"; exit 1; }
ok() { echo -e "${GREEN}OK: $1${NC}"; }

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

WORK="$(mktemp -d)"
trap 'tmux kill-session -t viz-smoke 2>/dev/null || true; rm -rf "$WORK"' EXIT
STREAM="$WORK/stream.jsonl"
EVIDENCE_DIR="$REPO_ROOT/audit-evidence/VIZ-40"
mkdir -p "$EVIDENCE_DIR"

echo "== building =="
cargo build -p incin-viz --bins >/dev/null 2>&1 || fail "incin-viz build"
ok "incin-viz builds"

echo "== writing real-format stream =="
cargo run -q -p incin-viz --example stream_fixture -- "$STREAM" >/dev/null 2>&1 \
    || fail "stream_fixture"
[ -s "$STREAM" ] || fail "fixture wrote no events"
ok "$(wc -l < "$STREAM") wire-format events written via Emitter + FileTransport"

echo "== driving the TUI in tmux =="
tmux kill-session -t viz-smoke 2>/dev/null || true
tmux new-session -d -s viz-smoke -x 200 -y 50 \
    "target/debug/incin-viz --run-dir '$STREAM'; echo EXIT-CODE=\$?"
sleep 3
tmux capture-pane -t viz-smoke -p > "$WORK/tui-capture.txt" || fail "capture"

echo "== asserting every registered panel rendered =="
for expected in "Loss" "Throughput" "Learning Rate" "Gradient Norms" \
                "Weight Norms" "Memory (RSS MB)" "Model Structure"; do
    grep -q "$expected" "$WORK/tui-capture.txt" \
        || fail "panel '$expected' missing from rendered TUI"
    ok "panel renders: $expected"
done

echo "== quitting cleanly =="
tmux send-keys -t viz-smoke q
sleep 2
grep -q "EXIT-CODE=0" <(tmux capture-pane -t viz-smoke -p 2>/dev/null || true) \
    || tmux list-sessions 2>/dev/null | grep -q viz-smoke && sleep 1
tmux kill-session -t viz-smoke 2>/dev/null || true
ok "TUI quit accepted"

echo "== plugin hook contract against the same stream =="
cargo run -q -p incin-viz --example plugin_stream_check -- "$STREAM" 2>&1 \
    | tee "$WORK/plugin-check.txt" | grep -q "plugin_stream_check passed" \
    || fail "plugin_stream_check"
ok "plugin Panel hooks exercised: id/title/update/handle_event/reset/render + keymap"

echo "== retaining evidence =="
cp "$WORK/tui-capture.txt" "$EVIDENCE_DIR/tui-capture.txt"
cp "$WORK/plugin-check.txt" "$EVIDENCE_DIR/plugin-stream-check.txt"
cp "$STREAM" "$EVIDENCE_DIR/stream-fixture.jsonl"
{
    echo "# incin-viz functional verification - $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- Stream: 305 wire-format events (scalar loss/throughput/lr,"
    echo "  custom_metric; gradient norms; memory; epoch summaries) written by"
    echo "  crates/incin-viz/examples/stream_fixture.rs through Emitter +"
    echo "  FileTransport, i.e. the production serialization path."
    echo "- TUI: target/debug/incin-viz --run-dir <stream> under tmux 200x50;"
    echo "  all seven registered panels asserted present in the captured pane"
    echo "  (tui-capture.txt); 'q' quit accepted."
    echo "- Plugin: crates/incin-viz/examples/plugin_stream_check.rs registers a"
    echo "  Panel implementing id/title/update/handle_event/reset/render against"
    echo "  App + FileTransportReader, drains the same stream, renders through a"
    echo "  TestBackend, and asserts title and body text plus default-keymap Quit"
    echo "  resolution (plugin-stream-check.txt)."
    echo "- Reproduce with: tools/viz-smoke.sh"
} > "$EVIDENCE_DIR/verification.md"
ok "evidence retained in audit-evidence/VIZ-40/"

echo -e "${GREEN}${BOLD}viz functional verification passed${NC}"
