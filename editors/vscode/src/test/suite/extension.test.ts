import * as assert from "assert";
import * as vscode from "vscode";

suite("Kindle VS Code extension", () => {
  test("activates in a workspace with a kindle Cargo.toml and configures rust-analyzer", async () => {
    const ext = vscode.extensions.getExtension("kindle.kindle-lsp-vscode");
    assert.ok(ext, "extension kindle.kindle-lsp-vscode was not found in this dev host");

    await ext!.activate();
    assert.strictEqual(ext!.isActive, true, "extension did not report itself as active");

    const raConfig = vscode.workspace.getConfiguration("rust-analyzer");
    assert.strictEqual(
      raConfig.get("server.path"),
      "kindle-lsp",
      "extension did not set rust-analyzer.server.path to the configured kindle-lsp path"
    );

    const env = raConfig.get<Record<string, string>>("server.extraEnv");
    assert.strictEqual(env?.KINDLE_LSP_HINTS, "1", "hints env var not set to enabled by default");
    assert.strictEqual(
      env?.KINDLE_LSP_SHORTEN_BACKEND,
      "0",
      "shorten-backend env var not set to disabled by default"
    );
  });

  test("Kindle: Toggle Shape Hints command flips the hints env var", async () => {
    // The toggle command also asks the real rust-analyzer extension to
    // restart its server, which means actually spawning whatever
    // `kindle.lspPath` resolves to. Point it at a real, always-present,
    // instantly-exiting binary so that restart attempt doesn't fail with an
    // unrelated ENOENT — this test is only about the config value flipping,
    // not about kindle-lsp's own behavior.
    await vscode.workspace
      .getConfiguration("kindle")
      .update("lspPath", "/bin/true", vscode.ConfigurationTarget.Workspace);

    const before = vscode.workspace
      .getConfiguration("rust-analyzer")
      .get<Record<string, string>>("server.extraEnv");
    assert.strictEqual(before?.KINDLE_LSP_HINTS, "1", "expected hints enabled before toggling");

    await vscode.commands.executeCommand("kindle.toggleShapeHints");

    const after = vscode.workspace
      .getConfiguration("rust-analyzer")
      .get<Record<string, string>>("server.extraEnv");
    assert.strictEqual(after?.KINDLE_LSP_HINTS, "0", "toggle command did not flip the hints env var to disabled");
  });
});
