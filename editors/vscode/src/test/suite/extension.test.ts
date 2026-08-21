import * as assert from "assert";
import * as vscode from "vscode";

suite("Incin VS Code extension", () => {
  test("activates in a workspace with a incin Cargo.toml and configures rust-analyzer", async () => {
    const ext = vscode.extensions.getExtension("incin.incin-lsp-vscode");
    assert.ok(ext, "extension incin.incin-lsp-vscode was not found in this dev host");

    await ext!.activate();
    assert.strictEqual(ext!.isActive, true, "extension did not report itself as active");

    const raConfig = vscode.workspace.getConfiguration("rust-analyzer");
    assert.strictEqual(
      raConfig.get("server.path"),
      "incin-lsp",
      "extension did not set rust-analyzer.server.path to the configured incin-lsp path"
    );

    const env = raConfig.get<Record<string, string>>("server.extraEnv");
    assert.strictEqual(env?.INCIN_LSP_HINTS, "1", "hints env var not set to enabled by default");
    assert.strictEqual(
      env?.INCIN_LSP_SHORTEN_BACKEND,
      "0",
      "shorten-backend env var not set to disabled by default"
    );
    assert.ok(
      env?.INCIN_LSP_RA_PATH,
      "extension did not point the proxy at rust-analyzer's bundled server"
    );
  });

  test("Incin: Toggle Shape Hints command flips the hints env var", async () => {
    // The toggle command also asks the real rust-analyzer extension to
    // restart its server, which means actually spawning whatever
    // `incin.lspPath` resolves to. Point it at a real, always-present,
    // instantly-exiting binary so that restart attempt doesn't fail with an
    // unrelated ENOENT; this test is only about the config value flipping,
    // not about incin-lsp's own behavior.
    await vscode.workspace
      .getConfiguration("incin")
      .update("lspPath", "/bin/true", vscode.ConfigurationTarget.Workspace);

    const before = vscode.workspace
      .getConfiguration("rust-analyzer")
      .get<Record<string, string>>("server.extraEnv");
    assert.strictEqual(before?.INCIN_LSP_HINTS, "1", "expected hints enabled before toggling");

    await vscode.commands.executeCommand("incin.toggleShapeHints");

    const after = vscode.workspace
      .getConfiguration("rust-analyzer")
      .get<Record<string, string>>("server.extraEnv");
    assert.strictEqual(after?.INCIN_LSP_HINTS, "0", "toggle command did not flip the hints env var to disabled");
  });
});
