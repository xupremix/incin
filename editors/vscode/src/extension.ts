// This extension does NOT talk LSP itself and contains no typenum parsing —
// all humanization logic lives in the `kindle-diagnostics` Rust crate, used
// by the `kindle-lsp` proxy binary. All this extension does is point the
// *existing* rust-analyzer extension's server binary at `kindle-lsp` instead
// of rust-analyzer directly (`kindle-lsp` then spawns the real rust-analyzer
// itself) — see docs/growth/02-ide-extensions.md for the full architecture.
import * as vscode from "vscode";

const RA_SECTION = "rust-analyzer";
const KINDLE_SECTION = "kindle";
const HINTS_STATE_KEY = "kindle.shapeHintsEnabled";

async function looksLikeAKindleProject(): Promise<boolean> {
  const manifests = await vscode.workspace.findFiles("**/Cargo.toml", "**/target/**", 25);
  for (const uri of manifests) {
    const bytes = await vscode.workspace.fs.readFile(uri);
    if (Buffer.from(bytes).toString("utf8").includes("kindle")) {
      return true;
    }
  }
  return false;
}

/**
 * Points `rust-analyzer.server.path` at the kindle-lsp proxy and merges
 * KINDLE_LSP_* into `rust-analyzer.server.extraEnv`. The `extraEnv` setting
 * is a best-effort integration point: verify it still exists in whichever
 * rust-analyzer extension version is installed before relying on runtime
 * toggling — if it's ever renamed/removed upstream, `server.path` alone
 * still gets diagnostics + hints working with kindle-lsp's default config.
 */
async function applyKindleLspConfig(hintsEnabled: boolean): Promise<void> {
  const kindleConfig = vscode.workspace.getConfiguration(KINDLE_SECTION);
  const raConfig = vscode.workspace.getConfiguration(RA_SECTION);

  const lspPath = kindleConfig.get<string>("lspPath", "kindle-lsp");
  const shortenBackend = kindleConfig.get<boolean>("shortenBackend", false);

  await raConfig.update("server.path", lspPath, vscode.ConfigurationTarget.Workspace);

  const existingEnv = raConfig.get<Record<string, string>>("server.extraEnv", {});
  const mergedEnv = {
    ...existingEnv,
    KINDLE_LSP_HINTS: hintsEnabled ? "1" : "0",
    KINDLE_LSP_SHORTEN_BACKEND: shortenBackend ? "1" : "0",
  };
  await raConfig.update("server.extraEnv", mergedEnv, vscode.ConfigurationTarget.Workspace);
}

async function restartRustAnalyzer(): Promise<void> {
  try {
    await vscode.commands.executeCommand("rust-analyzer.restartServer");
  } catch {
    vscode.window.showInformationMessage(
      "Kindle: settings updated — reload the window for them to take effect."
    );
  }
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  if (!(await looksLikeAKindleProject())) {
    return;
  }

  const hintsEnabled = context.workspaceState.get<boolean>(HINTS_STATE_KEY, true);
  await applyKindleLspConfig(hintsEnabled);

  context.subscriptions.push(
    vscode.commands.registerCommand("kindle.toggleShapeHints", async () => {
      const current = context.workspaceState.get<boolean>(HINTS_STATE_KEY, true);
      const next = !current;
      await context.workspaceState.update(HINTS_STATE_KEY, next);
      await applyKindleLspConfig(next);
      await restartRustAnalyzer();
      vscode.window.showInformationMessage(`Kindle: shape hints ${next ? "enabled" : "disabled"}.`);
    })
  );
}

export function deactivate(): void {
  // Deliberately a no-op: the settings this extension writes are workspace
  // configuration, not process state, so nothing needs tearing down here.
}
