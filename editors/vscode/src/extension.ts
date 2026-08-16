// This extension does NOT talk LSP itself and contains no typenum parsing;
// all humanization logic lives in the `incin-diagnostics` Rust crate, used
// by the `incin-lsp` proxy binary. All this extension does is point the
// *existing* rust-analyzer extension's server binary at `incin-lsp` instead
// of rust-analyzer directly (`incin-lsp` then spawns the real rust-analyzer
// itself). See docs/growth/02-ide-extensions.md for the full architecture.
import * as vscode from "vscode";

const RA_SECTION = "rust-analyzer";
const INCIN_SECTION = "incin";
const HINTS_STATE_KEY = "incin.shapeHintsEnabled";

async function looksLikeAIncinProject(): Promise<boolean> {
  const manifests = await vscode.workspace.findFiles("**/Cargo.toml", "**/target/**", 25);
  for (const uri of manifests) {
    const bytes = await vscode.workspace.fs.readFile(uri);
    if (Buffer.from(bytes).toString("utf8").includes("incin")) {
      return true;
    }
  }
  return false;
}

/**
 * Points `rust-analyzer.server.path` at the incin-lsp proxy and merges
 * INCIN_LSP_* into `rust-analyzer.server.extraEnv`. The `extraEnv` setting
 * is a best-effort integration point: verify it still exists in whichever
 * rust-analyzer extension version is installed before relying on runtime
 * toggling. If it's ever renamed/removed upstream, `server.path` alone
 * still gets diagnostics + hints working with incin-lsp's default config.
 */
async function applyIncinLspConfig(hintsEnabled: boolean): Promise<void> {
  const incinConfig = vscode.workspace.getConfiguration(INCIN_SECTION);
  const raConfig = vscode.workspace.getConfiguration(RA_SECTION);

  const lspPath = incinConfig.get<string>("lspPath", "incin-lsp");
  const shortenBackend = incinConfig.get<boolean>("shortenBackend", false);

  await raConfig.update("server.path", lspPath, vscode.ConfigurationTarget.Workspace);

  const existingEnv = raConfig.get<Record<string, string>>("server.extraEnv", {});
  const mergedEnv = {
    ...existingEnv,
    INCIN_LSP_HINTS: hintsEnabled ? "1" : "0",
    INCIN_LSP_SHORTEN_BACKEND: shortenBackend ? "1" : "0",
  };
  await raConfig.update("server.extraEnv", mergedEnv, vscode.ConfigurationTarget.Workspace);
}

async function restartRustAnalyzer(): Promise<void> {
  try {
    await vscode.commands.executeCommand("rust-analyzer.restartServer");
  } catch {
    vscode.window.showInformationMessage(
      "Incin: settings updated; reload the window for them to take effect."
    );
  }
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  if (!(await looksLikeAIncinProject())) {
    return;
  }

  const hintsEnabled = context.workspaceState.get<boolean>(HINTS_STATE_KEY, true);
  await applyIncinLspConfig(hintsEnabled);

  context.subscriptions.push(
    vscode.commands.registerCommand("incin.toggleShapeHints", async () => {
      const current = context.workspaceState.get<boolean>(HINTS_STATE_KEY, true);
      const next = !current;
      await context.workspaceState.update(HINTS_STATE_KEY, next);
      await applyIncinLspConfig(next);
      await restartRustAnalyzer();
      vscode.window.showInformationMessage(`Incin: shape hints ${next ? "enabled" : "disabled"}.`);
    })
  );
}

export function deactivate(): void {
  // Deliberately a no-op: the settings this extension writes are workspace
  // configuration, not process state, so nothing needs tearing down here.
}
