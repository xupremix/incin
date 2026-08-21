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
const RA_PATH_ENV = "INCIN_LSP_RA_PATH";

async function manifestMentionsIncin(uri: vscode.Uri): Promise<boolean> {
  try {
    const bytes = await vscode.workspace.fs.readFile(uri);
    return Buffer.from(bytes).toString("utf8").includes("incin");
  } catch {
    return false;
  }
}

async function looksLikeAnIncinProject(): Promise<boolean> {
  // Most Cargo workspaces have a root manifest. Reading it directly avoids a
  // recursive workspace search (and a noticeable activation delay) on the
  // common path while still supporting nested manifests below.
  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    if (await manifestMentionsIncin(vscode.Uri.joinPath(folder.uri, "Cargo.toml"))) {
      return true;
    }
  }

  const manifests = await vscode.workspace.findFiles("**/Cargo.toml", "**/target/**", 25);
  for (const uri of manifests) {
    if (await manifestMentionsIncin(uri)) {
      return true;
    }
  }
  return false;
}

async function bundledRustAnalyzerPath(): Promise<string | undefined> {
  const extension = vscode.extensions.getExtension("rust-lang.rust-analyzer");
  if (!extension) {
    return undefined;
  }

  const executable = process.platform === "win32" ? "rust-analyzer.exe" : "rust-analyzer";
  const uri = vscode.Uri.joinPath(extension.extensionUri, "server", executable);
  try {
    await vscode.workspace.fs.stat(uri);
    return uri.fsPath;
  } catch {
    return undefined;
  }
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
  const mergedEnv: Record<string, string> = {
    ...existingEnv,
    INCIN_LSP_HINTS: hintsEnabled ? "1" : "0",
    INCIN_LSP_SHORTEN_BACKEND: shortenBackend ? "1" : "0",
  };
  if (!mergedEnv[RA_PATH_ENV]) {
    const bundledPath = await bundledRustAnalyzerPath();
    if (bundledPath) {
      mergedEnv[RA_PATH_ENV] = bundledPath;
    }
  }
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
  if (!(await looksLikeAnIncinProject())) {
    return;
  }

  const hintsEnabled = context.workspaceState.get<boolean>(HINTS_STATE_KEY, true);
  await applyIncinLspConfig(hintsEnabled);
  // `rust-analyzer` is an extension dependency, so VS Code activates it before
  // this extension can replace its server path. Restart it once after writing
  // the workspace settings so the first session also runs through incin-lsp.
  await restartRustAnalyzer();

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
