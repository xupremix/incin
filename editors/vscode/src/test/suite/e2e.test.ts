import * as assert from "assert";
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

const enabled = process.env.INCIN_REAL_E2E === "1";
const timeoutMs = Number(process.env.INCIN_E2E_TIMEOUT_MS || "120000");

function labelText(label: string | vscode.InlayHintLabelPart[]): string {
  return typeof label === "string" ? label : label.map((part) => part.value).join("");
}

async function eventually<T>(description: string, probe: () => Promise<T | undefined>): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await probe();
    if (value !== undefined) return value;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`${description} did not complete within ${timeoutMs}ms`);
}

suite("Incin VS Code real LSP pipeline", () => {
  test("humanizes diagnostics and full inlay labels while completion remains available", async function () {
    if (!enabled) this.skip();
    this.timeout(timeoutMs + 30_000);

    const lspPath = process.env.INCIN_E2E_LSP_PATH;
    const workspace = process.env.INCIN_E2E_WORKSPACE;
    assert.ok(lspPath && fs.existsSync(lspPath), "INCIN_E2E_LSP_PATH must name a built incin-lsp binary");
    assert.ok(workspace && fs.existsSync(path.join(workspace, "Cargo.toml")), "test runner did not create an Incin E2E workspace");
    const file = path.join(workspace, "src", "main.rs");

    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(file));
    await vscode.window.showTextDocument(document);
    const extension = vscode.extensions.getExtension("incin.incin-lsp-vscode");
    assert.ok(extension, "Incin extension is not installed in the development host");
    await extension.activate();

    const diagnostics = await eventually("humanized rust-analyzer diagnostic", async () => {
      const found = vscode.languages.getDiagnostics(document.uri).find((diagnostic) =>
        diagnostic.message.includes("Cannot reshape: source has 6 elements but the target shape has 8 elements")
      );
      return found;
    });
    assert.ok(!diagnostics.message.includes("UInt<"), "diagnostic leaked raw typenum text");

    const hints = await eventually("humanized tensor inlay hint", async () => {
      const result = await vscode.commands.executeCommand<vscode.InlayHint[]>(
        "vscode.executeInlayHintProvider",
        document.uri,
        new vscode.Range(new vscode.Position(0, 0), new vscode.Position(document.lineCount - 1, 0))
      );
      return result?.find((hint) => labelText(hint.label).includes("Tensor<[2, 3]"));
    });
    assert.ok(!labelText(hints.label).includes("UInt<"), "inlay hint leaked raw typenum text");

    const completions = await vscode.commands.executeCommand<vscode.CompletionList>(
      "vscode.executeCompletionItemProvider",
      document.uri,
      new vscode.Position(6, 24),
      "."
    );
    assert.ok(completions && completions.items.length > 0, "completion request did not pass through the proxy");
  });
});
