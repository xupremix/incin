// Real, automated smoke test: launches an actual VS Code Extension
// Development Host (under whatever display/Xvfb is available) with this
// extension loaded from source, opens a throwaway workspace containing a
// Cargo.toml that mentions "incin", and runs src/test/suite/extension.test.ts
// inside it. The default run exercises activation and settings rewriting. CI
// additionally enables the real pipeline suite, which checks a complete VS
// Code -> incin-lsp -> rust-analyzer exchange for diagnostics, inlay hints,
// and completions.
import * as path from "path";
import * as fs from "fs";
import * as os from "os";
import { spawnSync } from "child_process";
import {
  runTests,
  downloadAndUnzipVSCode,
  resolveCliArgsFromVSCodeExecutablePath,
} from "@vscode/test-electron";

const VSCODE_VERSION = "1.134.0";
const RUST_ANALYZER_EXTENSION = "rust-lang.rust-analyzer@0.3.2971";

async function main(): Promise<void> {
  const extensionDevelopmentPath = path.resolve(__dirname, "../../");
  const extensionTestsPath = path.resolve(__dirname, "./suite/index");

  const workspaceDir = fs.mkdtempSync(path.join(os.tmpdir(), "incin-vscode-ws-"));
  const userDataDir = path.join(workspaceDir, ".vscode-user-data");
  process.once("exit", () => fs.rmSync(workspaceDir, { recursive: true, force: true }));
  fs.writeFileSync(
    path.join(workspaceDir, "Cargo.toml"),
    '[package]\nname = "incin-test-fixture"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\nincin = "0.1"\n'
  );
  if (process.env.INCIN_REAL_E2E === "1") {
    const repoRoot = process.env.INCIN_E2E_REPO_ROOT || path.resolve(extensionDevelopmentPath, "../..");
    const lspPath = process.env.INCIN_E2E_LSP_PATH || path.join(repoRoot, "target/debug/incin-lsp");
    fs.mkdirSync(path.join(workspaceDir, "src"));
    fs.mkdirSync(path.join(workspaceDir, ".vscode"));
    fs.writeFileSync(
      path.join(workspaceDir, "Cargo.toml"),
      `[package]\nname = "incin-vscode-e2e"\nversion = "0.1.0"\nedition = "2024"\n\n[dependencies]\nincin = { path = ${JSON.stringify(path.join(repoRoot, "crates", "incin"))} }\n`
    );
    fs.writeFileSync(path.join(workspaceDir, ".vscode", "settings.json"), JSON.stringify({ "incin.lspPath": lspPath }));
    fs.writeFileSync(
      path.join(workspaceDir, "src", "main.rs"),
      "use incin::prelude::*;\n\nfn main() -> Result<()> {\n    let image: Tensor<s![2, 3], DefaultBackend> = Tensor::zeros(())?;\n    let doubled = &image + &image;\n    let _ = doubled;\n    let _invalid = image.reshape(shape![2, 4])?;\n    Ok(())\n}\n"
    );
    process.env.INCIN_E2E_WORKSPACE = workspaceDir;
  }

  // Deliberately does NOT point at the system's snap-packaged `code`: snap's
  // sandboxing was confirmed (empirically, in this same investigation) to
  // silently swallow `--extensionTestsPath`; the process exits 0 without
  // ever invoking this module's `run()`. Letting test-electron download and
  // use its own unconfined VS Code build is the standard, documented, and
  // (here) actually-working path.
  const vscodeExecutablePath = process.env.INCIN_TEST_VSCODE_PATH ||
    (await downloadAndUnzipVSCode(VSCODE_VERSION));

  // package.json declares `rust-lang.rust-analyzer` as an extensionDependency
  // (deliberately; see docs/growth/02-ide-extensions.md's 2026-07-23
  // follow-up), so this extension flatly refuses to activate without it, even
  // in the test/dev host. The fresh profile test-electron manages has no
  // extensions at all, so it must be installed before `runTests()`.
  const [cliPath, ...resolvedCliArgs] = resolveCliArgsFromVSCodeExecutablePath(vscodeExecutablePath);
  const cliArgs = resolvedCliArgs.filter((arg) => !arg.startsWith("--user-data-dir="));
  const rustAnalyzerExtension = process.env.INCIN_TEST_RA_VSIX || RUST_ANALYZER_EXTENSION;
  const install = spawnSync(cliPath, [...cliArgs, `--user-data-dir=${userDataDir}`, "--install-extension", rustAnalyzerExtension], {
    encoding: "utf-8",
    stdio: "inherit",
  });
  if (install.status !== 0) {
    throw new Error(`Failed to install rust-lang.rust-analyzer into the test profile (exit ${install.status})`);
  }

  const exitCode = await runTests({
    vscodeExecutablePath,
    extensionDevelopmentPath,
    extensionTestsPath,
    extensionTestsEnv: {
      INCIN_REAL_E2E: process.env.INCIN_REAL_E2E,
      INCIN_E2E_LSP_PATH: process.env.INCIN_E2E_LSP_PATH,
      INCIN_E2E_REPO_ROOT: process.env.INCIN_E2E_REPO_ROOT,
      INCIN_E2E_WORKSPACE: process.env.INCIN_E2E_WORKSPACE,
      INCIN_E2E_TIMEOUT_MS: process.env.INCIN_E2E_TIMEOUT_MS || "120000",
    },
    launchArgs: [workspaceDir, `--user-data-dir=${userDataDir}`, "--disable-workspace-trust"],
  });
  console.log(`runTests() resolved with exit code ${exitCode}`);
  if (exitCode !== 0) {
    process.exitCode = exitCode;
  }
}

main().catch((err) => {
  console.error("Failed to run VS Code extension tests:", err);
  process.exit(1);
});
