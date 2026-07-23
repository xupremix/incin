// Real, automated smoke test: launches an actual VS Code Extension
// Development Host (under whatever display/Xvfb is available) with this
// extension loaded from source, opens a throwaway workspace containing a
// Cargo.toml that mentions "kindle", and runs src/test/suite/extension.test.ts
// inside it. This exercises the extension's own activation and
// settings-rewriting logic for real — it does not spin up rust-analyzer or
// kindle-lsp themselves, so it cannot prove the end-to-end humanized-
// diagnostic pipeline; see docs/growth/02-ide-extensions.md for what that
// would additionally require.
import * as path from "path";
import * as fs from "fs";
import * as os from "os";
import { spawnSync } from "child_process";
import {
  runTests,
  downloadAndUnzipVSCode,
  resolveCliArgsFromVSCodeExecutablePath,
} from "@vscode/test-electron";

async function main(): Promise<void> {
  const extensionDevelopmentPath = path.resolve(__dirname, "../../");
  const extensionTestsPath = path.resolve(__dirname, "./suite/index");

  const workspaceDir = fs.mkdtempSync(path.join(os.tmpdir(), "kindle-vscode-ws-"));
  fs.writeFileSync(
    path.join(workspaceDir, "Cargo.toml"),
    '[package]\nname = "kindle-test-fixture"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\nkindle = "0.2"\n'
  );

  // Deliberately does NOT point at the system's snap-packaged `code`: snap's
  // sandboxing was confirmed (empirically, in this same investigation) to
  // silently swallow `--extensionTestsPath` — the process exits 0 without
  // ever invoking this module's `run()`. Letting test-electron download and
  // use its own unconfined VS Code build is the standard, documented, and
  // (here) actually-working path.
  const vscodeExecutablePath =
    process.env.KINDLE_TEST_VSCODE_PATH || (await downloadAndUnzipVSCode());

  // package.json declares `rust-lang.rust-analyzer` as an extensionDependency
  // (deliberately — see docs/growth/02-ide-extensions.md's 2026-07-23
  // follow-up), so this extension flatly refuses to activate without it, even
  // in the test/dev host. The fresh profile test-electron manages has no
  // extensions at all, so it must be installed before `runTests()`.
  const [cliPath, ...cliArgs] = resolveCliArgsFromVSCodeExecutablePath(vscodeExecutablePath);
  const install = spawnSync(cliPath, [...cliArgs, "--install-extension", "rust-lang.rust-analyzer"], {
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
    launchArgs: [workspaceDir, "--disable-workspace-trust"],
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
