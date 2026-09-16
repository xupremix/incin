from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/hardware.yml"


class HardwareResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        match = re.search(
            r"        id: resolve\n.*?        run: \|\n((?: {10}[^\n]*\n|\n)+)",
            text,
            re.DOTALL,
        )
        if match is None:
            raise AssertionError("hardware resolver script not found")
        cls.script = "\n".join(line[10:] for line in match[1].splitlines())

    def resolve(self, requested: str, runner: str | None):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            env = {
                "PATH": os.environ["PATH"],
                "REQUESTED": requested,
                "GITHUB_OUTPUT": str(output),
                "WGPU_RUNNER": "",
                "METAL_RUNNER": "",
            }
            if runner is not None:
                env["CUDA_RUNNER"] = runner
            result = subprocess.run(
                ["bash", "-c", self.script],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            outputs = dict(
                line.split("=", 1)
                for line in output.read_text().splitlines()
            ) if output.exists() else {}
            return result, outputs

    def test_explicit_cuda_suites_require_runner(self) -> None:
        for requested in ("cuda", "dist2-network", "multinode"):
            for runner in (None, ""):
                with self.subTest(requested=requested, runner=runner):
                    result, outputs = self.resolve(requested, runner)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn(f"::error::job={requested} was requested", result.stdout)
                    self.assertIn("HARDWARE_CUDA_RUNNER variable is unset", result.stdout)
                    self.assertEqual(outputs, {})

    def test_default_skips_cuda_suites_without_runner(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("REQUESTED: ${{ github.event.inputs.job || 'all' }}", text)
        self.assertIn("default: 'all'", text)
        for runner in (None, ""):
            with self.subTest(runner=runner):
                result, outputs = self.resolve("all", runner)
                self.assertEqual(result.returncode, 0, result.stderr)
                for key in ("cuda", "dist2_network", "multinode", "wgpu_native"):
                    self.assertEqual(outputs[key], "false")
                for key in ("wgpu_software", "metal"):
                    self.assertEqual(outputs[key], "true")
                self.assertIn("::warning::CUDA collective jobs skipped", result.stdout)
                self.assertNotIn("::error::", result.stdout)

    def test_configured_runner_preserves_selection(self) -> None:
        runner = '["self-hosted","linux","cuda"]'
        for requested in ("all", "cuda", "dist2-network", "multinode", "wgpu", "metal"):
            with self.subTest(requested=requested):
                result, outputs = self.resolve(requested, runner)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(outputs["cuda_runner"], runner)
                for key in ("cuda", "dist2_network", "multinode"):
                    expected = requested in ("all", key.replace("_", "-"))
                    self.assertEqual(outputs[key], str(expected).lower())

    def test_other_explicit_suites_do_not_require_cuda_runner(self) -> None:
        for requested, key in (("wgpu", "wgpu_software"), ("metal", "metal")):
            with self.subTest(requested=requested):
                result, outputs = self.resolve(requested, None)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(outputs[key], "true")
                self.assertEqual(outputs["dist2_network"], "false")
                self.assertEqual(outputs["multinode"], "false")

    def test_unknown_suite_fails(self) -> None:
        result, outputs = self.resolve("unknown", None)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("::error::unknown job 'unknown'", result.stdout)
        self.assertEqual(outputs, {})

    def test_resolver_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n"], input=self.script, capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
