from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/hardware.yml"

HARDWARE_JOBS = {
    "cuda": "cuda",
    "wgpu-software": "wgpu_software",
    "wgpu-native": "wgpu_native",
    "metal": "metal",
    "dist2-network": "dist2_network",
    "multinode": "multinode",
}


def extract_step_script(step_id: str) -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        rf"        id: {re.escape(step_id)}\n.*?        run: \|\n((?: {{10}}[^\n]*\n|\n)+)",
        text,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"step '{step_id}' script not found")
    return "\n".join(line[10:] for line in match.group(1).splitlines())


class HardwareResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = extract_step_script("resolve")
        cls.conclusion_script = extract_step_script("conclude")

    def resolve(self, requested: str, runner: str | None, wgpu_runner: str | None = None):
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
            if wgpu_runner is not None:
                env["WGPU_RUNNER"] = wgpu_runner
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

    def conclude(
        self,
        select_result: str,
        coverage_skipped: str,
        skipped_jobs: str = "",
        cuda_state: str = "unset",
        wgpu_state: str = "unset",
    ):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary"
            env = {
                "PATH": os.environ["PATH"],
                "SELECT_RESULT": select_result,
                "COVERAGE_SKIPPED": coverage_skipped,
                "SKIPPED_JOBS": skipped_jobs,
                "CUDA_RUNNER_STATE": cuda_state,
                "WGPU_RUNNER_STATE": wgpu_state,
                "GITHUB_STEP_SUMMARY": str(summary),
            }
            result = subprocess.run(
                ["bash", "-c", self.conclusion_script],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            summary_text = summary.read_text() if summary.exists() else ""
            return result, summary_text

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
                self.assertEqual(outputs["coverage_skipped"], "true")
                self.assertEqual(
                    outputs["skipped_jobs"],
                    "cuda, dist2-network, multinode, wgpu-native",
                )
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

    def test_coverage_skipped_tracks_unset_runner_variables(self) -> None:
        cuda_runner = '["self-hosted","linux","cuda"]'
        wgpu_runner = '["self-hosted","linux","gpu"]'
        cases = [
            # requested, cuda runner, wgpu runner, coverage_skipped, skipped_jobs
            ("all", None, None, "true", "cuda, dist2-network, multinode, wgpu-native"),
            ("all", "", "", "true", "cuda, dist2-network, multinode, wgpu-native"),
            ("all", cuda_runner, None, "true", "wgpu-native"),
            ("all", cuda_runner, wgpu_runner, "false", ""),
            ("dist2-network", cuda_runner, wgpu_runner, "false", ""),
            ("multinode", cuda_runner, wgpu_runner, "false", ""),
            ("metal", None, None, "false", ""),
            ("wgpu", None, None, "true", "wgpu-native"),
            ("wgpu", cuda_runner, wgpu_runner, "false", ""),
        ]
        for requested, runner, wgpu, expected_flag, expected_jobs in cases:
            with self.subTest(
                requested=requested,
                runner=runner,
                wgpu=wgpu,
            ):
                result, outputs = self.resolve(requested, runner, wgpu)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(outputs["coverage_skipped"], expected_flag)
                self.assertEqual(outputs["skipped_jobs"], expected_jobs)

    def test_conclusion_fails_the_run_when_coverage_was_skipped(self) -> None:
        result, summary = self.conclude(
            "success",
            "true",
            "cuda, dist2-network, multinode, wgpu-native",
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("::error title=Hardware coverage skipped::", result.stdout)
        self.assertIn(
            "cuda, dist2-network, multinode, wgpu-native did not run",
            result.stdout,
        )
        self.assertIn("HARDWARE_CUDA_RUNNER=unset", result.stdout)
        self.assertIn("HARDWARE_WGPU_RUNNER=unset", result.stdout)
        self.assertIn(
            "A skipped suite must not read as a pass: "
            "this run fails rather than reporting success.",
            result.stdout,
        )
        self.assertIn("**Coverage conclusion: skipped**", summary)
        self.assertNotIn("complete", summary)

    def test_conclusion_passes_only_when_every_requested_suite_ran(self) -> None:
        result, summary = self.conclude("success", "false")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("::error::", result.stdout)
        self.assertIn("**Coverage conclusion: complete**", summary)

    def test_conclusion_fails_closed_without_a_resolver_decision(self) -> None:
        for select_result, coverage in (("failure", ""), ("success", ""), ("cancelled", "")):
            with self.subTest(select_result=select_result, coverage=coverage):
                result, summary = self.conclude(select_result, coverage)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("::error title=Hardware coverage unknown::", result.stdout)
                self.assertIn("**Coverage conclusion: unknown**", summary)

    def test_resolver_output_drives_the_conclusion(self) -> None:
        cuda_runner = '["self-hosted","linux","cuda"]'
        wgpu_runner = '["self-hosted","linux","gpu"]'
        _, skipped_outputs = self.resolve("all", None)
        result, summary = self.conclude(
            "success",
            skipped_outputs["coverage_skipped"],
            skipped_outputs["skipped_jobs"],
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("::error title=Hardware coverage skipped::", result.stdout)
        self.assertIn(skipped_outputs["skipped_jobs"], result.stdout)
        self.assertIn("**Coverage conclusion: skipped**", summary)

        _, full_outputs = self.resolve("all", cuda_runner, wgpu_runner)
        result, summary = self.conclude(
            "success",
            full_outputs["coverage_skipped"],
            full_outputs["skipped_jobs"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("**Coverage conclusion: complete**", summary)

    def test_resolver_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n"], input=self.script, capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_conclusion_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n"],
            input=self.conclusion_script,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class WorkflowStructureTests(unittest.TestCase):
    """The workflow must route every hardware job through the resolver and
    publish a per-job artifact, so an absent runner label concludes skipped
    (job) and the run cannot conclude success (conclusion job)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        cls.jobs = cls.workflow["jobs"]

    def test_hardware_jobs_are_gated_on_the_resolver(self) -> None:
        for job, output in HARDWARE_JOBS.items():
            with self.subTest(job=job):
                self.assertEqual(
                    self.jobs[job]["if"],
                    f"needs.select.outputs.{output} == 'true'",
                )

    def test_select_exposes_the_coverage_decision(self) -> None:
        outputs = self.jobs["select"]["outputs"]
        self.assertEqual(
            outputs["coverage_skipped"], "${{ steps.resolve.outputs.coverage_skipped }}"
        )
        self.assertEqual(
            outputs["skipped_jobs"], "${{ steps.resolve.outputs.skipped_jobs }}"
        )

    def test_conclusion_job_needs_every_job_and_always_runs(self) -> None:
        conclusion = self.jobs["conclusion"]
        self.assertEqual(conclusion["if"], "always()")
        self.assertEqual(set(conclusion["needs"]), {"select", *HARDWARE_JOBS})
        self.assertEqual(conclusion["runs-on"], "ubuntu-latest")
        step_ids = [step.get("id") for step in conclusion["steps"]]
        self.assertIn("conclude", step_ids)

    def test_each_hardware_job_publishes_its_own_artifact(self) -> None:
        for job in HARDWARE_JOBS:
            with self.subTest(job=job):
                steps = self.jobs[job]["steps"]
                uploads = [
                    step
                    for step in steps
                    if str(step.get("uses", "")).startswith("actions/upload-artifact@")
                ]
                self.assertEqual(len(uploads), 1, steps)
                upload = uploads[0]
                self.assertEqual(upload.get("if"), "always()")
                self.assertEqual(
                    upload["with"]["name"],
                    "hardware-${{ github.job }}-${{ github.run_attempt }}",
                )
                self.assertIn(f"result-{job}.md", upload["with"]["path"])
                recorders = [
                    step
                    for step in steps
                    if step.get("if") == "always()"
                    and f"> result-{job}.md" in str(step.get("run", ""))
                ]
                self.assertEqual(len(recorders), 1, steps)

    def test_hardware_suite_logs_are_captured(self) -> None:
        for job in HARDWARE_JOBS:
            with self.subTest(job=job):
                steps = self.jobs[job]["steps"]
                uploads = [
                    step
                    for step in steps
                    if str(step.get("uses", "")).startswith("actions/upload-artifact@")
                ]
                logs = [
                    name
                    for name in str(uploads[0]["with"]["path"]).splitlines()
                    if name.startswith("hardware") and name.endswith(".log")
                ]
                self.assertTrue(logs, f"{job} uploads no suite log")
                for step in steps:
                    run = str(step.get("run", ""))
                    if "cargo test" in run:
                        self.assertRegex(
                            run,
                            r"tee (?:-a )?hardware",
                            f"{job} does not capture: {step.get('name')}",
                        )


if __name__ == "__main__":
    unittest.main()
