#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "publish-crates.py"
SPEC = importlib.util.spec_from_file_location("publish_crates", SCRIPT)
assert SPEC and SPEC.loader
publish_crates = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish_crates)


class PublishOrderTests(unittest.TestCase):
    def test_accepts_release_tag(self) -> None:
        self.assertEqual(publish_crates.version_from_tag("v0.1.0"), "0.1.0")
        with self.assertRaisesRegex(ValueError, "tag must"):
            publish_crates.version_from_tag("v0.1.0-")

    def test_accepts_registry_prefix(self) -> None:
        self.assertEqual(publish_crates.prefix_length([True, True, False, False]), 2)

    def test_rejects_registry_gap(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-prefix"):
            publish_crates.prefix_length([True, False, True])

    def test_requires_exact_publishable_set(self) -> None:
        packages = [{"name": name, "dependencies": []} for name in publish_crates.PACKAGE_ORDER]
        self.assertEqual([p["name"] for p in publish_crates.require_expected_packages(packages)], list(publish_crates.PACKAGE_ORDER))
        with self.assertRaisesRegex(ValueError, "package set changed"):
            publish_crates.require_expected_packages(packages[:-1])

    def test_internal_dependency_must_target_release_version(self) -> None:
        packages = [{"name": "incin-core", "dependencies": [{"name": "incin-macros", "kind": None, "req": "^0.1.0"}]}, {"name": "incin-macros", "dependencies": []}]
        self.assertEqual(publish_crates.internal_dependency_problems(packages, "0.1.0"), [])
        packages[0]["dependencies"][0]["req"] = ">=0.0.1"
        self.assertEqual(len(publish_crates.internal_dependency_problems(packages, "0.1.0")), 1)

    def test_package_versions_must_match_tag(self) -> None:
        packages = [{"name": "incin-core", "version": "0.1.1"}]
        self.assertEqual(
            publish_crates.package_version_problems(packages, "0.1.0"),
            ["incin-core is version 0.1.1, not tag version 0.1.0"],
        )

    def test_registry_report_is_non_secret_and_prefix_aware(self) -> None:
        packages = [
            {
                "name": name,
                "targets": [{"name": name.replace("-", "_"), "kind": ["lib"], "doc": True}],
                "metadata": {"docs": {"rs": {"all-features": True}}},
            }
            for name in ("incin-macros", "incin-core")
        ]
        states, report = publish_crates.registry_report(
            packages,
            "0.1.0",
            lambda _: 200,
            lambda _: (200, {"users": [{"login": "release-owner"}]}),
        )
        self.assertEqual(states, [True, True])
        self.assertEqual(report["existing_prefix"], 2)
        self.assertIn("docs_rs", report["packages"][0])
        self.assertEqual(report["packages"][0]["owners"]["status"], "available")
        self.assertEqual(report["packages"][0]["owners"]["logins"], ["release-owner"])
        self.assertEqual(report["packages"][0]["docs_rs"]["status"], "available")


if __name__ == "__main__":
    unittest.main()
