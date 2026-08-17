#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "release-preflight.py"
SPEC = importlib.util.spec_from_file_location("release_preflight", SCRIPT)
assert SPEC and SPEC.loader
release_preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_preflight)


class ReleaseTagTests(unittest.TestCase):
    def test_version_includes_prerelease(self) -> None:
        match = release_preflight.TAG.fullmatch("v1.2.3-rc.1")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group("version"), "1.2.3-rc.1")

    def test_accepts_stable_semver(self) -> None:
        self.assertIsNotNone(release_preflight.TAG.fullmatch("v0.1.0"))

    def test_rejects_malformed_tags(self) -> None:
        for tag in ("1.2.3", "v1.2", "v1.2.3-", "v01.2.3", "v1.2.3-01"):
            with self.subTest(tag=tag):
                self.assertIsNone(release_preflight.TAG.fullmatch(tag))


if __name__ == "__main__":
    unittest.main()
