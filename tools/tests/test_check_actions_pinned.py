#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "check-actions-pinned.py"
SPEC = importlib.util.spec_from_file_location("check_actions_pinned", SCRIPT)
assert SPEC and SPEC.loader
check_actions_pinned = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_actions_pinned)


class ActionPinTests(unittest.TestCase):
    def test_accepts_commit_pin_with_version_comment(self) -> None:
        text = "      uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1\n"
        self.assertEqual(check_actions_pinned.check_workflow_text(text), [])

    def test_allows_local_action(self) -> None:
        self.assertEqual(check_actions_pinned.check_workflow_text("      uses: ./local-action\n"), [])

    def test_rejects_tag_and_missing_comment(self) -> None:
        errors = check_actions_pinned.check_workflow_text("      uses: actions/checkout@v7\n")
        self.assertEqual(len(errors), 2)
        self.assertIn("not pinned", errors[0])
        self.assertIn("version comment", errors[1])


if __name__ == "__main__":
    unittest.main()
