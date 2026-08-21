"""Regression tests for the conservative cfg(test) classifier."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "check-unsafe-ledger.py"
SPEC = importlib.util.spec_from_file_location("check_unsafe_ledger", SCRIPT)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CfgTestClassifierTests(unittest.TestCase):
    def test_cfg_test_is_excluded(self) -> None:
        self.assertTrue(CHECKER.cfg_guarantees_test("test"))
        self.assertTrue(CHECKER.cfg_guarantees_test("all(test, unix)"))

    def test_cfg_not_test_remains_production(self) -> None:
        self.assertFalse(CHECKER.cfg_guarantees_test("not(test)"))

    def test_cfg_any_with_test_remains_production(self) -> None:
        self.assertFalse(CHECKER.cfg_guarantees_test('any(feature = "cpu", test)'))

    def test_only_guaranteed_test_item_is_masked(self) -> None:
        lines = [
            "#[cfg(test)]",
            "mod test_only {",
            "    unsafe { call(); }",
            "}",
            "#[cfg(not(test))]",
            "fn production() { unsafe { call(); } }",
            '#[cfg(any(feature = "cpu", test))]',
            "fn optional_production() { unsafe { call(); } }",
        ]
        self.assertEqual(CHECKER.test_only_lines(lines), {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
