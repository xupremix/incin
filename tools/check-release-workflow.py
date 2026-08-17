#!/usr/bin/env python3
"""Static guard for release workflow integrity invariants."""
from __future__ import annotations

import sys
from pathlib import Path
import re


WORKFLOW = Path(".github/workflows/release.yml")
REQUIRED = (
    "xvfb-run -a npm test",
    "actions/upload-artifact@v4",
    "actions/download-artifact@v4",
    "release-assets.py checksums",
    "release-assets.py verify",
    "gh release create \"$RELEASE_TAG\" --target \"$RELEASE_TAG\" --title \"$RELEASE_TAG\" --generate-notes --draft",
    "release-assets.py verify-github",
    "gh release edit \"$RELEASE_TAG\" --draft=false",
    "incin-rustrover-external-tool-",
)
FORBIDDEN = (
    "npm version ",
    "needs: create-release",
    "gh release upload \"$RELEASE_TAG\" incin-",
)


def main() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")
    missing = [needle for needle in REQUIRED if needle not in text]
    present = [needle for needle in FORBIDDEN if needle in text]
    preflight = re.search(r"(?ms)^  preflight:\n.*?(?=^  [A-Za-z0-9_-]+:\n|\Z)", text)
    book = re.search(r"(?ms)^  book:\n.*?(?=^  [A-Za-z0-9_-]+:\n|\Z)", text)
    if preflight is None or 'release-preflight.py --tag "$RELEASE_TAG"' not in preflight.group():
        missing.append("preflight job validates RELEASE_TAG")
    if book is None:
        missing.append("book job")
    else:
        book_text = book.group()
        browser = "run: sudo apt-get update && sudo apt-get install -y chromium"
        browser_at = book_text.find(browser)
        test_at = book_text.find("python3 tools/test-book-site.py")
        if browser_at < 0:
            missing.append("book job installs Chromium")
        elif test_at < 0 or browser_at > test_at:
            missing.append("Chromium install precedes the book browser test")
    for job_name in ("verify-assets", "draft-release", "publish-release"):
        job = re.search(
            rf"(?ms)^  {re.escape(job_name)}:\n.*?(?=^  [A-Za-z0-9_-]+:\n|\Z)",
            text,
        )
        if job is None:
            missing.append(f"{job_name} job")
        elif "uses: actions/checkout@v5" not in job.group():
            missing.append(f"{job_name} job checks out release tooling")
    if missing or present:
        if missing:
            print("missing release safeguards: " + ", ".join(missing), file=sys.stderr)
        if present:
            print("forbidden release behavior: " + ", ".join(present), file=sys.stderr)
        return 1
    print("release workflow static checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
