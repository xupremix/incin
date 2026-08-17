#!/usr/bin/env python3
"""Validate a release tag, its checkout, and versioned publishable packages."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_NUMERIC = r"(?:0|[1-9][0-9]*)"
_PRERELEASE_IDENTIFIER = r"(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
TAG = re.compile(
    rf"^v(?P<version>{_NUMERIC}\.{_NUMERIC}\.{_NUMERIC}"
    rf"(?:-{_PRERELEASE_IDENTIFIER}(?:\.{_PRERELEASE_IDENTIFIER})*)?)$"
)


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    match = TAG.fullmatch(args.tag)
    if not match:
        sys.exit(f"invalid release tag {args.tag!r}; expected vMAJOR.MINOR.PATCH[-PRERELEASE]")
    version = match.group("version")

    try:
        tag_commit = run("git", "rev-parse", "--verify", f"refs/tags/{args.tag}^{{}}")
    except subprocess.CalledProcessError:
        sys.exit(f"release tag {args.tag!r} does not exist locally")
    head_commit = run("git", "rev-parse", "HEAD")
    if head_commit != tag_commit:
        sys.exit(f"checkout is {head_commit}, but {args.tag} resolves to {tag_commit}")

    metadata = json.loads(run("cargo", "metadata", "--no-deps", "--format-version", "1"))
    mismatches = []
    for package in metadata["packages"]:
        if package["publish"] != [] and package["version"] != version:
            mismatches.append(f"{package['name']}={package['version']}")
    if mismatches:
        sys.exit("publishable Cargo package versions do not match tag: " + ", ".join(mismatches))

    editor = json.loads(Path("editors/vscode/package.json").read_text(encoding="utf-8"))
    if editor["version"] != version:
        sys.exit(f"VS Code package version {editor['version']} does not match tag {args.tag}")

    print(f"release preflight passed: {args.tag} at {head_commit}")
    if args.github_output:
        args.github_output.open("a", encoding="utf-8").write(f"version={version}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
