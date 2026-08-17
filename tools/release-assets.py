#!/usr/bin/env python3
"""Create and verify the exact release artifact and checksum manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

TARGETS = ("x86_64-unknown-linux-gnu", "aarch64-apple-darwin", "x86_64-pc-windows-msvc")


def expected(version: str) -> list[str]:
    return [
        f"incin-book-{version}.html",
        f"incin-book-{version}.tar.gz",
        f"incin-book-site-{version}.tar.gz",
        f"incin-lsp-nvim-{version}.tar.gz",
        f"incin-lsp-vscode-{version}.vsix",
        f"incin-rustrover-external-tool-{version}.tar.gz",
        f"incin-{version}-{TARGETS[0]}.tar.gz",
        f"incin-{version}-{TARGETS[1]}.tar.gz",
        f"incin-{version}-{TARGETS[2]}.zip",
    ]


def checksum_name(version: str) -> str:
    return f"incin-{version}-SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_manifest(directory: Path, version: str) -> list[str]:
    manifest = directory / "expected-assets.txt"
    if not manifest.is_file():
        sys.exit(f"missing expected asset manifest: {manifest}")
    listed = manifest.read_text(encoding="utf-8").splitlines()
    if listed != expected(version):
        sys.exit("expected asset manifest does not match the release contract")
    return listed


def cmd_manifest(args: argparse.Namespace) -> None:
    (args.directory / "expected-assets.txt").write_text("\n".join(expected(args.version)) + "\n", encoding="utf-8")


def cmd_checksums(args: argparse.Namespace) -> None:
    assets = require_manifest(args.directory, args.version)
    missing = [name for name in assets if not (args.directory / name).is_file()]
    if missing:
        sys.exit("cannot checksum missing assets: " + ", ".join(missing))
    lines = [f"{sha256(args.directory / name)}  {name}" for name in assets]
    (args.directory / checksum_name(args.version)).write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_verify(args: argparse.Namespace) -> None:
    assets = require_manifest(args.directory, args.version)
    actual = sorted(path.name for path in args.directory.iterdir() if path.is_file())
    allowed = sorted([*assets, "expected-assets.txt", checksum_name(args.version)])
    if actual != allowed:
        sys.exit(f"release asset set differs from expected: found {actual}, expected {allowed}")
    checksum_file = args.directory / checksum_name(args.version)
    if not checksum_file.is_file():
        sys.exit(f"missing checksum file: {checksum_file}")
    expected_lines = [f"{sha256(args.directory / name)}  {name}" for name in assets]
    if checksum_file.read_text(encoding="utf-8").splitlines() != expected_lines:
        sys.exit("SHA-256 checksum manifest does not match release assets")
    print(f"release assets verified: {len(assets)} assets and SHA-256 manifest")


def cmd_verify_github(args: argparse.Namespace) -> None:
    version = args.tag.removeprefix("v")
    cmd_verify(argparse.Namespace(directory=args.directory, version=version))
    release = json.loads(subprocess.check_output(["gh", "release", "view", args.tag, "--json", "isDraft,assets"], text=True))
    if not release["isDraft"]:
        sys.exit("release must remain a draft until final verification completes")
    uploaded = sorted(asset["name"] for asset in release["assets"])
    required = sorted([*expected(version), checksum_name(version)])
    if uploaded != required:
        sys.exit(f"draft release assets differ from verified manifest: {uploaded}")
    print(f"draft release verified: {len(uploaded)} uploaded assets")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    for name in ("manifest", "checksums", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--version", required=True)
        command.add_argument("--directory", required=True, type=Path)
        command.set_defaults(function=globals()[f"cmd_{name}"])
    github = commands.add_parser("verify-github")
    github.add_argument("--tag", required=True)
    github.add_argument("--directory", required=True, type=Path)
    github.set_defaults(function=cmd_verify_github)
    args = parser.parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
