#!/usr/bin/env python3
"""Safely publish Incin's crates.io release, one package at a time.

This tool deliberately makes the irreversible step small: ``publish`` accepts
one explicit package confirmation and never advances to the next package.
Use ``check`` before the release and ``verify --smoke`` after all packages are
visible.  It neither reads nor prints Cargo credentials.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ORDER = (
    "incin-macros", "incin-core", "incin-telemetry", "incin-viz-plugin-api",
    "incin-backends", "incin-data", "incin-diagnostics", "incin-lsp",
    "incin-viz", "incin",
)
_NUMERIC = r"(?:0|[1-9][0-9]*)"
_PRERELEASE = r"(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
TAG_RE = re.compile(rf"^v(?P<version>{_NUMERIC}\.{_NUMERIC}\.{_NUMERIC}(?:-{_PRERELEASE}(?:\.{_PRERELEASE})*)?)$")
HttpGet = Callable[[str], int]
JsonGet = Callable[[str], tuple[int, dict | None]]


def command(*args: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> str:
    return subprocess.run(args, cwd=cwd, env=env, text=True, check=True,
                          capture_output=True).stdout.strip()


def metadata() -> dict:
    return json.loads(
        command("cargo", "metadata", "--locked", "--no-deps", "--format-version", "1")
    )


def publishable_packages(data: dict) -> list[dict]:
    return [package for package in data["packages"] if package.get("publish") != []]


def require_expected_packages(packages: list[dict]) -> list[dict]:
    by_name = {package["name"]: package for package in packages}
    actual = set(by_name)
    expected = set(PACKAGE_ORDER)
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"publishable package set changed (missing={missing}, extra={extra})")
    ordered = [by_name[name] for name in PACKAGE_ORDER]
    positions = {name: index for index, name in enumerate(PACKAGE_ORDER)}
    for package in ordered:
        for dependency in package["dependencies"]:
            if dependency["name"] in positions and dependency.get("kind") != "dev":
                if positions[dependency["name"]] >= positions[package["name"]]:
                    raise ValueError(
                        f"fixed publish order places {package['name']} before its "
                        f"dependency {dependency['name']}"
                    )
    return ordered


def version_from_tag(tag: str) -> str:
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError("tag must be vMAJOR.MINOR.PATCH[-PRERELEASE]")
    return match.group("version")


def internal_dependency_problems(packages: list[dict], version: str) -> list[str]:
    names = {package["name"] for package in packages}
    problems: list[str] = []
    for package in packages:
        for dependency in package["dependencies"]:
            if dependency["name"] not in names or dependency.get("kind") == "dev":
                continue
            requirement = dependency["req"]
            # Cargo metadata has already normalized workspace inheritance.  The
            # release version must be the first compatible candidate; allowing a
            # wider lower bound would let Cargo select an older registry crate.
            accepted = {version, f"={version}", f"^{version}", f"~{version}"}
            if requirement not in accepted:
                problems.append(
                    f"{package['name']} -> {dependency['name']} uses {requirement!r}, "
                    f"which does not resolve the release version {version}"
                )
    return problems


def package_version_problems(packages: list[dict], version: str) -> list[str]:
    return [
        f"{package['name']} is version {package['version']}, not tag version {version}"
        for package in packages
        if package["version"] != version
    ]


def prefix_length(states: list[bool]) -> int:
    """Return the existing prefix, rejecting a registry gap."""
    first_missing = next((index for index, exists in enumerate(states) if not exists), len(states))
    if any(states[first_missing:]):
        raise ValueError("crates.io has a non-prefix release state; do not publish across a gap")
    return first_missing


def http_status(url: str) -> int:
    request = Request(url, headers={"User-Agent": "incin-release-helper/0.1"})
    try:
        with urlopen(request, timeout=20) as response:  # nosec B310: fixed HTTPS hosts below
            return response.status
    except HTTPError as error:
        return error.code
    except (URLError, TimeoutError) as error:
        reason = getattr(error, "reason", error)
        raise RuntimeError(f"request failed for {url}: {reason}") from error


def http_json(url: str) -> tuple[int, dict | None]:
    request = Request(url, headers={"User-Agent": "incin-release-helper/0.1"})
    try:
        with urlopen(request, timeout=20) as response:  # nosec B310: fixed HTTPS host below
            try:
                payload = json.load(response)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON from {url}") from error
            return response.status, payload
    except HTTPError as error:
        return error.code, None
    except (URLError, TimeoutError) as error:
        reason = getattr(error, "reason", error)
        raise RuntimeError(f"request failed for {url}: {reason}") from error


def crate_exists(name: str, version: str, get: HttpGet = http_status) -> bool:
    status = get(f"https://crates.io/api/v1/crates/{name}/{version}")
    if status == 200:
        return True
    if status == 404:
        return False
    raise RuntimeError(f"crates.io returned HTTP {status} for {name} {version}")


def clean_tagged_checkout(tag: str) -> str:
    if command("git", "status", "--porcelain"):
        raise ValueError("publish mode requires a clean working tree")
    commit = command("git", "rev-parse", "--verify", f"refs/tags/{tag}^{{}}")
    if command("git", "rev-parse", "HEAD") != commit:
        raise ValueError(f"HEAD is not the peeled commit for {tag}")
    result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/master"], cwd=ROOT)
    if result.returncode != 0:
        raise ValueError(f"{tag} is not reachable from origin/master")
    return commit


def run_static_gates(packages: list[dict], version: str) -> dict[str, list[str]]:
    subprocess.run([sys.executable, "tools/check-publish-metadata.py"], cwd=ROOT, check=True)
    if problems := internal_dependency_problems(packages, version):
        raise ValueError("\n".join(problems))
    # `--list` exposes exactly what would be uploaded without publishing.  The
    # package checker also asserts required source/license entries are present.
    subprocess.run(["bash", "tools/check-package.sh"], cwd=ROOT, check=True)
    file_lists: dict[str, list[str]] = {}
    for package in packages:
        files = command("cargo", "package", "-p", package["name"], "--allow-dirty", "--no-verify", "--list")
        if not files:
            raise ValueError(f"{package['name']} has an empty package file list")
        file_lists[package["name"]] = files.splitlines()
        print(f"{package['name']}: {len(file_lists[package['name']])} files inspected")
    return file_lists


def docs_rs_url(package: dict, version: str) -> str | None:
    target = next(
        (
            target
            for target in package["targets"]
            if target.get("doc")
            and any(kind in ("lib", "rlib", "proc-macro") for kind in target["kind"])
        ),
        None,
    )
    if target is None:
        return None
    return f"https://docs.rs/{package['name']}/{version}/{target['name']}/"


def owner_report(name: str, get_json: JsonGet = http_json) -> dict:
    url = f"https://crates.io/api/v1/crates/{name}/owners"
    status, payload = get_json(url)
    if status == 404:
        return {"url": url, "status": "not-found", "logins": []}
    if status != 200:
        return {"url": url, "status": f"http-{status}", "logins": []}
    users = payload.get("users", []) if isinstance(payload, dict) else []
    logins = sorted(
        user["login"]
        for user in users
        if isinstance(user, dict) and isinstance(user.get("login"), str)
    )
    return {"url": url, "status": "available", "logins": logins}


def registry_report(
    packages: list[dict],
    version: str,
    get: HttpGet = http_status,
    get_json: JsonGet = http_json,
) -> tuple[list[bool], dict]:
    states = [crate_exists(package["name"], version, get) for package in packages]
    existing = prefix_length(states)

    def endpoint_status(url: str) -> str:
        status = get(url)
        if status == 200:
            return "available"
        if status == 404:
            return "not-found"
        return f"http-{status}"

    package_reports = []
    for package, present in zip(packages, states):
        docs_url = docs_rs_url(package, version)
        package_reports.append(
            {
                "name": package["name"],
                "crates_io": "published" if present else "missing",
                "owners": owner_report(package["name"], get_json),
                "docs_rs": {
                    "url": docs_url,
                    "status": endpoint_status(docs_url) if docs_url else "not-applicable",
                    "configuration": (package.get("metadata") or {}).get("docs", {}).get("rs"),
                },
            }
        )

    report = {
        "version": version,
        "packages": package_reports,
        "existing_prefix": existing,
    }
    return states, report


def write_report(report: dict, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def wait_for_registry(name: str, version: str, timeout: int, interval: int) -> None:
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryDirectory(prefix="incin-publish-resolve-") as directory:
        root = Path(directory)
        (root / "Cargo.toml").write_text(
            f"[package]\nname = \"incin-release-resolver\"\nversion = \"0.0.0\"\nedition = \"2024\"\n"
            f"[dependencies]\n{name} = \"={version}\"\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src/main.rs").write_text("fn main() {}\n", encoding="utf-8")
        while True:
            if crate_exists(name, version):
                result = subprocess.run(["cargo", "metadata", "--format-version", "1"], cwd=root)
                if result.returncode == 0:
                    return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out waiting for {name} {version} in crates.io and Cargo")
            time.sleep(interval)


def smoke_verify(version: str) -> None:
    with tempfile.TemporaryDirectory(prefix="incin-release-smoke-") as directory:
        base = Path(directory)
        cargo_home, install_root = base / "cargo", base / "install"
        env = os.environ | {"CARGO_HOME": str(cargo_home), "CARGO_INSTALL_ROOT": str(install_root)}
        subprocess.run(
            [
                "cargo", "install", "incin-lsp", "--version", f"={version}",
                "--registry", "crates-io", "--locked",
            ],
            env=env,
            check=True,
        )
        lsp_bins = sorted(path.name for path in (install_root / "bin").iterdir())
        expected_lsp = ["incin-lsp.exe" if os.name == "nt" else "incin-lsp"]
        if lsp_bins != expected_lsp:
            raise RuntimeError(f"incin-lsp installation produced unexpected binaries: {lsp_bins}")
        subprocess.run(
            [
                "cargo", "install", "incin", "--version", f"={version}",
                "--registry", "crates-io", "--bin", "cargo-incin", "--locked",
            ],
            env=env,
            check=True,
        )
        env["PATH"] = str(install_root / "bin") + os.pathsep + env.get("PATH", "")
        subprocess.run(["cargo", "incin", "doctor"], env=env, check=True)
        consumer = base / "consumer"
        (consumer / "src").mkdir(parents=True)
        (consumer / "Cargo.toml").write_text(
            f"[package]\nname = \"incin-release-consumer\"\nversion = \"0.0.0\"\nedition = \"2024\"\n[dependencies]\nincin = \"={version}\"\n", encoding="utf-8")
        (consumer / "src/main.rs").write_text("fn main() {}\n", encoding="utf-8")
        subprocess.run(["cargo", "check"], cwd=consumer, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "publish", "verify"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--confirm", metavar="PACKAGE", help="publish this one next package only")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--smoke", action="store_true", help="run slow post-publication clean-CARGO_HOME checks")
    args = parser.parse_args()
    try:
        version = version_from_tag(args.tag)
        packages = require_expected_packages(publishable_packages(metadata()))
        if problems := package_version_problems(packages, version):
            raise ValueError("\n".join(problems))
        if args.command == "publish":
            clean_tagged_checkout(args.tag)
        file_lists = run_static_gates(packages, version)
        states, report = registry_report(packages, version)
        for package in report["packages"]:
            package["files"] = file_lists[package["name"]]
        write_report(report, args.report)
        next_index = prefix_length(states)
        if args.command == "check":
            print(f"publication check passed: {next_index}/{len(packages)} packages already visible")
        elif args.command == "publish":
            if next_index == len(packages):
                print("all packages are already published; cargo publish was not called")
            else:
                next_name = PACKAGE_ORDER[next_index]
                if args.confirm != next_name:
                    raise ValueError(f"next package is {next_name}; re-run with --confirm {next_name}")
                subprocess.run(
                    [
                        "cargo", "publish", "-p", next_name, "--locked",
                        "--registry", "crates-io",
                    ],
                    cwd=ROOT,
                    check=True,
                )
                wait_for_registry(next_name, version, args.timeout, args.interval)
                print(f"published and resolved {next_name} {version}; re-run for the next package")
        else:
            if next_index != len(packages):
                raise ValueError("cannot verify installation before every package is visible on crates.io")
            missing_owners = [
                package["name"]
                for package in report["packages"]
                if package["owners"]["status"] != "available"
                or not package["owners"]["logins"]
            ]
            if missing_owners:
                raise ValueError(
                    "could not record crates.io owners for: " + ", ".join(missing_owners)
                )
            missing_docs = [
                package["name"]
                for package in report["packages"]
                if package["docs_rs"]["status"] not in ("available", "not-applicable")
            ]
            if missing_docs:
                raise ValueError(
                    "docs.rs artifacts are not available for: " + ", ".join(missing_docs)
                )
            if args.smoke:
                smoke_verify(version)
            print("registry verification passed" + (" (including install smoke)" if args.smoke else ""))
        return 0
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"publish helper failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
