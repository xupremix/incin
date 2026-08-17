#!/usr/bin/env python3
"""Verify every publishable workspace crate carries complete crates.io metadata.

`cargo publish --dry-run` cannot serve as this gate before the first release:
it resolves each crate's path dependencies against the registry, and for an
unreleased version those candidates do not exist, so the command fails for a
reason that says nothing about whether the manifest is publishable. This checks
the part that is checkable today and stays checkable afterwards — that the
fields crates.io and docs.rs read are present, non-empty, and within the limits
the registry enforces.

The gate exists because the workspace shipped for months with no `rust-version`
anywhere, no keywords, no categories, and no docs.rs feature configuration, so
docs.rs would have built the default feature set and documented a fraction of
the API.

Run directly, or through CI:

    python3 tools/check-publish-metadata.py
"""

from __future__ import annotations

import json
import subprocess
import sys

# crates.io limits.
MAX_KEYWORDS = 5
MAX_KEYWORD_LEN = 20
MAX_CATEGORIES = 5
MAX_DESCRIPTION = 1000

REQUIRED_STRINGS = ("description", "license", "repository", "documentation", "homepage")


def workspace_packages() -> list[dict]:
    raw = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(raw)["packages"]


def publishable(package: dict) -> bool:
    # `publish` is null when unrestricted, and a (possibly empty) list when the
    # manifest names registries. An empty list is `publish = false`.
    return package.get("publish") != []


def check(package: dict) -> list[str]:
    name = package["name"]
    problems: list[str] = []

    for field in REQUIRED_STRINGS:
        value = package.get(field)
        if not value or not str(value).strip():
            problems.append(f"{name}: `{field}` is missing or empty")

    description = package.get("description") or ""
    if len(description) > MAX_DESCRIPTION:
        problems.append(
            f"{name}: `description` is {len(description)} characters, "
            f"over the crates.io limit of {MAX_DESCRIPTION}"
        )

    if not package.get("rust_version"):
        problems.append(
            f"{name}: `rust-version` is missing. An MSRV nothing declares is an "
            "MSRV nothing can hold to."
        )

    keywords = package.get("keywords") or []
    if not keywords:
        problems.append(f"{name}: `keywords` is empty, so the crate is unsearchable")
    if len(keywords) > MAX_KEYWORDS:
        problems.append(
            f"{name}: {len(keywords)} keywords, over the crates.io limit of {MAX_KEYWORDS}"
        )
    for keyword in keywords:
        if len(keyword) > MAX_KEYWORD_LEN:
            problems.append(
                f"{name}: keyword {keyword!r} is longer than {MAX_KEYWORD_LEN} characters"
            )
        if not keyword.replace("-", "").replace("_", "").isalnum():
            problems.append(f"{name}: keyword {keyword!r} is not alphanumeric")

    categories = package.get("categories") or []
    if not categories:
        problems.append(f"{name}: `categories` is empty")
    if len(categories) > MAX_CATEGORIES:
        problems.append(
            f"{name}: {len(categories)} categories, over the crates.io limit of "
            f"{MAX_CATEGORIES}"
        )

    # docs.rs builds the default feature set unless the manifest says otherwise.
    # For a workspace whose interesting surface is behind features, that is the
    # difference between documenting the crate and documenting a slice of it.
    docs_rs = (package.get("metadata") or {}).get("docs", {}).get("rs")
    if docs_rs is None:
        problems.append(
            f"{name}: no `[package.metadata.docs.rs]`, so docs.rs will build only "
            "the default features"
        )
    elif not (docs_rs.get("features") or docs_rs.get("all-features")):
        problems.append(
            f"{name}: `[package.metadata.docs.rs]` names neither `features` nor "
            "`all-features`"
        )

    return problems


def main() -> int:
    packages = [p for p in workspace_packages() if publishable(p)]
    if not packages:
        print("no publishable packages found; is this a workspace?", file=sys.stderr)
        return 1

    problems: list[str] = []
    for package in sorted(packages, key=lambda p: p["name"]):
        problems.extend(check(package))

    if problems:
        print("publish metadata is incomplete:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            f"\n{len(problems)} problem(s) across {len(packages)} publishable crate(s).",
            file=sys.stderr,
        )
        return 1

    versions = {p["version"] for p in packages}
    if len(versions) != 1:
        print(
            f"publishable crates disagree on version: {sorted(versions)}",
            file=sys.stderr,
        )
        return 1

    print(
        f"publish metadata complete for {len(packages)} crate(s) at "
        f"version {versions.pop()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
