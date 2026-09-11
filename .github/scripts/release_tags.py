#!/usr/bin/env python3
"""Classify a release tag against the stable tags in this repository."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable


_IDENTIFIER = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
_TAG = re.compile(
    rf"^(0|[1-9][0-9]*)\."
    rf"(0|[1-9][0-9]*)\."
    rf"(0|[1-9][0-9]*)"
    rf"(?:-({_IDENTIFIER}(?:\.{_IDENTIFIER})*))?$"
)


def parse_tag(tag: str) -> tuple[tuple[int, int, int], bool] | None:
    """Return the numeric version and stability of a canonical release tag."""
    match = _TAG.fullmatch(tag)
    if match is None:
        return None
    return tuple(map(int, match.group(1, 2, 3))), match.group(4) is None


def classify(candidate: str, tags: Iterable[str]) -> dict[str, bool]:
    """Return publication decisions for *candidate*."""
    parsed = parse_tag(candidate)
    if parsed is None:
        return {
            "is_version": False,
            "is_stable": False,
            "is_latest": False,
            "is_series_latest": False,
        }

    version, is_stable = parsed
    stable_versions = [
        parsed_tag[0]
        for tag in tags
        if (parsed_tag := parse_tag(tag)) is not None and parsed_tag[1]
    ]
    series_versions = [other for other in stable_versions if other[:2] == version[:2]]
    return {
        "is_version": True,
        "is_stable": is_stable,
        "is_latest": is_stable and version == max(stable_versions, default=version),
        "is_series_latest": is_stable
        and version == max(series_versions, default=version),
    }


def main() -> None:
    """Print GitHub Actions outputs for the tag passed on the command line."""
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} TAG")
    tags = subprocess.run(
        ["git", "tag", "--list"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for name, enabled in classify(sys.argv[1], tags).items():
        print(f"{name}={str(enabled).lower()}")


if __name__ == "__main__":
    main()
