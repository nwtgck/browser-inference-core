"""Compose the image runtime from independent, locally checked-out upstreams."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_tools.source.checkout import checked_sources, commit_file
from build_tools.source.files import absolute, directory, regular_file, relative_path
from build_tools.source.workspace import prepare_sources

from build_tools.project import REPOSITORY_ROOT as REPO, runtime_root

ROOT = runtime_root("stable-diffusion-cpp")
ROLES = frozenset({"stable-diffusion", "ggml"})


def prepare(sources: dict[str, Path], destination: Path, patches: Path = ROOT / "upstream-patches",
            *, source_files: dict[str, tuple[str, ...]] | None = None,
            upstreams: dict | None = None, replace: bool = False) -> dict:
    """Compose already selected local inputs; the CLI checks their gitlinks first."""
    if set(sources) != ROLES:
        raise ValueError("Unknown image-runtime source set")
    commits = {role: entry["commit"] for role, entry in upstreams.items()} if upstreams is not None else None
    return prepare_sources(sources, destination, patches, source_files=source_files,
                           source_commits=commits, upstreams=upstreams, replace=replace)


def resolve_sources(repository: Path, runtime: Path, *, revision: str | None = None) -> tuple[dict[str, Path], dict]:
    """Resolve source roles from the selected parent commit or local index.

    Configuration owns source roles/repository names, not a second copy of the
    pinned SHA. The returned provenance records the exact checked-out commits.
    """
    repository, runtime = directory(repository), directory(runtime)
    relative_runtime = runtime.relative_to(repository)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate source declaration: {key}")
            result[key] = value
        return result

    contents = (regular_file(runtime / 'config/upstreams.json').read_bytes() if revision is None
                else commit_file(repository, revision, (relative_runtime / 'config/upstreams.json').as_posix()))
    declarations = json.loads(contents, object_pairs_hook=unique)
    if not isinstance(declarations, dict) or set(declarations) != ROLES:
        raise ValueError("Unknown image-runtime source set")
    paths = {}
    for role, entry in declarations.items():
        fields = {"path", "repository", "sourceSubdirectory"} if role == "ggml" else {"path", "repository"}
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"Unknown source declaration fields: {role}")
        if not isinstance(entry["repository"], str) or not entry["repository"].strip():
            raise ValueError(f"Missing upstream repository: {role}")
        path = relative_path(entry["path"])
        if "sourceSubdirectory" in entry:
            relative_path(entry["sourceSubdirectory"])
        paths[role] = (relative_runtime / path).as_posix()
    checkouts = checked_sources(repository, paths, revision=revision)
    for role, entry in declarations.items():
        if "sourceSubdirectory" in entry:
            directory(checkouts[role].path / relative_path(entry["sourceSubdirectory"]))
    return ({role: checkout.path for role, checkout in checkouts.items()},
            {role: {**declarations[role], "commit": checkout.commit} for role, checkout in checkouts.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = absolute(args.output)
    if not output.is_relative_to(ROOT / "build"):
        parser.error("Output must be inside the runtime build directory")
    try:
        sources, upstreams = resolve_sources(REPO, ROOT)
        prepare(sources, output, upstreams=upstreams)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
