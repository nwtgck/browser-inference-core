#!/usr/bin/env python3
"""Apply an mtmd workaround to a build-tree copy, never vendor/."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from source_config import patch_series, DEFAULT_SOURCE

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'


def prepare(source: Path, output: Path, patch: Path, *, capture_output: bool = False,
            filename: str = "clip.cpp") -> Path:
    if filename not in ("clip.cpp", "mtmd-audio.cpp"):
        raise ValueError("unsupported mtmd overlay translation unit")
    # Resolve symlinks before checking overlap so a build path cannot disguise
    # a write inside the pinned checkout (or replace one of its ancestors).
    source = source.resolve()
    output = output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("the overlay must be outside the upstream source tree")
    output.mkdir(parents=True, exist_ok=True)
    # Always start from pristine input, including after a previous configure.
    # --check fails closed when an upstream update no longer matches the patch.
    with tempfile.TemporaryDirectory(prefix="mtmd-overlay-", dir=output) as temporary:
        work = Path(temporary)
        shutil.copyfile(source / "tools/mtmd" / filename, work / filename)
        # --no-index applies to this scratch directory even though a build tree
        # can be nested in the lcore repository. Never patch the repository root.
        for args in (["--check"], []):
            subprocess.run(["git", "apply", "--no-index", "--whitespace=error", *args,
                            str(patch.resolve())], cwd=work, check=True,
                           capture_output=capture_output, text=capture_output)
        destination = output / filename
        patched = (work / filename).read_bytes()
        # Publish only after successful application, leaving the previous copy
        # intact on failure. The exception still aborts configure; a stale copy
        # is not a fallback. Avoid recompiling on a no-op CMake configure.
        if not destination.exists() or destination.read_bytes() != patched:
            (work / filename).replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", default=DEFAULT_SOURCE)
    parser.add_argument("--component", choices=("vision", "audio"), default="vision")
    args = parser.parse_args()
    entries=patch_series(ROOT,args.source_id)
    item=next(p for p in entries if p['component']==args.component)
    prepare(args.source,args.output,ROOT/item['file'],filename=item['translationUnit'])



if __name__ == "__main__":
    main()
