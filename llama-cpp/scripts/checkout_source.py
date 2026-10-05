#!/usr/bin/env python3
"""Materialize one already-registered pinned source; never repair pins/gitlinks.

The updater must not recursively fetch every unrelated runtime just to update
nightly. Source validation happens before network or checkout work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from source_config import get_source, patch_series
from source_git import check_gitlink, check_index_gitlink, git

ROOT = Path(__file__).resolve().parents[1]


def check_existing_checkout(directory: Path) -> None:
    """Reject edits before update can move a detached HEAD or remove tracked files."""
    for component in [*reversed(directory.parents), directory]:
        if component.is_symlink():
            raise ValueError('Linked source checkout: ' + str(component))
    if not directory.exists():
        return
    if not directory.is_dir():
        raise ValueError('Selected source is not a directory')
    if not any(directory.iterdir()):
        return
    top = git(directory, 'rev-parse', '--show-toplevel', check=False)
    if top.returncode or Path(top.stdout.strip()).resolve() != directory.resolve():
        raise ValueError('Selected source is not a separate Git checkout')
    if git(directory, 'status', '--porcelain', '--untracked-files=all', '--ignore-submodules=none').stdout:
        raise ValueError('Selected upstream checkout is dirty; refusing to continue')


def checkout_source(root: Path, source_id: str) -> dict:
    root = root.absolute()
    entry = get_source(root, source_id)
    check_gitlink(root, entry)
    check_index_gitlink(root, entry)
    patch_series(root, entry)
    directory = root / entry['vendorPath']
    check_existing_checkout(directory)
    # No --remote or --force. Explicit --checkout prevents local update=merge,
    # rebase, none or custom-command policies from changing this pinned operation.
    subprocess.run(['git', '--no-replace-objects', 'submodule', 'update', '--init',
                    '--checkout', '--recursive', '--depth=1', '--', entry['vendorPath']],
                   cwd=root, check=True, timeout=600)
    top = Path(git(directory, 'rev-parse', '--show-toplevel').stdout.strip()).resolve()
    if top != directory.resolve():
        raise ValueError('Selected source is not a separate Git checkout')
    actual = git(directory, 'rev-parse', '--verify', 'HEAD^{commit}').stdout.strip()
    if actual != entry['commit']:
        raise ValueError('Checked-out source does not match the committed pin')
    if git(directory, 'status', '--porcelain', '--untracked-files=all', '--ignore-submodules=none').stdout:
        raise ValueError('Selected upstream checkout is dirty; refusing to continue')
    return {'source': source_id, 'vendorPath': entry['vendorPath'], 'commit': actual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(checkout_source(ROOT, args.source)))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
