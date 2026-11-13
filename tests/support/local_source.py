"""Create explicitly synthetic local Git history for source-bound contract tests.

A source-only archive can exercise capture/change detection without pretending
that its commit is an upstream pin. An initialized checkout is cloned locally;
a pruned archive receives a new fixture commit and a conspicuous commit message.
This helper is test-only and never used by a production source preparer.
"""
from pathlib import Path
import shutil
import subprocess

from build_tools.source.checkout import git, git_environment


def source_repository(source: Path, destination: Path) -> Path:
    source, destination = source.resolve(), destination.absolute()
    if not source.is_dir():
        raise ValueError('Missing source fixture directory: ' + str(source))
    if destination.exists() or destination.is_symlink():
        raise ValueError('Source fixture destination must be absent')
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source fixture must not overlap its input')
    if (source / '.git').exists():
        subprocess.run(['git', '-c', 'protocol.file.allow=always', 'clone', '--quiet', '--no-hardlinks',
                        str(source), str(destination)], check=True, env=git_environment())
    else:
        shutil.copytree(source, destination, symlinks=True,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
        git(destination, 'init', '--quiet')
        git(destination, 'config', 'user.name', 'Local synthetic source fixture')
        git(destination, 'config', 'user.email', 'fixture@example.invalid')
        git(destination, 'add', '--force', '--all')
        git(destination, 'commit', '--quiet', '-m',
            'test: source archive fixture; not verified upstream Git history')
    return destination
