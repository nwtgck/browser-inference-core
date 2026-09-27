"""Copy complete notices, including repository licenses for compiled subtrees."""
from pathlib import Path
from pipeline_metrics import measured
import shutil


def copy_notice(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f'Missing, linked, or empty notice: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


@measured('package.collect_notices')
def collect_notices(source: Path, destination: Path, *, required: bool = True) -> int:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f'Missing or linked notice source: {source}')
    count = 0
    for path in sorted(source.rglob('*')):
        if '.git' in path.parts:
            continue
        # Never follow a directory symlink out of the requested tree.
        if path.is_symlink():
            if path.name.upper().startswith(('LICENSE', 'COPYING', 'COPYRIGHT')):
                raise ValueError(f'Linked notice input: {path}')
            continue
        if path.is_file() and path.name.upper().startswith(('LICENSE', 'COPYING', 'COPYRIGHT')):
            copy_notice(path, destination / path.relative_to(source))
            count += 1
    if required and not count:
        raise ValueError(f'No license notices: {source}')
    return count


@measured('package.collect_subtree_notices')
def collect_subtree_notices(repository: Path, subtree: str, destination: Path) -> None:
    source = repository / subtree
    if repository.is_symlink() or not source.resolve().is_relative_to(repository.resolve()):
        raise ValueError('Unsafe notice subtree')
    # The root license applies to this subtree, even when it has no LICENSE file.
    # Keep subtree notices separately to avoid overwriting either file.
    copy_notice(repository / 'LICENSE', destination / 'LICENSE')
    collect_notices(source, destination / subtree, required=False)


def stage_toolchain_notices(roots: list[Path], output: Path) -> None:
    names = [root.name for root in roots]
    if len(names) != len(set(names)):
        raise ValueError('Toolchain license roots must have distinct names')
    target = output / 'toolchain-licenses'
    if target.exists() or target.is_symlink():
        raise ValueError('Toolchain notices already staged')
    for root in roots:
        collect_notices(root, target / root.name)
