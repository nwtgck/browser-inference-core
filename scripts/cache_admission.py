#!/usr/bin/env python3
"""Admit restored intermediates before use; preserve the freshly installed SDK.

Cache writers remain trusted: checks of file type, key, and SDK baseline do NOT
prove that additional compiler objects were built honestly. No compiler runs
between park and finish, including on a partially failed Actions restore.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat

from browser_toolchain import load_toolchain
from package_inputs import file_identity, read_regular, regular_path
from pipeline_metrics import emit, print_observation, span

ROOT = Path(__file__).resolve().parents[1]
EM_PATH = Path('.tools/emsdk/upstream/emscripten/cache')
MAX_ENTRIES = 100_000
MAX_BYTES = 8 * 1024**3


class Rejected(ValueError):
    """Only fixed reasons, not arbitrary paths or exception messages, are logged."""


def remove_entry(path: Path) -> None:
    regular_path(path.parent, directory=True)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode):
        shutil.rmtree(path)  # fd-based implementation; never traverses links.
    else:
        path.unlink()


def inspect_tree(root: Path, expected: dict | None = None, *, baseline: bool = False) -> dict:
    """Bounded, non-dereferencing traversal; hash only installed baseline files.

    Unlike package paths, SDK/cache paths may be case-sensitive (_exit / _Exit).
    Existing SDK symlinks are admitted only with the exact captured link target.
    New symlinks and multiply linked files from cache archives are not admitted.
    """
    regular_path(root, directory=True)
    records = {}; total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                name = Path(entry.path).relative_to(root).as_posix()
                if len(records) >= MAX_ENTRIES:
                    raise Rejected('too-many-entries')
                if '\\' in name or any(ord(c) < 32 for c in name):
                    raise Rejected('unsafe-path')
                info = entry.stat(follow_symlinks=False)
                if name == 'cache.lock' and not stat.S_ISREG(info.st_mode):
                    raise Rejected('invalid-cache-lock')
                if stat.S_ISDIR(info.st_mode):
                    records[name] = {'directory': True}
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    if not baseline and info.st_nlink != 1:
                        raise Rejected('hard-linked-file')
                    total += info.st_size
                    if total > MAX_BYTES:
                        raise Rejected('oversized-tree')
                    # cache.lock is runtime state, not an SDK input. Its type is
                    # checked, but lock acquisition may legitimately change it.
                    if name != 'cache.lock' and (baseline or (expected or {}).get(name, {}).get('sha256')):
                        records[name] = file_identity(Path(entry.path))
                    else:
                        records[name] = {'bytes': info.st_size}
                elif stat.S_ISLNK(info.st_mode):
                    link = {'link': os.readlink(entry.path)}
                    if not baseline and (expected or {}).get(name) != link:
                        raise Rejected('unexpected-link')
                    records[name] = link
                else:
                    raise Rejected('special-file')
    for name, identity in (expected or {}).items():
        if name != 'cache.lock' and records.get(name) != identity:
            raise Rejected('sdk-baseline-mismatch')
    return records


def decision(kind: str, requested: str, matched: str, hit: str, outcome: str, mode: str) -> str:
    if mode not in ('enabled', 'cold') or kind not in ('dawn', 'em', 'cc'):
        raise ValueError('Unknown cache mode or kind')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,512}', requested):
        raise ValueError('Invalid requested cache key')
    if mode == 'cold':
        return 'cold'
    if outcome != 'success':
        return 'restore-not-completed'
    if not matched:
        return 'miss'
    if matched == requested and hit == 'true':
        return 'exact'
    # Only a complete SHA suffix within this compiler/runtime/profile/variant.
    if (kind == 'cc' and hit == 'false' and len(requested) > 40 and
            re.fullmatch('[0-9a-f]{40}', requested[-40:]) and
            len(matched) == len(requested) and matched.startswith(requested[:-40]) and
            re.fullmatch('[0-9a-f]{40}', matched[-40:])):
        return 'prefix'
    return 'key-or-status-mismatch'


def state_paths(root: Path) -> tuple[Path, Path, Path]:
    active = root / EM_PATH
    return active, active.with_name('cache.bic-pristine'), root / '.tools/cache-admission-em.json'


def park_em(root: Path, requested: str) -> None:
    decision('em', requested, '', '', '', 'enabled')  # validate supplied identity
    active, pristine, state = state_paths(root)
    regular_path(active.parent, directory=True)
    regular_path(state.parent, directory=True)
    if pristine.exists() or pristine.is_symlink() or state.exists() or state.is_symlink():
        raise ValueError('Unfinished cache admission; start with a fresh toolchain directory')
    # Some SDKs populate this directory lazily. Preserve that empty baseline,
    # but never invent the compiler installation if its parent is absent.
    if not active.exists() and not active.is_symlink():
        active.mkdir()
    regular_path(active, directory=True)
    with span('cache.park_sdk', kind='em'):
        baseline = inspect_tree(active, baseline=True)
        # Write before moving; a metadata I/O failure cannot lose the SDK.
        with state.open('x') as output:
            json.dump({'schemaVersion': 1, 'requested': requested, 'phase': 'parked',
                       'baseline': baseline}, output, separators=(',', ':'))
        try:
            active.rename(pristine)
            try:
                active.mkdir()
            except BaseException:
                pristine.rename(active)
                raise
        except BaseException:
            state.unlink()
            raise


def read_state(path: Path) -> dict:
    record = json.loads(read_regular(path, limit=32 * 1024**2))
    if (not isinstance(record, dict) or type(record.get('schemaVersion')) is not int or
            record['schemaVersion'] != 1 or not isinstance(record.get('baseline'), dict) or
            any(not isinstance(value, dict) for value in record['baseline'].values())):
        raise ValueError('Invalid cache admission state')
    return record


def finish_em(root: Path, requested: str, matched: str, hit: str,
              outcome: str, mode: str) -> dict:
    active, pristine, state = state_paths(root)
    record = read_state(state)
    if record.get('requested') != requested or record.get('phase') != 'parked':
        raise ValueError('Cache admission state does not match this restore')
    regular_path(pristine, directory=True)
    reason = decision('em', requested, matched, hit, outcome, mode)
    accepted = False
    with span('cache.admit', kind='em'):
        if reason == 'exact':
            try:
                inspect_tree(active, record['baseline'])
                accepted = True
            except (OSError, ValueError):
                reason = 'invalid-tree-or-sdk-baseline'
        if accepted:
            remove_entry(pristine)
        else:
            # Delete even a partial restore. Do not merge it with SDK files or
            # clear the SDK's sysroot as a substitute for restoring the baseline.
            remove_entry(active)
            pristine.rename(active)
        record['phase'] = 'ready'
        state.write_text(json.dumps(record, separators=(',', ':')) + '\n')
    return report('em', reason, accepted, requested, matched, mode)


def make_directories(root: Path, relative: Path) -> Path:
    """Reject existing parent links before creating anything beneath them.

    This is a job-owned workspace, not a sandbox against concurrent same-UID
    modification. The check must still precede mkdir on an existing parent.
    """
    regular_path(root, directory=True)
    path = root
    for part in relative.parts:
        if part in ('..', '.') or not part:
            raise ValueError('Invalid directory component')
        path = path / part
        try:
            path.mkdir()
        except FileExistsError:
            pass
        regular_path(path, directory=True)
    return path


def cc_path(root: Path, relative: str) -> Path:
    if not re.fullmatch(r'\.cache/ccache/(llama-cpp|stable-diffusion-cpp)/[a-z0-9-]+/(browser|test)', relative):
        raise ValueError('Invalid compiler-cache path')
    path = root / relative
    make_directories(root, path.parent.relative_to(root))
    return path


def clear_cc(root: Path, relative: str) -> None:
    path = cc_path(root, relative)
    remove_entry(path)
    path.mkdir()


def finish_cc(root: Path, relative: str, requested: str, matched: str, hit: str,
              outcome: str, mode: str) -> dict:
    path = cc_path(root, relative)
    reason = decision('cc', requested, matched, hit, outcome, mode)
    accepted = False
    with span('cache.admit', kind='cc'):
        if reason in ('exact', 'prefix'):
            try:
                inspect_tree(path)
                accepted = True
            except (OSError, ValueError):
                reason = 'invalid-tree'
        if not accepted:
            clear_cc(root, relative)
    return report('cc', reason, accepted, requested, matched, mode)


def dawn_path(root: Path) -> Path:
    cfg = load_toolchain(root)
    # Pins are repository-owned, but never interpolate arbitrary path components.
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', cfg['dawnTag']):
        raise ValueError('Invalid Dawn tag')
    path = root / '.tools/downloads' / f'emdawnwebgpu_pkg-{cfg["dawnTag"]}.zip'
    make_directories(root, path.parent.relative_to(root))
    return path


def finish_dawn(root: Path, requested: str, matched: str, hit: str,
                outcome: str, mode: str) -> dict:
    path = dawn_path(root)
    reason = decision('dawn', requested, matched, hit, outcome, mode)
    accepted = False
    with span('cache.admit', kind='dawn'):
        if reason == 'exact':
            try:
                if file_identity(path)['sha256'] != load_toolchain(root)['dawnSha256']:
                    raise Rejected('archive-digest-mismatch')
                accepted = True
            except (OSError, ValueError):
                reason = 'invalid-archive'
        if not accepted:
            remove_entry(path)
    # setup_toolchain still revalidates and safely extracts the pinned download.
    # An invalid restore is discarded, never a reason to accept another digest.
    return report('dawn', reason, accepted, requested, matched, mode)


def report(kind: str, reason: str, accepted: bool, requested: str, matched: str, mode: str) -> dict:
    values = {'kind': kind, 'state': reason, 'accepted': accepted,
              # A corrupt exact cache is immutable. Don't repeatedly try to save
              # over it; remove that cache through GitHub before repopulating it.
              'saveAllowed': mode == 'enabled' and (matched != requested or accepted),
              'exactHit': accepted and matched == requested}
    emit('cache_admission', **values)
    print_observation('cache_admission', values)
    if not accepted and reason not in ('miss', 'cold'):
        print(f'[cache-admission] {kind}: {reason}; using clean inputs, not restored data.')
    return values


def check_save(root: Path, kind: str, relative: str = '') -> bool:
    """Don't publish a malformed live tree into a trusted shared cache."""
    try:
        with span('cache.check_save', kind=kind):
            if kind == 'em':
                active, _, state = state_paths(root)
                record = read_state(state)
                if record.get('phase') != 'ready':
                    raise Rejected('not-ready')
                inspect_tree(active, record['baseline'])
            elif kind == 'cc':
                inspect_tree(cc_path(root, relative))
            elif kind == 'dawn':
                if file_identity(dawn_path(root))['sha256'] != load_toolchain(root)['dawnSha256']:
                    raise Rejected('digest-mismatch')
            else:
                raise ValueError('Invalid cache kind')
    except (OSError, ValueError, KeyError):
        print(f'[cache-admission] {kind}: save withheld; input validation failed.')
        return False
    return True


def write_output(values: dict) -> None:
    target = os.environ.get('GITHUB_OUTPUT')
    if target:
        with open(target, 'a') as output:
            for key, value in values.items():
                if type(value) is not bool:
                    raise ValueError('Only boolean admission outputs are supported')
                output.write(f'{key}={str(value).lower()}\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('clear-dawn', 'clear-cc', 'park-em', 'finish', 'check-save'))
    parser.add_argument('--kind', choices=('dawn', 'em', 'cc'))
    parser.add_argument('--path', default='')
    parser.add_argument('--requested', default='')
    parser.add_argument('--matched', default='')
    parser.add_argument('--hit', default='')
    parser.add_argument('--outcome', default='')
    parser.add_argument('--mode', choices=('enabled', 'cold'), default='enabled')
    args = parser.parse_args()
    if args.operation in ('finish', 'check-save') and not args.kind:
        parser.error('--kind is required')
    if args.operation == 'clear-dawn':
        remove_entry(dawn_path(ROOT))
    elif args.operation == 'clear-cc':
        clear_cc(ROOT, args.path)
    elif args.operation == 'park-em':
        park_em(ROOT, args.requested)
    elif args.operation == 'check-save':
        write_output({'allowed': check_save(ROOT, args.kind, args.path)})
    else:
        common = (ROOT, args.requested, args.matched, args.hit, args.outcome, args.mode)
        if args.kind == 'em':
            result = finish_em(*common)
        elif args.kind == 'dawn':
            result = finish_dawn(*common)
        else:
            result = finish_cc(ROOT, args.path, *common[1:])
        write_output({'accepted': result['accepted'], 'save-allowed': result['saveAllowed'], 'hit': result['exactHit']})


if __name__ == '__main__':
    main()
