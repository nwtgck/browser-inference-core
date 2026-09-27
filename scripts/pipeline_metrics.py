#!/usr/bin/env python3
"""Best-effort, bounded pipeline timings. Never log command arguments or credentials."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
_PARENT: ContextVar[str | None] = ContextVar('metric_parent', default=None)
_FILE = f'events-{os.getpid()}-{uuid.uuid4().hex}.jsonl'
_WARNED = False
MAX_BYTES = 2 * 1024 * 1024
MAX_FILES = 128
MAX_RECORD = 16 * 1024
CANCEL_GRACE_SECONDS = 2.0
_FIELDS = {'runtime', 'profile', 'variant', 'packageKind', 'context', 'attempt',
           'bytes', 'files', 'kind', 'requestedKey', 'matchedKey', 'hit', 'outcome',
           'state', 'saveEligible', 'cacheMode', 'emCache', 'compilerWrapper',
           'ccacheDirectory', 'ccacheConfiguration', 'compilerIdentity',
           'cacheUseMode', 'accepted', 'saveAllowed', 'exactHit'}
_LABEL = re.compile(r'[A-Za-z0-9_.:-]{1,600}')


def enabled() -> bool:
    return os.environ.get('BIC_METRICS_MODE', 'off') == 'basic'


def _warning() -> None:
    global _WARNED
    if not _WARNED:
        _WARNED = True
        try:
            print('[pipeline-metrics] Optional telemetry unavailable or truncated.', file=sys.stderr)
        except Exception:
            pass


def _label(value: str | None) -> str | None:
    return value if isinstance(value, str) and _LABEL.fullmatch(value) else None


def context() -> dict:
    return {'runId': _label(os.environ.get('GITHUB_RUN_ID')),
            'runAttempt': _label(os.environ.get('GITHUB_RUN_ATTEMPT')),
            'job': _label(os.environ.get('GITHUB_JOB')),
            'sourceCommit': _label(os.environ.get('LCB_SOURCE_COMMIT')),
            'runtime': _label(os.environ.get('BIC_METRICS_RUNTIME')),
            'profile': _label(os.environ.get('BIC_METRICS_PROFILE')),
            'variant': _label(os.environ.get('BIC_METRICS_VARIANT'))}


def _fields(values: dict) -> dict:
    return {key: value for key, value in values.items() if key in _FIELDS and
            (value is None or type(value) in (int, bool) or _label(value) is not None)}


def _open_regular(path: Path, flags: int, mode: str):
    # A FIFO or hard link at an optional log path must not block useful work or
    # redirect writes into another file. Check the opened fd, not just the name.
    fd = os.open(path, flags | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0), 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('not a private regular metrics file')
        return os.fdopen(fd, mode)
    except BaseException:
        os.close(fd)
        raise


def emit(event: str, **values) -> None:
    if not enabled():
        return
    try:
        # Call sites supply fixed fields only. No process argv, environment dump,
        # exception message, repository URL or arbitrary test result is recorded.
        record = {'schemaVersion': 1, 'event': event, 'mode': 'basic',
                  'clock': 'monotonic_ns', **context(), **values}
        raw = (json.dumps(record, ensure_ascii=True, separators=(',', ':')) + '\n').encode()
        if len(raw) > MAX_RECORD:
            raise ValueError('oversized metric')
        directory = Path(os.environ.get('BIC_METRICS_DIR') or str(Path(os.environ['RUNNER_TEMP']) / 'bic-pipeline-metrics')).absolute()
        if directory.resolve().is_relative_to(ROOT) or directory.is_symlink():
            raise ValueError('metrics must stay outside the source tree')
        directory.mkdir(parents=True, exist_ok=True)
        # Only the optional writer uses this lock, never a build subprocess.
        # Do not block useful work on a competing observer.
        import fcntl
        flags = os.O_WRONLY | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
        with _open_regular(directory / '.writer.lock', flags, 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            paths = []
            for path in directory.glob('events-*.jsonl'):
                paths.append(path)
                if len(paths) > MAX_FILES:
                    raise ValueError('too many metric files')
            target = directory / _FILE
            if ((not target.exists() and len(paths) >= MAX_FILES) or
                    sum(p.lstat().st_size for p in paths) + len(raw) > MAX_BYTES):
                # An explicit marker lets a summary distinguish missing from zero.
                with _open_regular(directory / 'TRUNCATED', flags, 'w'):
                    pass
                raise ValueError('metric budget exceeded')
            with _open_regular(target, flags | os.O_APPEND, 'ab') as output:
                output.write(raw)
    except Exception:
        _warning()


@contextmanager
def span(phase: str, **fields):
    """Yield mutable numeric/status fields; preserve values and all exceptions."""
    if not enabled():
        yield fields
        return
    identifier = uuid.uuid4().hex
    parent = _PARENT.get() or _label(os.environ.get('BIC_METRICS_PARENT'))
    started = time.perf_counter_ns()
    token = _PARENT.set(identifier)
    details = {'phase': _label(phase) or 'unknown', 'spanId': identifier,
               'parentSpanId': parent, 'startedNs': started}
    emit('span_start', **details, **_fields(fields))
    result, exit_code = 'success', 0
    try:
        yield fields
    except BaseException as error:
        result = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failure'
        exit_code = error.returncode if isinstance(error, subprocess.CalledProcessError) else None
        raise
    finally:
        elapsed = max(0, time.perf_counter_ns() - started)
        _PARENT.reset(token)
        # Exit status supplied by a command is not confused with logger status.
        status = fields.get('exitCode', exit_code)
        if type(status) is int and status != 0:
            result = 'cancelled' if status < 0 else 'failure'
        emit('span_end', **details, elapsedNs=elapsed, result=result,
             exitCode=status, **_fields(fields))


def measured(phase: str, **fields):
    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with span(phase, **fields):
                return function(*args, **kwargs)
        return wrapper
    return decorate


def run_command(command: list[str], phase: str, *, cwd: Path | None = None,
                env: dict | None = None) -> int:
    """Stream output, preserve status, and reap a cancelled process group."""
    with span(phase) as timing:
        environment = dict(os.environ if env is None else env)
        if _PARENT.get():
            environment['BIC_METRICS_PARENT'] = _PARENT.get()
        child = None
        handlers, received = {}, []
        cancelled_at = None
        def send(number):
            if child is not None:
                try:
                    if os.name == 'posix':
                        os.killpg(child.pid, number)
                    else:
                        child.send_signal(number)
                except ProcessLookupError:
                    pass
        def forward(number, _frame):
            nonlocal cancelled_at
            if cancelled_at is None:
                cancelled_at = time.monotonic()
            received.append(number)
            send(number)
        try:
            # Install before spawning, so a cancellation in the creation window
            # is queued and sent as soon as a child process exists.
            if threading.current_thread() is threading.main_thread():
                for number in (signal.SIGINT, signal.SIGTERM):
                    handlers[number] = signal.signal(number, forward)
            try:
                child = subprocess.Popen(command, cwd=cwd, env=environment,
                                         start_new_session=(os.name == 'posix'))
            except FileNotFoundError:
                timing['exitCode'] = -received[0] if received else 127
                return timing['exitCode']
            except OSError:
                timing['exitCode'] = -received[0] if received else 126
                return timing['exitCode']
            for number in received:
                send(number)
            while True:
                try:
                    code = child.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    if cancelled_at is not None and time.monotonic() - cancelled_at >= CANCEL_GRACE_SECONDS:
                        send(signal.SIGKILL if os.name == 'posix' else signal.SIGTERM)
                        code = child.wait()
                        break
            if received:
                # Children that ignore cancellation must not outlive a failed
                # wrapper, even if their parent handled TERM and exited normally.
                if os.name == 'posix':
                    send(signal.SIGKILL)
                code = -received[0]
            timing['exitCode'] = code
            return code
        finally:
            for number, previous in handlers.items():
                signal.signal(number, previous)


def exit_like_child(code: int) -> None:
    if code < 0 and os.name == 'posix':
        if -code not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(-code, signal.SIG_DFL)
        os.kill(os.getpid(), -code)
    raise SystemExit(code if code >= 0 else 128 - code)


def cache_state(kind: str, requested: str, matched: str, hit: str, outcome: str,
                *, save_eligible: bool = False) -> dict:
    # Observation only. This does NOT grant permission to adopt or save data.
    if outcome != 'success':
        state = 'not-completed'
    elif not matched:
        state = 'miss' if not hit else 'inconsistent'
    elif matched == requested and hit == 'true':
        state = 'exact'
    elif (kind == 'cc' and hit == 'false' and len(requested) > 40 and
          re.fullmatch('[0-9a-f]{40}', requested[-40:]) and
          matched.startswith(requested[:-40]) and
          re.fullmatch('[0-9a-f]{40}', matched[len(requested)-40:])):
        state = 'prefix'
    else:
        state = 'unexpected-key-or-status'
    return _fields({'kind': kind, 'requestedKey': requested or None,
                    'matchedKey': matched or None, 'hit': hit or None,
                    'outcome': outcome or None, 'state': state,
                    'saveEligible': save_eligible,
                    'cacheMode': os.environ.get('ACTIONS_CACHE_MODE')})


def observe_environment() -> dict:
    def location(name, expected):
        value = os.environ.get(name)
        if not value:
            return 'unset'
        return 'expected' if Path(value).resolve() == expected.resolve() else 'different'
    return {'cacheUseMode': _label(os.environ.get('BIC_CACHE_MODE', 'enabled')),
            'emCache': location('EM_CACHE', ROOT / '.tools/emsdk/upstream/emscripten/cache'),
            'compilerWrapper': 'ccache' if os.environ.get('EM_COMPILER_WRAPPER') == 'ccache' else 'unset-or-different',
            'ccacheDirectory': 'set' if os.environ.get('CCACHE_DIR') else 'unset',
            'ccacheConfiguration': location('CCACHE_CONFIGPATH', ROOT / '.tools/ccache.conf'),
            'compilerIdentity': _label(os.environ.get('BIC_CACHE_ID'))}


def print_observation(event: str, values: dict) -> None:
    """Keep essential observations in the downloadable Actions logs as well."""
    if enabled():
        try:
            print('[pipeline-metrics] ' + json.dumps({'event': event, **values},
                                                     separators=(',', ':')))
        except Exception:
            _warning()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    run = commands.add_parser('run')
    run.add_argument('--phase', required=True)
    run.add_argument('command', nargs=argparse.REMAINDER)
    cache = commands.add_parser('cache')
    cache.add_argument('--kind', choices=('dawn', 'em', 'cc'), required=True)
    cache.add_argument('--requested', default='')
    cache.add_argument('--matched', default='')
    cache.add_argument('--hit', default='')
    cache.add_argument('--outcome', default='')
    cache.add_argument('--save-eligible', choices=('true', 'false'), default='false')
    saved = commands.add_parser('cache-save')
    saved.add_argument('--kind', choices=('dawn', 'em', 'cc'), required=True)
    saved.add_argument('--outcome', default='')
    commands.add_parser('environment')
    args = parser.parse_args()
    if args.operation == 'run':
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            parser.error('a command is required')
        exit_like_child(run_command(command, args.phase))
    elif args.operation == 'cache-save':
        # A successful action can include an immutable-key conflict or a service
        # warning; it is not proof of a new durable entry. Keep that distinction.
        values = _fields({'kind': args.kind, 'outcome': args.outcome or None,
                          'state': 'action-completed-not-server-proof' if args.outcome == 'success' else 'not-confirmed'})
        emit('cache_save_observation', **values)
        print_observation('cache_save_observation', values)
    elif args.operation == 'cache':
        values = cache_state(args.kind, args.requested, args.matched,
                             args.hit, args.outcome, save_eligible=args.save_eligible == 'true')
        emit('cache_observation', **values)
        print_observation('cache_observation', values)
    else:
        try:
            values = observe_environment()
            emit('environment_observation', **values)
            print_observation('environment_observation', values)
        except Exception:
            _warning()


if __name__ == '__main__':
    main()
