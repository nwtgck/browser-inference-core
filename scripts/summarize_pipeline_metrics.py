#!/usr/bin/env python3
"""Summarize bounded, untrusted timing records without executing their contents."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re

from pipeline_metrics import ROOT, MAX_BYTES, MAX_FILES, MAX_RECORD, _open_regular, _warning, emit, print_observation


def union_length(intervals: list[tuple[int, int]]) -> int:
    total = 0; end = None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError('Negative interval')
        total += stop - max(start, end if end is not None else start) if end is None or stop > end else 0
        end = stop if end is None else max(end, stop)
    return total


def ninja_summary(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        return {'status': 'unavailable'}
    rows = defaultdict(list)
    try:
        with _open_regular(path, os.O_RDONLY, 'r') as source:
            if source.readline() != '# ninja log v5\n':
                return {'status': 'unknown-format'}
            for index, line in enumerate(source):
                if index >= 100000 or len(line) > MAX_RECORD:
                    return {'status': 'oversized'}
                values = line.rstrip('\n').split('\t')
                if len(values) != 5:
                    return {'status': 'malformed'}
                start, end = int(values[0]), int(values[1])
                if start < 0 or end < start or end > 7 * 24 * 3600 * 1000:
                    return {'status': 'malformed'}
                output = values[3].replace('\\', '/')
                if output.endswith(('.o', '.obj')):
                    kind = 'object-outputs'
                elif output.endswith(('.a', '.lib')):
                    kind = 'archive-outputs'
                elif '/runtime/core.' in '/' + output:
                    kind = 'runtime-outputs'
                else:
                    kind = 'other-outputs'
                rows[kind].append((start, end))
    except (OSError, UnicodeError, ValueError):
        return {'status': 'malformed'}
    all_intervals = [item for intervals in rows.values() for item in intervals]
    return {'status': 'observed', 'singleFreshInvocationRequired': True,
            # A multi-output edge writes multiple rows. Never call these commands
            # or sum their durations as a build wall time.
            'outputRows': len(all_intervals), 'coveredMs': union_length(all_intervals),
            'categories': {kind: {'outputRows': len(intervals), 'coveredMs': union_length(intervals),
                                  'longestOutputMs': max(end - start for start, end in intervals)}
                           for kind, intervals in rows.items()}}


def read_records(directory: Path) -> tuple[list[dict], int]:
    records = []; errors = int((directory / 'TRUNCATED').exists())
    total = 0
    # Iterate without sorting/materializing an unbounded directory listing.
    for index, path in enumerate(directory.glob('events-*.jsonl')):
        if index >= MAX_FILES:
            errors += 1; break
        if path.is_symlink() or not path.is_file():
            errors += 1; continue
        if total + path.stat().st_size > MAX_BYTES:
            errors += 1; break
        try:
            with _open_regular(path, os.O_RDONLY, 'rb') as source:
                while True:
                    line = source.readline(MAX_RECORD + 1)
                    if not line:
                        break
                    total += len(line)
                    if len(line) > MAX_RECORD or total > MAX_BYTES:
                        errors += 1; break
                    try:
                        item = json.loads(line)
                        if not isinstance(item, dict) or item.get('schemaVersion') != 1:
                            raise ValueError('format')
                        records.append(item)
                    except (ValueError, UnicodeError):
                        errors += 1
        except (OSError, ValueError):
            errors += 1
    return records, errors


def summarize(records: list[dict], errors: int = 0) -> dict:
    phases = defaultdict(lambda: {'started': 0, 'ended': 0, 'failed': 0,
                                  'inclusiveNs': 0, 'exclusiveNs': 0, 'exclusiveUnknown': 0})
    starts = {}; ends = {}; duplicate = set()
    def bucket(item):
        kind = item.get('packageKind')
        return item['phase'] + (':' + kind if kind in ('llama', 'image', 'root') else '')
    safe_phase = re.compile(r'[A-Za-z0-9_.:-]{1,100}')
    for item in records:
        if item.get('event') not in ('span_start', 'span_end'):
            continue
        phase, identifier = item.get('phase'), item.get('spanId')
        if (not isinstance(phase, str) or not safe_phase.fullmatch(phase) or
                not isinstance(identifier, str) or not re.fullmatch('[0-9a-f]{32}', identifier) or
                type(item.get('startedNs')) is not int or item['startedNs'] < 0):
            errors += 1; continue
        target = starts if item['event'] == 'span_start' else ends
        if identifier in target:
            duplicate.add(identifier); errors += 1
        target[identifier] = item
    children_by_parent = defaultdict(list)
    for identifier, item in starts.items():
        phases[bucket(item)]['started'] += 1
        parent = item.get('parentSpanId')
        if isinstance(parent, str):
            children_by_parent[parent].append(item)
    for identifier, item in ends.items():
        start = starts.get(identifier)
        elapsed = item.get('elapsedNs')
        if (identifier in duplicate or start is None or
                any(start.get(key) != item.get(key) for key in
                    ('phase', 'packageKind', 'startedNs', 'parentSpanId', 'runId', 'runAttempt', 'job', 'sourceCommit', 'runtime', 'profile', 'variant')) or
                type(elapsed) is not int or elapsed < 0 or elapsed > 7 * 24 * 3600 * 10**9 or
                item.get('result') not in ('success', 'failure', 'cancelled')):
            errors += 1; continue
        entry = phases[bucket(item)]
        entry['ended'] += 1; entry['failed'] += item['result'] != 'success'
        entry['inclusiveNs'] += elapsed
        children = children_by_parent[identifier]
        if any(child['spanId'] not in ends or child['spanId'] in duplicate for child in children):
            entry['exclusiveUnknown'] += 1
            continue
        intervals = []
        bad = False
        for child in children:
            duration = ends[child['spanId']].get('elapsedNs')
            if type(duration) is not int or duration < 0:
                bad = True; break
            begin = max(item['startedNs'], child['startedNs'])
            stop = min(item['startedNs'] + elapsed, child['startedNs'] + duration)
            if stop >= begin:
                intervals.append((begin, stop))
        if bad:
            entry['exclusiveUnknown'] += 1
        else:
            entry['exclusiveNs'] += elapsed - union_length(intervals)
    for entry in phases.values():
        entry['incomplete'] = entry['started'] - entry['ended']
        if not entry['ended']:
            entry['inclusiveNs'] = None
        if not entry['ended'] or entry['exclusiveUnknown'] or errors:
            entry['exclusiveNs'] = None
    return {'schemaVersion': 1, 'invalidOrMissingRecords': errors,
            'note': 'Phase sums are cumulative inclusive work, NOT workflow wall time. Parent/child spans overlap.',
            'phases': dict(phases)}


def console_summary(report: dict) -> None:
    # report is generated by summarize(), whose phase names and numbers were
    # validated. Fixed prefixes prevent workflow-command injection in logs.
    rows = sorted(report['phases'].items())
    print_observation('summary', {'invalidOrMissingRecords': report['invalidOrMissingRecords'],
                      'phases': len(rows), 'note': 'Inclusive phase times overlap; do not sum as workflow wall time.'})
    for name, values in rows[:100]:
        print_observation('phase_summary', {'phase': name, **{
            key: values[key] for key in ('started', 'ended', 'failed', 'incomplete')},
            'inclusiveSeconds': None if values['inclusiveNs'] is None else round(values['inclusiveNs'] / 10**9, 6),
            'exclusiveSeconds': None if values['exclusiveNs'] is None else round(values['exclusiveNs'] / 10**9, 6)})
    if len(rows) > 100:
        print_observation('summary_truncated', {'remainingPhases': len(rows) - 100})


def write_summary(path: Path, report: dict) -> None:
    # Open without O_TRUNC: a hard link must be rejected BEFORE truncation.
    with _open_regular(path, os.O_WRONLY | os.O_CREAT, 'w') as output:
        output.truncate(0)
        json.dump(report, output, indent=2)
        output.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--ninja', type=Path)
    args = parser.parse_args()
    try:
        if args.ninja:
            values = ninja_summary(args.ninja)
            emit('ninja_observation', **values)
            print_observation('ninja_observation', values)
        if args.directory:
            if args.directory.is_symlink() or args.directory.resolve().is_relative_to(ROOT):
                raise ValueError('metrics summary must stay outside the source tree')
            records, errors = read_records(args.directory)
            report = summarize(records, errors)
            args.directory.mkdir(parents=True, exist_ok=True)
            write_summary(args.directory / 'summary.json', report)
            console_summary(report)
    except Exception:
        _warning()  # Optional observer failures must not change build status.


if __name__ == '__main__':
    main()
