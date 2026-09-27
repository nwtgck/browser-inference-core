"""Observer regressions: real subprocess exit/signal/stream behavior, no compiler."""
import json
import io
from contextlib import redirect_stdout
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pipeline_metrics as metrics
from summarize_pipeline_metrics import console_summary, ninja_summary, read_records, summarize, union_length, write_summary


class PipelineMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.directory = self.root / 'metrics'
        self.env = dict(os.environ, BIC_METRICS_MODE='basic', BIC_METRICS_DIR=str(self.directory))

    def invoke(self, code, *, mode='basic', directory=None):
        env = dict(self.env, BIC_METRICS_MODE=mode)
        if directory: env['BIC_METRICS_DIR'] = str(directory)
        return subprocess.run([sys.executable, str(ROOT / 'scripts/pipeline_metrics.py'),
            'run', '--phase', 'test.command', '--', sys.executable, '-c', code],
            env=env, capture_output=True, timeout=8)

    def test_off_has_no_metrics_and_child_streams_are_unmodified(self):
        result = self.invoke('import sys; print("out"); print("err",file=sys.stderr);sys.exit(7)', mode='off')
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, b'out\n'); self.assertEqual(result.stderr, b'err\n')
        self.assertFalse(self.directory.exists())

    def test_enabled_preserves_stdout_stderr_and_exit(self):
        result = self.invoke('import sys;print("commit");print("diagnostic",file=sys.stderr);sys.exit(7)')
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, b'commit\n'); self.assertEqual(result.stderr, b'diagnostic\n')
        records, errors = read_records(self.directory)
        end = next(item for item in records if item['event'] == 'span_end')
        self.assertEqual(end['exitCode'], 7); self.assertEqual(end['result'], 'failure')
        self.assertGreater(end['elapsedNs'], 0); self.assertEqual(errors, 0)

    def test_logger_failure_does_not_replace_child_exit(self):
        path = self.root / 'not-a-directory'; path.write_text('fixture')
        result = self.invoke('import sys;print("commit");sys.exit(7)', directory=path)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, b'commit\n')
        self.assertIn(b'Optional telemetry', result.stderr)

    def test_no_environment_or_command_argument_dump(self):
        self.env['GH_TOKEN'] = 'SECRET-NOT-FOR-LOG'
        self.invoke('argument_secret="ARGUMENT-SECRET";print("child")')
        content = ''.join(p.read_text() for p in self.directory.glob('*.jsonl'))
        self.assertNotIn('SECRET-NOT-FOR-LOG', content)
        self.assertNotIn('ARGUMENT-SECRET', content)
        self.assertNotIn('python', content)

    def test_large_output_streams_without_capture_in_wrapper(self):
        result = self.invoke('import sys;sys.stdout.write("x"*1048576)')
        self.assertEqual(result.returncode, 0); self.assertEqual(len(result.stdout), 1048576)

    @unittest.skipUnless(os.name == 'posix', 'POSIX signal contract')
    def test_child_signal_is_preserved(self):
        result = self.invoke('import os,signal;os.kill(os.getpid(),signal.SIGTERM)')
        self.assertEqual(result.returncode, -signal.SIGTERM)
        records, _ = read_records(self.directory)
        self.assertEqual(records[-1]['result'], 'cancelled')

    @unittest.skipUnless(os.name == 'posix', 'POSIX process-group contract')
    def test_wrapper_cancellation_reaches_child_and_preserves_signal(self):
        ready, stopped = self.root / 'ready', self.root / 'stopped'
        code = (f'import signal,time,pathlib,sys\n'
                f'def stop(*_):\n pathlib.Path({str(stopped)!r}).write_text("stopped");sys.exit(0)\n'
                f'signal.signal(signal.SIGTERM,stop)\npathlib.Path({str(ready)!r}).touch()\ntime.sleep(20)\n')
        process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/pipeline_metrics.py'),
            'run', '--phase', 'test.cancel', '--', sys.executable, '-c', code], env=self.env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline: time.sleep(0.01)
            self.assertTrue(ready.exists())
            process.terminate(); process.communicate(timeout=5)
            self.assertEqual(process.returncode, -signal.SIGTERM)
            self.assertTrue(stopped.exists())
        finally:
            if process.poll() is None: process.kill(); process.communicate()

    def test_nested_spans_keep_exception_and_parent_link(self):
        with patch.dict(os.environ, self.env):
            with self.assertRaisesRegex(ValueError, 'original'):
                with metrics.span('outer'):
                    with metrics.span('inner'): raise ValueError('original')
        records, errors = read_records(self.directory)
        start = next(item for item in records if item['phase'] == 'outer' and item['event'] == 'span_start')
        end = next(item for item in records if item['phase'] == 'inner' and item['event'] == 'span_end')
        self.assertEqual(end['parentSpanId'], start['spanId']); self.assertEqual(errors, 0)
        self.assertEqual(end['result'], 'failure')

    def test_budget_is_bounded_and_explicitly_truncated(self):
        with patch.dict(os.environ, self.env), patch.object(metrics, 'MAX_BYTES', 1024):
            for _ in range(20):
                with metrics.span('limited'): pass
        self.assertLessEqual(sum(p.stat().st_size for p in self.directory.glob('*.jsonl')), 1024)
        self.assertTrue((self.directory / 'TRUNCATED').exists())
        self.assertGreater(read_records(self.directory)[1], 0)

    def test_requested_and_matched_keys_do_not_turn_false_into_true(self):
        prefix = 'bic-v1-cc-partition-'
        self.assertEqual(metrics.cache_state('cc', prefix+'a'*40, prefix+'b'*40, 'false', 'success')['state'], 'prefix')
        self.assertEqual(metrics.cache_state('em', 'expected', 'other', 'false', 'success')['state'], 'unexpected-key-or-status')
        self.assertEqual(metrics.cache_state('em', 'expected', '', '', 'success')['state'], 'miss')
        self.assertEqual(metrics.cache_state('em', 'expected', 'expected', 'true', 'success')['state'], 'exact')
        self.assertEqual(metrics.cache_state('em', 'expected', '', '', 'failure')['state'], 'not-completed')

    def test_source_directory_cannot_be_used_as_telemetry_output(self):
        result = self.invoke('print("ok")', directory=ROOT / 'forbidden-metrics-fixture')
        self.assertEqual(result.returncode, 0)
        self.assertFalse((ROOT / 'forbidden-metrics-fixture').exists())


    def test_fifo_at_writer_lock_cannot_block_command(self):
        self.directory.mkdir(); os.mkfifo(self.directory / '.writer.lock')
        result = self.invoke('import sys;print("still ran");sys.exit(7)')
        self.assertEqual(result.returncode, 7); self.assertEqual(result.stdout, b'still ran\n')

    def test_fifo_at_event_file_does_not_block_observer(self):
        self.directory.mkdir(); os.mkfifo(self.directory / metrics._FILE)
        with patch.dict(os.environ, self.env):
            metrics.emit('probe')
        self.assertTrue((self.directory / metrics._FILE).exists())

    def test_hard_linked_log_is_not_modified(self):
        self.directory.mkdir(); victim = self.root / 'untouched'; victim.write_text('original')
        os.link(victim, self.directory / metrics._FILE)
        with patch.dict(os.environ, self.env): metrics.emit('probe')
        self.assertEqual(victim.read_text(), 'original')

    def test_summary_rejects_hardlink_and_fifo_before_write(self):
        victim = self.root / 'untouched'; victim.write_text('original')
        target = self.root / 'summary.json'; os.link(victim, target)
        with self.assertRaises(ValueError): write_summary(target, {'fixture': 1})
        self.assertEqual(victim.read_text(), 'original')
        target.unlink(); os.mkfifo(target)
        with self.assertRaises((OSError, ValueError)): write_summary(target, {'fixture': 1})

    def test_read_records_rejects_hardlinks(self):
        victim = self.root / 'outside.jsonl'; victim.write_text('{"schemaVersion":1}\n')
        self.directory.mkdir(); os.link(victim, self.directory / 'events-linked.jsonl')
        records, errors = read_records(self.directory)
        self.assertEqual(records, []); self.assertEqual(errors, 1)

    def test_sigkill_exit_is_preserved_without_installing_a_sigkill_handler(self):
        result = self.invoke('import os, signal;os.kill(os.getpid(), signal.SIGKILL)')
        self.assertEqual(result.returncode, -signal.SIGKILL)
        self.assertNotIn(b'Traceback', result.stderr)

    def test_cancellation_that_arrives_during_spawn_is_not_lost(self):
        code = f"""import sys, os, signal
sys.path.insert(0, {str(ROOT / 'scripts')!r})
import pipeline_metrics as m
real = m.subprocess.Popen
def spawn(*a, **kw):
    os.kill(os.getpid(), signal.SIGTERM)
    return real(*a, **kw)
m.subprocess.Popen = spawn
m.exit_like_child(m.run_command([sys.executable, '-c', 'import time; time.sleep(30)'], 'test.spawn'))
"""
        result = subprocess.run([sys.executable, '-c', code], env=self.env, capture_output=True, timeout=8)
        self.assertEqual(result.returncode, -signal.SIGTERM)

    def test_cancellation_kills_ignoring_children_with_a_bounded_grace(self):
        ready = self.root / 'ready'
        child = ('import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                 f'pathlib.Path({str(ready)!r}).touch();time.sleep(30)')
        process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/pipeline_metrics.py'),
                    'run', '--phase', 'test.ignore', '--', sys.executable, '-c', child],
                    env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 6
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready.exists())
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=6)
            self.assertEqual(process.returncode, -signal.SIGTERM)
        finally:
            if process.poll() is None:
                process.kill(); process.communicate(timeout=2)


    def test_parent_handling_term_does_not_leave_an_ignoring_grandchild(self):
        ready = self.root / 'grandchild-ready'
        grandchild = ('import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                      f'pathlib.Path({str(ready)!r}).touch();time.sleep(30)')
        child = ('import signal,sys,subprocess,time;'
                 'signal.signal(signal.SIGTERM,lambda *args:sys.exit(0));'
                 f'subprocess.Popen([sys.executable,"-c",{grandchild!r}]);time.sleep(30)')
        process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/pipeline_metrics.py'),
                    'run', '--phase', 'test.grandchild', '--', sys.executable, '-c', child],
                    env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 6
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready.exists())
            process.send_signal(signal.SIGTERM)
            # communicate also waits for the inherited pipes held by the grandchild.
            process.communicate(timeout=6)
            self.assertEqual(process.returncode, -signal.SIGTERM)
        finally:
            if process.poll() is None:
                process.kill(); process.communicate(timeout=2)


class Summaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def ninja(self, body):
        file = self.root / '.ninja_log'; file.write_text(body)
        return ninja_summary(file)

    def test_ninja_multi_output_rows_do_not_double_wall_coverage(self):
        data = self.ninja('# ninja log v5\n0\t8\t0\ta.o\taa\n0\t8\t0\tb.o\tbb\n8\t18\t0\truntime/core.mjs\tcc\n8\t18\t0\truntime/core.wasm\tcc\n')
        self.assertEqual(data['outputRows'], 4); self.assertEqual(data['coveredMs'], 18)
        self.assertEqual(data['categories']['runtime-outputs']['coveredMs'], 10)
        self.assertNotIn('commandCount', data)

    def test_ninja_unknown_malformed_or_missing_are_unknown_not_zero(self):
        for body in ('# ninja log v99\n', '# ninja log v5\n100\t1\t0\tx\tcc\n', '# ninja log v5\ninvalid\n'):
            value = self.ninja(body)
            self.assertNotEqual(value['status'], 'observed'); self.assertNotIn('coveredMs', value)
        self.assertEqual(ninja_summary(self.root / 'missing'), {'status': 'unavailable'})

    def test_parallel_spans_subtract_union_not_sum(self):
        self.assertEqual(union_length([(2,7),(4,9)]), 7)
        def pair(identifier, parent, start, length, phase):
            base = {'spanId': identifier*32, 'parentSpanId': parent*32 if parent else None,
                    'startedNs': start, 'phase': phase}
            return [{**base,'event':'span_start'}, {**base,'event':'span_end','elapsedNs':length,'result':'success'}]
        data = pair('a', None, 0, 10, 'parent') + pair('b','a',2,5,'child') + pair('c','a',4,5,'child')
        report = summarize(data)
        self.assertEqual(report['phases']['parent']['exclusiveNs'], 3)
        self.assertEqual(report['phases']['child']['inclusiveNs'], 10)

    def test_unfinished_span_is_not_success_or_zero_time(self):
        report = summarize([{'event':'span_start','spanId':'a'*32,'phase':'unfinished','startedNs':1}])
        phase = report['phases']['unfinished']
        self.assertEqual(phase['incomplete'], 1); self.assertIsNone(phase['inclusiveNs'])
        self.assertIsNone(phase['exclusiveNs'])

    def test_npm_packages_are_reported_separately(self):
        records = []
        for kind, letter in [('llama','a'),('image','b'),('root','c')]:
            common = {'phase':'package.npm_pack','packageKind':kind,'startedNs':1,'spanId':letter*32}
            records += [{**common,'event':'span_start'}, {**common,'event':'span_end','elapsedNs':10,'result':'success'}]
        self.assertEqual(set(summarize(records)['phases']), {'package.npm_pack:'+k for k in ('llama','image','root')})

    def test_console_summary_reports_unknown_and_counts_without_raw_inputs(self):
        report = summarize([{'event':'span_start','spanId':'a'*32,'phase':'package.npm_pack',
                             'startedNs':1, 'packageKind':'image'}])
        output = io.StringIO()
        with patch.dict(os.environ, {'BIC_METRICS_MODE':'basic'}), redirect_stdout(output):
            console_summary(report)
        lines = output.getvalue().splitlines()
        self.assertTrue(all(line.startswith('[pipeline-metrics] ') for line in lines))
        data = json.loads(lines[-1].removeprefix('[pipeline-metrics] '))
        self.assertEqual(data['phase'], 'package.npm_pack:image')
        self.assertIsNone(data['inclusiveSeconds'])
        self.assertEqual(data['incomplete'], 1)

    def test_summary_refuses_source_tree_output(self):
        import summarize_pipeline_metrics as collector
        with patch.object(collector, 'ROOT', self.root), patch.object(sys, 'argv',
                ['summary', '--directory', str(self.root / 'inside-source')]):
            collector.main()
        self.assertFalse((self.root / 'inside-source').exists())

    def test_broken_and_linked_records_are_rejected_without_execution(self):
        (self.root / 'events-1.jsonl').write_text('{broken\n{"schemaVersion":999}\n')
        (self.root / 'events-link.jsonl').symlink_to(self.root / 'events-1.jsonl')
        records, errors = read_records(self.root)
        self.assertEqual(records, []); self.assertEqual(errors, 3)



if __name__ == '__main__': unittest.main()
