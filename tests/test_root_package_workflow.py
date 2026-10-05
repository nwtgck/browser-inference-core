"""Exercise the workflow's actual final-verification command on tiny packages.

Fixtures contain empty Wasm modules, not compiled inference engines. npm packing
and the shipped JavaScript verifier really run; no GitHub or network is needed.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_multi_runtime as legacy_fixtures
import test_source_package as source_fixtures
from test_pipeline_workflow import JOBS

ROOT = Path(__file__).resolve().parents[1]
VERIFY_STEP = 'Validate the final runtime-only npm payload'


def workflow_step(job, name):
    """Read a named step in the existing, intentionally simple workflow layout.

    Not a general YAML parser: an incompatible layout fails these contract tests
    instead of silently substituting a separately maintained command string.
    """
    matches = [step for step in JOBS[job].split('\n      - ')
               if step.startswith('name: ' + name + '\n')]
    if len(matches) != 1:
        raise AssertionError(f'Expected exactly one {job}/{name} step')
    return matches[0]


def workflow_argv(job, name):
    lines = workflow_step(job, name).splitlines()
    runs = [i for i, line in enumerate(lines) if line.startswith('        run: ')]
    if len(runs) != 1:
        raise AssertionError(f'Expected exactly one command for {job}/{name}')
    at = runs[0]
    command = lines[at].removeprefix('        run: ')
    if command in ('>', '>-'):
        pieces = []
        for line in lines[at + 1:]:
            if not line.startswith('          '):
                break
            pieces.append(line.strip())
        command = ' '.join(pieces)
    argv = shlex.split(command)
    if not argv or argv[0] != 'python3' or not argv[1].startswith('scripts/'):
        raise AssertionError(f'Expected a root Python command: {command!r}')
    return argv


class WorkflowRootArguments(unittest.TestCase):
    def test_assembly_verification_publication_and_report_match_real_parsers(self):
        commands = [
            ('assemble', 'Join exact raw packages and verify every compressed alternative'),
            ('assemble', VERIFY_STEP),
            ('publish', 'Publish append-only artifact commit'),
            ('publish', 'Generate npm lock and consumer integration metadata'),
        ]
        parse_args = argparse.ArgumentParser.parse_args

        class Parsed(Exception):
            pass

        def parse_then_stop(parser, *args, **kwargs):
            # Run the actual parser, then stop before packaging, publishing or
            # resolving a remote npm lock. This test checks arguments only.
            parse_args(parser, *args, **kwargs)
            raise Parsed

        for job, name in commands:
            with self.subTest(job=job, step=name):
                argv = workflow_argv(job, name)
                with patch.object(sys, 'argv', argv[1:]), \
                     patch.object(argparse.ArgumentParser, 'parse_args', parse_then_stop), \
                     self.assertRaises(Parsed):
                    runpy.run_path(str(ROOT / argv[1]), run_name='__main__')

    def test_final_verification_is_mandatory_and_precedes_upload(self):
        job = JOBS['assemble']
        self.assertIn('defaults:\n      run:\n        working-directory: .', job)
        step = workflow_step('assemble', VERIFY_STEP)
        self.assertNotIn('continue-on-error:', step)
        self.assertNotIn('\n        if:', step)
        argv = workflow_argv('assemble', VERIFY_STEP)
        self.assertEqual(argv[:2], ['python3', 'scripts/package_runtime.py'])
        self.assertEqual(argv.count('--verify-only'), 1)
        self.assertEqual(argv.count('--output'), 1)
        self.assertEqual(argv[argv.index('--output') + 1], 'dist/package')
        self.assertNotIn('--defer-npm-pack', argv)
        self.assertNotIn('--inputs', argv)
        self.assertLess(job.index('name: ' + VERIFY_STEP),
                        job.index('uses: actions/upload-artifact@'))
        self.assertIn('name: source-runtime-catalog\n          path: dist/package/', job)


class RootVerificationIntegration(unittest.TestCase):
    def setUp(self):
        fixture = source_fixtures.SourcePackage()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        self.out = self.root / 'dist/package'
        fixture.out = self.out
        self.env = os.environ.copy()
        self.env['BIC_METRICS_MODE'] = 'off'
        self.env['npm_config_cache'] = str(self.root / 'npm-cache')
        self.env['npm_config_offline'] = 'true'
        self.env['npm_config_update_notifier'] = 'false'

    def build(self):
        with redirect_stdout(io.StringIO()):
            return self.fixture.build(packing={'codecs': ['gzip', 'brotli'], 'pairs': []})

    def execute_workflow(self):
        argv = workflow_argv('assemble', VERIFY_STEP)
        # Preserve the YAML arguments and its root-relative output path. Only
        # locate the Python interpreter/script outside the temporary workspace.
        return subprocess.run([sys.executable, str(ROOT / argv[1]), *argv[2:]],
                              cwd=self.root, env=self.env, capture_output=True,
                              text=True, timeout=90)

    def assert_rejected(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertNotEqual(result.returncode, 2, result.stderr)
        self.assertNotIn('unrecognized arguments', result.stderr)
        self.assertNotIn('"sourceCommit":', result.stdout)

    def snapshot(self):
        return {p.relative_to(self.out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.out.rglob('*') if p.is_file()}

    def refresh_manifest(self):
        metadata = json.loads((self.out / 'manifest.json').read_text())
        legacy_fixtures.write_manifest(self.out, metadata)

    def test_exact_workflow_command_verifies_all_packed_alternatives_without_writes(self):
        manifest = self.build()
        catalog = json.loads((self.out / manifest['packedWasm']['catalog']).read_text())
        count = sum(len(t['representations']) for t in catalog['targets'].values())
        before = self.snapshot()
        result = self.execute_workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertIn({'verifiedRepresentations': count, 'targets': len(catalog['targets'])}, output)
        self.assertEqual(output[-1], {'sourceCommit': manifest['sourceCommit']})
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(list(self.out.rglob('*.tgz')))

    def test_exact_workflow_command_retains_legacy_manifest_verification(self):
        legacy_fixtures.package.assemble(self.fixture.fixture.inputs, self.out, check_npm_pack=False)
        before = self.snapshot()
        result = self.execute_workflow()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'sourceCommit': legacy_fixtures.SOURCE})
        self.assertEqual(self.snapshot(), before)

    def test_missing_package_is_not_reassembled(self):
        result = self.execute_workflow()
        self.assert_rejected(result)
        self.assertIn('FileNotFoundError', result.stderr)
        self.assertFalse(self.out.exists())

    def test_raw_tampering_is_rejected_without_rewriting_the_package(self):
        manifest = self.build()
        first = next(iter(manifest['runtimes']['llama-cpp']['sources'].values()))
        root = self.out / Path(first['manifest']).parent
        raw = next(root.glob('profiles/*/browser/core.wasm'))
        raw.write_bytes(b'corrupt raw module')
        before = self.snapshot()
        result = self.execute_workflow()
        self.assert_rejected(result)
        self.assertIn('identity mismatch', result.stderr)
        self.assertEqual(self.snapshot(), before)

    def test_rehashed_but_invalid_compressed_payload_still_runs_the_decoder(self):
        manifest = self.build()
        path = self.out / manifest['packedWasm']['catalog']
        catalog = json.loads(path.read_text())
        key, asset = next((key, a) for key, a in catalog['assets'].items() if a['codec'] == 'gzip')
        data = b'not a gzip stream'
        new_key = hashlib.sha256(data).hexdigest()
        (path.parent / asset['path']).unlink()
        asset.update(path=f'data/{new_key}.gz', bytes=len(data), sha256=new_key)
        (path.parent / asset['path']).write_bytes(data)
        del catalog['assets'][key]
        catalog['assets'][new_key] = asset
        for target in catalog['targets'].values():
            for rep in target['representations']:
                if rep['payload'] == key:
                    rep['payload'] = new_key
        path.write_text(json.dumps(catalog))
        self.refresh_manifest()
        result = self.execute_workflow()
        self.assert_rejected(result)
        self.assertIn('incorrect header check', result.stderr)

    def test_final_verification_does_not_defer_npm_inventory_validation(self):
        self.build()
        path = self.out / 'package.json'
        metadata = json.loads(path.read_text())
        metadata['files'].remove('packed/')
        path.write_text(json.dumps(metadata))
        self.refresh_manifest()
        result = self.execute_workflow()
        self.assert_rejected(result)
        self.assertIn('verifiedRepresentations', result.stdout)
        self.assertIn('npm pack differs from manifest', result.stderr)


if __name__ == '__main__':
    unittest.main()
