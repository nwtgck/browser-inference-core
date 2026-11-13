"""Test-only orchestration of offline publication and the actual inline YAML shell.

Real npm packing and local Git remotes are used. No Docker isolation, GitHub
push, registry acquisition or real-model inference is claimed by this harness.
"""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tests.support import runtime_package as fixtures
from tests.support.workflow import jobs, step_name, shell_body
from build_tools.project import REPOSITORY_ROOT as ROOT
from build_tools.packaging import assemble as package
from build_tools.publication import artifacts, consumer as reporter

run = artifacts.run
verify_git_tree = artifacts.verify_git_tree
source_commit_subject = artifacts.source_commit_subject
publisher = None  # Set to this test-only module below, never production transport.


def publish(package: Path, remote: str, branch='artifacts', attempts=20, *,
            expected_manifest_sha256=None, validator=None, source_repo=None, git_mode=None):
    with tempfile.TemporaryDirectory(prefix='bicore-publish-fixture-') as temporary:
        workspace = Path(temporary)
        work = workspace / 'build/publication/package'
        receipt = workspace / 'build/publication/frozen.json'
        # Only the aggregate validator is production behavior. The optional
        # standalone validator is explicit for existing distribution fixtures.
        validator = artifacts.validate if validator is None else validator
        with patch.object(artifacts, 'run', run), patch.object(artifacts, 'verify_git_tree', verify_git_tree), \
             patch.object(artifacts, 'validate', validator), patch.dict(os.environ, {'GITHUB_OUTPUT': ''}):
            frozen = artifacts.freeze(package, work, receipt, branch=branch,
                                      expected_manifest_sha256=expected_manifest_sha256)
        output = workspace / 'push-output'
        script = shell_body(step_name(jobs((ROOT / '.github/workflows/build.yml').read_text())['publish'],
                                      'Publish append-only artifact commit'))
        env = {**os.environ, 'GITHUB_WORKSPACE': str(workspace), 'RUNNER_TEMP': str(workspace),
               'GITHUB_OUTPUT': str(output), 'GH_TOKEN': 'local-fixture-token',
               'PUBLISH_REMOTE': remote, 'ARTIFACT_BRANCH': branch, 'PUBLISH_ATTEMPTS': str(attempts),
               'FROZEN_TREE': frozen['tree'], 'FROZEN_SOURCE': frozen['sourceCommit'],
               'GIT_ALLOW_PROTOCOL': 'file', 'GIT_TERMINAL_PROMPT': '0'}
        if git_mode:
            bindir = workspace / 'fixture-bin'; bindir.mkdir()
            real_git = shutil.which('git')
            fixture = ROOT / 'tests/support/publication_git_fixture.py'
            (bindir / 'git').write_text('#!' + sys.executable + '\n' + fixture.read_text())
            (bindir / 'git').chmod(0o755)
            (bindir / 'sleep').write_text('#!/bin/sh\nexit 0\n'); (bindir / 'sleep').chmod(0o755)
            env.update({'PATH': str(bindir) + os.pathsep + env['PATH'], 'FIXTURE_REAL_GIT': real_git,
                        'FIXTURE_GIT_MODE': git_mode, 'FIXTURE_STATE': str(workspace / 'fixture-state')})
        result = subprocess.run(['bash', '-euo', 'pipefail', '-c', script], cwd=workspace,
                                env=env, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        values = dict(line.split('=', 1) for line in output.read_text().splitlines())
        commit = values['commit']
        with patch.object(artifacts, 'source_commit_subject', source_commit_subject):
            artifacts.confirm(work, receipt, commit, source_repo)
        return commit


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Local test-only publication adapter')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--remote', required=True)
    parser.add_argument('--branch', default='artifacts')
    parser.add_argument('--source-repo', type=Path)
    args = parser.parse_args()
    print(publish(args.package.resolve(), args.remote, args.branch, source_repo=args.source_repo))


publisher = sys.modules[__name__]


def git(*args, cwd):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def tree(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob('*') if p.is_file()}


def pack_calls(mock):
    return [call for call in mock.call_args_list
            if call.args[0][:3] == ['npm', 'pack', '--dry-run']]


class PublicationFixture(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RuntimePackageFixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root, self.inputs, self.out = self.fixture.root, self.fixture.inputs, self.fixture.out
        self.remote = self.root / 'remote.git'
        git('init', '--bare', '--quiet', str(self.remote), cwd=self.root)
        self.env = {'GITHUB_REPOSITORY': 'example/lcore', 'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2',
                    'GITHUB_STEP_SUMMARY': str(self.root / 'test-step-summary')}


    def assemble(self):
        package.assemble(self.inputs, self.out, check_npm_pack=False)


    def report_main(self, *, digest=None, lock_effect=None, commit='d' * 40):
        args = ['consumer_metadata.py', '--package', str(self.out), '--commit', commit,
                '--lock-directory', str(self.root / 'lock'),
                '--output', str(self.root / 'report')]
        if digest is not None:
            args += ['--published-manifest-sha256', digest]
        lock = {'specifier': 'github:example/lcore#' + commit, 'packageEntry': {'version': '0.1.0'}}
        with patch.object(sys, 'argv', args), patch.dict(os.environ, self.env), \
             patch.object(reporter.npm_lock, 'read', side_effect=lock_effect, return_value=lock) as resolve, \
             patch.object(reporter, 'collect', return_value={'baseCommit': 'b' * 40}), redirect_stdout(io.StringIO()):
            reporter.main()
        return resolve
