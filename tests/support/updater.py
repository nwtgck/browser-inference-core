"""Real local source/remote fixtures for offline updater and literal YAML tests."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import REPOSITORY_ROOT
from build_tools.runtimes.llama_cpp import update
from tests.runtimes.llama_cpp.mtmd_fixture_support import CheckedMtmdFixture
from tests.support.workflow import jobs, step_name, shell_body, docker_standin


class UpdaterFixture(unittest.TestCase):
    def setUp(self):
        for target, name in ((update.webgpu_source, 'prepare'), (update.qsa, 'preflight')):
            mock = patch.object(target, name)
            value = mock.start(); self.addCleanup(mock.stop)
            if name == 'prepare': self.webgpu_prepare = value
            else: self.qsa_preflight = value
        model_graph = patch.object(update.model_graph, 'prepare')
        self.model_graph_prepare = model_graph.start()
        self.addCleanup(model_graph.stop)
        self.temporary = tempfile.TemporaryDirectory(prefix='bicore-updater-fixture-')
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.repository, self.upstream, self.remote = [self.home / name for name in ('source', 'upstream', 'remote.git')]
        self.root = self.repository / 'runtimes/llama-cpp'
        for directory in (self.repository, self.upstream):
            directory.mkdir()
            self.git('init', '-q', cwd=directory)
            self.git('config', 'user.name', 'Fixture', cwd=directory)
            self.git('config', 'user.email', 'fixture@example.invalid', cwd=directory)
        self.root.mkdir(parents=True)
        fixture = CheckedMtmdFixture(self.upstream, self.root / update.PATCH_DIRECTORY)
        fixture.install(self)
        self.git('add', '.', cwd=self.upstream); self.git('commit', '-qm', 'Upstream A', cwd=self.upstream)
        self.old = self.git('rev-parse', 'HEAD', cwd=self.upstream)
        (self.upstream / 'new-feature').write_text('B\n')
        self.git('add', '.', cwd=self.upstream); self.git('commit', '-qm', 'Upstream B', cwd=self.upstream)
        self.new = self.git('rev-parse', 'HEAD', cwd=self.upstream)
        self.git('-c', 'protocol.file.allow=always', 'submodule', 'add', str(self.upstream), 'vendor/llama.cpp', cwd=self.root)
        self.git('checkout', '--detach', self.old, cwd=self.root / 'vendor/llama.cpp')
        (self.root / 'config').mkdir()
        (self.root / 'config/toolchain.json').write_text(json.dumps({'llamaCommit': self.old, 'otherPin': 'unchanged'}, indent=2) + '\n')
        (self.root / '.gitignore').write_text('build/\n')
        self.git('add', '.', cwd=self.repository); self.git('commit', '-qm', 'Core base', cwd=self.repository)
        self.base = self.git('rev-parse', 'HEAD', cwd=self.root)
        self.git('init', '--bare', '-q', str(self.remote), cwd=self.home)
        self.target = {'commit': self.new, 'ref': 'refs/tags/v1.0.0', 'requested': 'latest'}
        self.relation = 'ahead'
        self.state = self.root / 'build/upstream-update'; self.state.mkdir(parents=True)
        self.bin = self.home / 'bin'; self.bin.mkdir()
        docker_standin(self.bin)
        real_git = shutil.which('git')
        tool = self.bin / 'git'
        tool.write_text('#!' + sys.executable + '\n' + (REPOSITORY_ROOT / 'tests/support/updater_git_fixture.py').read_text())
        tool.chmod(0o755)
        self.env = {**os.environ, 'GITHUB_REPOSITORY': 'example/core', 'GH_TOKEN': 'fixture-token',
                    'GITHUB_WORKSPACE': str(self.repository), 'RUNNER_TEMP': str(self.home),
                    'GITHUB_OUTPUT': str(self.home / 'step-output'), 'GITHUB_STEP_SUMMARY': str(self.home / 'summary'),
                    'BIC_OFFLINE_IMAGE': 'test-only-no-isolation',
                    'PATH': str(self.bin) + os.pathsep + os.environ['PATH'], 'GIT_ALLOW_PROTOCOL': 'file',
                    'BIC_TEST_CONTAINER_PATH': os.environ['PATH'], 'FIXTURE_REAL_GIT': real_git, 'UPDATER_FIXTURE': str(self.home)}
        # The offline Docker stand-in sees the real module through PYTHONPATH,
        # while the fixture's checkout remains the working/source repository.
        (self.repository / 'build_tools').symlink_to(REPOSITORY_ROOT / 'build_tools', target_is_directory=True)
        # Test-only symlink is ignored and never included in candidate trees.
        with (self.repository / '.git/info/exclude').open('a') as out: out.write('\n/build_tools\n')
        self.workflow = jobs((REPOSITORY_ROOT / '.github/workflows/update-llama-cpp.yml').read_text())['update']

    @staticmethod
    def git(*args, cwd):
        return subprocess.check_output(['git', *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()

    def run_step(self, name, **env):
        (self.home / 'step-output').write_text('')
        result = subprocess.run(['bash', '-euo', 'pipefail', '-c', shell_body(step_name(self.workflow, name))],
                                cwd=self.root, env={**self.env, **env}, text=True, capture_output=True)
        return result

    def propose(self, allow_non_fast_forward=False, *, advance=None, deny_push=False, before_candidate=None):
        requested = update.request(self.root, 'example/core', 'develop', 'custom', self.target['ref'], allow_non_fast_forward)
        selected = {key: value for key, value in self.target.items() if key != 'commit'}
        result = update.plan(self.root, requested, selected,
                             {'ref': self.target['ref'], 'commit': self.target['commit']}, {'status': self.relation})
        if result['status'] == 'unchanged': return result
        update.save(self.state, 'plan.json', result)
        inspected = self.run_step('Inspect the existing candidate branch (online)', BRANCH=result['branch'],
                                 UPDATER_ADVANCE=advance or '')
        if inspected.returncode: raise RuntimeError(inspected.stderr)
        existing = update.load(self.state, 'existing.json')
        update.checkout(self.root, result, existing)
        if before_candidate: before_candidate(result)
        result = update.candidate(self.root, result, existing)
        update.save(self.state, 'report.json', result)
        if result['status'] == 'candidate-prepared':
            pushed = self.run_step('Push the candidate branch (online)', BRANCH=result['branch'],
                                   CANDIDATE_COMMIT=result['sourceCommit'], UPDATER_DENY_PUSH='1' if deny_push else '0')
            if pushed.returncode: raise RuntimeError(pushed.stderr)
            result['status'] = ('branch-created-needs-overlay-repair' if result['preflight']['status'] == 'failed'
                                else 'branch-created')
        return result

    def restore_base(self):
        self.git('reset', '--hard', self.base, cwd=self.root)
        self.git('-c', 'protocol.file.allow=always', 'submodule', 'update', '--init', cwd=self.root)

    def start_repair(self, first):
        self.git('checkout', '--detach', first['sourceCommit'], cwd=self.root)

    def calls(self):
        path = self.home / 'git-calls.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
