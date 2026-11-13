"""Execute the workflow's online shell with local stand-ins for remote tools."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import REPOSITORY_ROOT
from tests.support.workflow import jobs, shell_body, step_name, docker_standin

WORKFLOW = (REPOSITORY_ROOT / '.github/workflows/build.yml').read_text()
JOBS = jobs(WORKFLOW)


@unittest.skipUnless(shutil.which('bash') and shutil.which('jq'), 'bash and jq required')
class ToolchainAcquisition(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        docker_standin(self.bin)
        self.config = {'emsdkVersion': '6.0.9', 'emscriptenRelease': 'a' * 40,
                       'dawnTag': 'v-fixture', 'dawnSha256': hashlib.sha256(b'fixture archive').hexdigest()}
        (self.root / 'toolchain').mkdir()
        (self.root / 'toolchain/config.json').write_text(json.dumps(self.config))
        self.env = {**os.environ, 'PATH': str(self.bin) + os.pathsep + os.environ['PATH'],
                    'GITHUB_OUTPUT': str(self.root / 'outputs'), 'DAWN_TAG': self.config['dawnTag'],
                    'DAWN_SHA256': self.config['dawnSha256'], 'EMSDK_VERSION': self.config['emsdkVersion'],
                    'EMSCRIPTEN_RELEASE': self.config['emscriptenRelease'], 'FIXTURE_ROOT': str(self.root), 'GITHUB_WORKSPACE': str(self.root),
                    'RUNNER_TEMP': str(self.root), 'BIC_OFFLINE_IMAGE': 'test-only-no-isolation'}
        self.tool('curl', '''
args = sys.argv[1:]
(root / 'curl-args.json').write_text(json.dumps(args))
assert args[args.index('--proto') + 1] == '=https'
assert args[args.index('--proto-redir') + 1] == '=https'
assert 'https://github.com/google/dawn/releases/download/v-fixture/emdawnwebgpu_pkg-v-fixture.zip' in args
Path(args[args.index('--output') + 1]).write_bytes(b'fixture archive')
''')

    def tool(self, name, code):
        target = self.bin / name
        target.write_text(f'#!{sys.executable}\nimport os,sys,json\nfrom pathlib import Path\nroot=Path(os.environ["FIXTURE_ROOT"])\n' + code)
        target.chmod(0o755)

    def run_step(self, name, **env):
        script = shell_body(step_name(JOBS['compile'], name))
        return subprocess.run(['bash', '-euo', 'pipefail', '-c', script], cwd=self.root,
                              env={**self.env, **env}, capture_output=True, text=True)

    def test_both_runtime_jobs_expose_the_same_fixed_acquisition_commands(self):
        for name in ('Read pinned acquisition settings', 'Acquire pinned compiler (online)',
                     'Acquire pinned Dawn archive (online on cache miss)'):
            self.assertEqual(shell_body(step_name(JOBS['compile'], name)),
                             shell_body(step_name(JOBS['image-compile'], name)))
        self.assertNotIn('setup_toolchain.py', WORKFLOW)
        self.assertNotIn('./.github/actions/', WORKFLOW)
        self.assertIn('Acquisition is intentionally inline', WORKFLOW)

    def test_settings_are_data_and_reject_path_or_output_injection(self):
        result = self.run_step('Read pinned acquisition settings')
        self.assertEqual(result.returncode, 0, result.stderr)
        values = dict(line.split('=', 1) for line in (self.root / 'outputs').read_text().splitlines())
        self.assertEqual(values['dawn-tag'], 'v-fixture')
        for key, value in [('dawnTag', '../outside'), ('emsdkVersion', '6.0.9\nBAD=1'),
                           ('emscriptenRelease', 'HEAD'), ('dawnSha256', 'missing')]:
            (self.root / 'toolchain/config.json').write_text(json.dumps({**self.config, key: value}))
            (self.root / 'outputs').unlink()
            result = self.run_step('Read pinned acquisition settings')
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((self.root / 'outputs').read_text(), '')

    def test_cache_miss_checks_download_and_atomically_installs_archive(self):
        result = self.run_step('Acquire pinned Dawn archive (online on cache miss)')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / '.tools/downloads/emdawnwebgpu_pkg-v-fixture.zip').read_bytes(), b'fixture archive')
        self.assertFalse(list((self.root / '.tools/downloads').glob('dawn-download.*')))
        self.assertTrue((self.root / 'curl-args.json').is_file())

    def test_invalid_download_cannot_replace_archive(self):
        result = self.run_step('Acquire pinned Dawn archive (online on cache miss)', DAWN_SHA256='b' * 64)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / '.tools/downloads/emdawnwebgpu_pkg-v-fixture.zip').exists())
        self.assertFalse(list((self.root / '.tools/downloads').glob('dawn-download.*')))

    def test_cache_hit_does_not_contact_remote_and_still_has_unconditional_local_verification(self):
        archive = self.root / '.tools/downloads/emdawnwebgpu_pkg-v-fixture.zip'
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b'cached archive')
        result = self.run_step('Acquire pinned Dawn archive (online on cache miss)')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / 'curl-args.json').exists())
        for job in ('compile', 'image-compile'):
            step = step_name(JOBS[job], 'Verify and prepare local compiler inputs (no downloads)')
            self.assertNotIn('        if:', step)
            self.assertIn('python3 -m build_tools.toolchain.prepare', step)

    def test_compiler_release_is_checked_before_upstream_install_or_activate(self):
        sdk = self.root / '.tools/emsdk'
        sdk.mkdir(parents=True)
        tags = sdk / 'emscripten-releases-tags.json'
        tags.write_text(json.dumps({'releases': {'6.0.9': 'c' * 40}}))
        self.tool('emsdk', '''
with (root / 'sdk-calls').open('a') as out:
    out.write(json.dumps(sys.argv[1:]) + '\\n')
''')
        shutil.copy2(self.bin / 'emsdk', sdk / 'emsdk')
        result = self.run_step('Acquire pinned compiler (online)')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'sdk-calls').exists())
        tags.write_text(json.dumps({'releases': {'6.0.9': 'a' * 40}}))
        result = self.run_step('Acquire pinned compiler (online)')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([json.loads(line) for line in (self.root / 'sdk-calls').read_text().splitlines()],
                         [['install', '6.0.9'], ['activate', '6.0.9']])


if __name__ == '__main__':
    unittest.main()
