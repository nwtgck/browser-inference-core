"""Audit explicit Docker/network boundaries without claiming a Docker execution."""
import ast
from pathlib import Path
import re
import subprocess
import unittest

from build_tools.project import REPOSITORY_ROOT, runtime_test_root
from tests.support.workflow import jobs, steps, shell_body, step_name

BUILD = (REPOSITORY_ROOT / '.github/workflows/build.yml').read_text()


def run_body(step):
    if '        run: |\n' in step: return shell_body(step)
    match = re.search(r'^        run: (.+)$', step, re.M)
    return match[1] + '\n' if match else None


class OfflineWorkflow(unittest.TestCase):
    def test_every_local_build_phase_has_a_literal_offline_container(self):
        online = {'Install browser test harness', 'Prepare Chromium and OS dependencies',
                  'Publish append-only artifact commit', 'Resolve the published npm dependency (online)'}
        count = 0
        for job in jobs(BUILD).values():
            for step in steps(job):
                body = run_body(step)
                if body is None: continue
                name = step.splitlines()[0].removeprefix('name: ')
                if name in online or name.startswith(('Acquire ', 'Check out')): continue
                count += 1
                self.assertIn('docker run --network none', body, name)
                self.assertIn('--cap-drop=ALL --security-opt=no-new-privileges:true', body, name)
                self.assertIn('-u "$(id -u):$(id -g)"', body, name)
                self.assertIn('-e HOME="$RUNNER_TEMP/bicore-home"', body, name)
                self.assertIn('-e CI=true', body, name)
                self.assertIn('-v "$GITHUB_WORKSPACE:$GITHUB_WORKSPACE"', body, name)
                self.assertNotRegex(body, r'-e (?:GH_TOKEN|GITHUB_TOKEN)\b')
        self.assertGreater(count, 50)
        self.assertNotIn('run-offline', BUILD)
        self.assertNotIn('--privileged', BUILD)
        self.assertNotIn('--network host', BUILD)
        self.assertNotIn('--read-only', BUILD)
        freeze = step_name(jobs(BUILD)['publish'], 'Freeze the validated publication tree')
        self.assertIn('--source-repo "$GITHUB_WORKSPACE"', freeze)

    def test_comment_and_update_containers_preserve_ci_environment(self):
        for filename in ('runtime-comments.yml', 'update-llama-cpp.yml'):
            for job in jobs((REPOSITORY_ROOT / '.github/workflows' / filename).read_text()).values():
                for step in steps(job):
                    body = run_body(step)
                    if body and 'docker run --network none' in body:
                        self.assertIn('-e CI=true', body, filename)

    def test_source_free_image_recipe_uses_stable_environment_content(self):
        for job in jobs(BUILD).values():
            step = step_name(job, 'Acquire offline execution image (online)')
            self.assertIn('docker build --tag', step)
            self.assertIn('"$image_context"', step)
            self.assertIn('COPY --from=node /usr/local/ /usr/local/', step)
            self.assertNotIn('COPY .', step)
            self.assertIn('dpkg-query -W', step)
            self.assertIn('BIC_OFFLINE_ENVIRONMENT=', step)
            self.assertIn("docker image inspect --format '{{.Id}}' node:22.16.0-bookworm ubuntu:24.04", step)
            self.assertNotIn("inspect --format '{{.Id}}' \"bicore-offline", step)

    def test_all_literal_shell_blocks_parse(self):
        count = 0
        for path in (REPOSITORY_ROOT / '.github/workflows').glob('*.yml'):
            for job in jobs(path.read_text()).values():
                for step in steps(job):
                    body = run_body(step)
                    if body is None: continue
                    result = subprocess.run(['bash', '-n'], input=body, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, path.name + ': ' + result.stderr)
                    count += 1
        self.assertGreater(count, 100)

    def test_production_publication_and_updater_have_no_remote_transport(self):
        paths = list((REPOSITORY_ROOT / 'build_tools/publication').glob('*.py'))
        paths += [REPOSITORY_ROOT / 'build_tools/runtimes/llama_cpp/update.py',
                  REPOSITORY_ROOT / 'build_tools/runtimes/llama_cpp/reporting.py',
                  REPOSITORY_ROOT / 'build_tools/github.py']
        for path in paths:
            source = path.read_text(); tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(item.name in ('requests', 'httpx', 'urllib.request', 'socket') for item in node.names), path)
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, ('urllib.request', 'requests', 'httpx', 'socket'), path)
            self.assertNotIn('git_auth_env', source, path)
            self.assertNotIn("['npm', 'install'", source, path)
        for old in ('scripts/publish_artifacts.py', 'runtimes/llama-cpp/scripts/publish_artifacts.py',
                    'runtimes/llama-cpp/scripts/github_api.py', 'runtimes/llama-cpp/scripts/runtime_comments.py',
                    'runtimes/llama-cpp/scripts/update_llama_cpp.py'):
            self.assertFalse((REPOSITORY_ROOT / old).exists(), old)

    def test_native_contract_job_acquires_only_dawn_and_prepares_it_offline(self):
        job = jobs(BUILD)['test']
        acquire = step_name(job, 'Acquire pinned Dawn headers for native contracts (online)')
        self.assertIn('https://github.com/google/dawn/releases/download/', acquire)
        self.assertIn('sha256sum --check -', acquire)
        self.assertNotIn('emsdk install', job)
        check = step_name(job, 'Test host and publication code')
        self.assertIn('docker run --network none', check)
        self.assertIn('build_tools.toolchain.prepare --dawn-only', check)
        self.assertIn('export LCB_TEST_DAWN_PACKAGE=', check)
        self.assertLess(check.index('--dawn-only'), check.index('unittest discover'))

    def test_browser_harness_uses_root_tools_and_relocated_host(self):
        for runtime in ('llama-cpp', 'stable-diffusion-cpp'):
            source = (runtime_test_root(runtime) / 'browser-smoke.mjs').read_text()
            self.assertIn('../../.tools/browser/node_modules/playwright/index.mjs', source)
            self.assertIn('BIC_PLAYWRIGHT_MODULE', source)
            self.assertNotIn('/examples/runtime/index.mjs', source)
        self.assertIn('BICORE_TYPESCRIPT_COMPILER=', BUILD)
        self.assertIn('Acquire pinned TypeScript compiler (online)', BUILD)


if __name__ == '__main__': unittest.main()
