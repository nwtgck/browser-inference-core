"""Cache partition/policy tests; no remote cache, compiler, or network required."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import REPOSITORY_ROOT as ROOT
from build_tools.toolchain.config import load_toolchain, runtime_toolchain
from build_tools.toolchain.cache import cache_plan, configure, write_values


class BrowserCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for directory in ('toolchain', 'build_tools'):
            shutil.copytree(ROOT / directory, self.root / directory, ignore=shutil.ignore_patterns('__pycache__'))
        for runtime in ('llama-cpp', 'stable-diffusion-cpp'):
            for part in ('config', 'scripts', 'host', 'bridge', 'upstream-patches', 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'):
                if (ROOT / 'runtimes' / runtime / part).is_dir():
                    shutil.copytree(ROOT / 'runtimes' / runtime / part, self.root / 'runtimes' / runtime / part, ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(ROOT / 'runtimes' / runtime / 'CMakeLists.txt', self.root / 'runtimes' / runtime / 'CMakeLists.txt')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        for path in ('runtimes/llama-cpp/vendor/llama.cpp',
                     'runtimes/stable-diffusion-cpp/vendor/stable-diffusion.cpp',
                     'runtimes/stable-diffusion-cpp/vendor/ggml-webgpu-source'):
            subprocess.run(['git', '-C', str(self.root), 'update-index', '--add', '--cacheinfo',
                            '160000,' + '1' * 40 + ',' + path], check=True)
        self.env = {'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64', 'ImageOS': 'ubuntu24', 'ImageVersion': 'fixture1'}

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, runtime='stable-diffusion-cpp', profile='webgpu-wasm32-jspi', variant='browser', source='a'*40):
        return cache_plan(self.root, runtime, profile, variant, source, environment=self.env, ccache_version='ccache version fixture')

    def change_json(self, path, key, value):
        p = self.root / path
        data = json.loads(p.read_text()); data[key] = value; p.write_text(json.dumps(data))

    def test_observer_and_packaging_edits_do_not_change_compiler_partition(self):
        def all_plans():
            return [self.plan(runtime=runtime, profile=profile, variant=variant)
                    for runtime in ('llama-cpp', 'stable-diffusion-cpp')
                    for profile in json.loads((self.root / 'runtimes' / runtime / 'config/profiles.json').read_text())
                    for variant in ('browser', 'test')]
        before = all_plans()
        self.assertEqual(len(before), 16)
        for name in ('build_tools/metrics.py', 'build_tools/validation/runtime.py',
                     'build_tools/packaging/files.py', 'build_tools/metrics_summary.py',
                     'build_tools/toolchain/admission.py',
                     'build_tools/runtimes/llama_cpp/package.py', 'build_tools/runtimes/stable_diffusion_cpp/package.py'):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            with (self.root / name).open('a') as out:
                out.write('\n# observational/package-only fixture change\n')
        self.assertEqual(before, all_plans())

    def test_source_snapshots_accumulate_ccache_but_share_only_exact_partition(self):
        old, new = self.plan(), self.plan(source='b'*40)
        self.assertNotEqual(old['cc-key'], new['cc-key'])
        self.assertEqual(old['cc-prefix'], new['cc-prefix'])
        self.assertEqual(old['em-key'], new['em-key'])
        self.assertTrue(new['cc-key'].startswith(old['cc-prefix']))
        self.assertEqual(len(new['cc-key'].removeprefix(new['cc-prefix'])), 40)

    def test_runtime_profile_and_variant_have_distinct_intermediate_caches(self):
        plans = [self.plan(), self.plan(runtime='llama-cpp'), self.plan(variant='test'),
                 self.plan(profile='webgpu-wasm32-asyncify')]
        for key in ('cc-prefix', 'cc-path', 'em-key'):
            self.assertEqual(len({p[key] for p in plans}), 4)
        self.assertEqual(len({p['dawn-key'] for p in plans}), 1)

    def test_toolchain_pin_change_invalidates_all_compiler_intermediates(self):
        before = self.plan()
        self.change_json('toolchain/config.json', 'emscriptenRelease', 'c'*40)
        after = self.plan()
        for key in ('identity', 'cc-prefix', 'em-key'): self.assertNotEqual(before[key], after[key])
        self.assertEqual(before['dawn-key'], after['dawn-key'])

    def test_shared_implementation_and_runner_image_are_part_of_identity(self):
        first = self.plan()
        with (self.root/'build_tools/toolchain/patch_emscripten.py').open('a') as out: out.write('\n# fixture change\n')
        second = self.plan()
        self.env['ImageVersion'] = 'fixture2'
        third = self.plan()
        self.assertEqual(len({p['identity'] for p in (first, second, third)}), 3)

    def test_container_environment_fingerprint_partitions_compiler_caches(self):
        self.env['BIC_OFFLINE_ENVIRONMENT'] = 'a' * 64
        first = self.plan()
        self.env['BIC_OFFLINE_ENVIRONMENT'] = 'b' * 64
        second = self.plan()
        self.assertNotEqual(first['identity'], second['identity'])
        self.assertNotEqual(first['em-key'], second['em-key'])
        self.assertNotEqual(first['cc-prefix'], second['cc-prefix'])
        self.assertEqual(first['dawn-key'], second['dawn-key'])
        self.env['BIC_OFFLINE_ENVIRONMENT'] = 'sha256:not-a-receipt'
        with self.assertRaisesRegex(ValueError, 'fingerprint'): self.plan()

    def test_image_cache_does_not_depend_on_llama_upstream_pin(self):
        image, llama = self.plan(), self.plan(runtime='llama-cpp')
        self.change_json('runtimes/llama-cpp/config/toolchain.json', 'llamaCommit', 'd'*40)
        self.assertEqual(image, self.plan())
        self.assertNotEqual(llama['em-key'], self.plan(runtime='llama-cpp')['em-key'])
        self.assertEqual(llama['cc-prefix'], self.plan(runtime='llama-cpp')['cc-prefix'])

    def test_staged_gitlink_change_updates_only_the_affected_runtime_em_snapshot(self):
        image, llama = self.plan(), self.plan(runtime='llama-cpp')
        subprocess.run(['git', '-C', str(self.root), 'update-index', '--add', '--cacheinfo',
                        '160000,' + '2' * 40 + ',runtimes/stable-diffusion-cpp/vendor/ggml-webgpu-source'], check=True)
        after = self.plan()
        self.assertNotEqual(image['em-key'], after['em-key'])
        self.assertEqual(image['cc-prefix'], after['cc-prefix'])
        self.assertEqual(llama, self.plan(runtime='llama-cpp'))

    def test_missing_gitlink_does_not_silently_reuse_a_compiler_snapshot(self):
        subprocess.run(['git', '-C', str(self.root), 'update-index', '--force-remove',
                        'runtimes/stable-diffusion-cpp/vendor/ggml-webgpu-source'], check=True)
        with self.assertRaisesRegex(ValueError, 'gitlink'):
            self.plan()

    def test_profile_flags_change_em_and_object_cache_partitions(self):
        before = self.plan()
        path = self.root/'runtimes/stable-diffusion-cpp/config/profiles.json'
        data = json.loads(path.read_text()); data['webgpu-wasm32-jspi']['maximumMemory'] = 2147483648
        path.write_text(json.dumps(data))
        after = self.plan()
        self.assertNotEqual(before['cc-prefix'], after['cc-prefix'])
        self.assertNotEqual(before['em-key'], after['em-key'])

    def test_runtime_build_recipe_changes_only_em_snapshot_not_object_lookup_scope(self):
        before = self.plan()
        with (self.root/'runtimes/stable-diffusion-cpp/CMakeLists.txt').open('a') as out: out.write('\n# fixture flag change\n')
        after = self.plan()
        self.assertNotEqual(before['em-key'], after['em-key'])
        self.assertEqual(before['cc-prefix'], after['cc-prefix'])

    def test_browser_host_inputs_only_change_llama_browser_em_snapshot(self):
        for name in ('build_tools/runtimes/llama_cpp/browser_module.py',
                     'runtimes/llama-cpp/host/emscripten/host-navigator-pre.js'):
            with self.subTest(name=name):
                browser = self.plan(runtime='llama-cpp')
                test = self.plan(runtime='llama-cpp', variant='test')
                image = self.plan()
                with (self.root/name).open('a') as out:
                    out.write('\n// Browser host input fixture change\n')
                after = self.plan(runtime='llama-cpp')
                self.assertNotEqual(browser['em-key'], after['em-key'])
                self.assertEqual(browser['cc-prefix'], after['cc-prefix'])
                self.assertEqual(test, self.plan(runtime='llama-cpp', variant='test'))
                self.assertEqual(image, self.plan())

    def test_dawn_key_is_content_addressed_and_paths_do_not_include_builds(self):
        before = self.plan()
        self.change_json('toolchain/config.json', 'dawnSha256', 'e'*64)
        after = self.plan()
        self.assertTrue(after['dawn-key'].endswith('e'*64))
        self.assertNotEqual(before['dawn-key'], after['dawn-key'])
        self.assertEqual(after['em-path'], '.tools/emsdk/upstream/emscripten/cache')
        for key in ('cc-path', 'em-path', 'dawn-path'):
            self.assertNotIn('build', Path(after[key]).parts)
            self.assertNotIn('dist', Path(after[key]).parts)
            self.assertNotIn('..', Path(after[key]).parts)

    def test_ccache_policy_is_recreated_and_wraps_clang_not_emcc(self):
        path = self.root / '.tools/ccache.conf'; path.parent.mkdir(); path.write_text('sloppiness = time_macros\n')
        result = configure(self.root, self.plan())
        config = path.read_text()
        self.assertIn('compiler_check = content', config)
        self.assertIn('direct_mode = false', config)
        self.assertIn('sloppiness =\n', config)
        self.assertNotIn('time_macros', config)
        self.assertNotIn('base_dir', config)
        self.assertEqual(result['EM_COMPILER_WRAPPER'], 'ccache')
        self.assertEqual(result['EM_CACHE'], str(self.root / self.plan()['em-path']))
        self.assertEqual(result['CCACHE_CONFIGPATH'], str(path))

    def test_invalid_selectors_and_output_injection_fail(self):
        for changes in ({'runtime': '../llama-cpp'}, {'profile': '../../x'}, {'variant': 'release'}, {'source': 'HEAD'}):
            with self.assertRaises(ValueError): self.plan(**changes)
        with self.assertRaises(ValueError): write_values(str(self.root/'output'), {'key': 'x\nEVIL=1'})

    def test_common_configuration_cannot_be_overridden_by_runtime(self):
        shared = load_toolchain(self.root)
        values = runtime_toolchain(self.root/'runtimes/llama-cpp')
        self.assertEqual(values['emsdkVersion'], shared['emsdkVersion'])
        self.assertNotIn('llamaCommit', shared)
        self.change_json('runtimes/llama-cpp/config/toolchain.json', 'emsdkVersion', 'bad')
        with self.assertRaisesRegex(ValueError, 'override'): runtime_toolchain(self.root/'runtimes/llama-cpp')

    def test_image_build_module_change_invalidates_its_em_snapshot(self):
        before = self.plan()
        llama = self.plan(runtime='llama-cpp')
        path = self.root / 'build_tools/runtimes/stable_diffusion_cpp/build.py'
        path.write_text(path.read_text() + '\n# changed image build recipe\n')
        after = self.plan()
        self.assertNotEqual(before['em-key'], after['em-key'])
        self.assertEqual(before['cc-prefix'], after['cc-prefix'])
        self.assertEqual(llama, self.plan(runtime='llama-cpp'))


if __name__ == '__main__': unittest.main()
