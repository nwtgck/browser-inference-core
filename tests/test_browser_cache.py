"""Cache partition/policy tests; no remote cache, compiler, or network required."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'scripts'))
from browser_toolchain import load_toolchain, runtime_toolchain
from ci_cache import cache_plan, configure, write_values


class BrowserCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for directory in ('toolchain', 'scripts'):
            shutil.copytree(ROOT / directory, self.root / directory, ignore=shutil.ignore_patterns('__pycache__'))
        for runtime in ('llama-cpp', 'stable-diffusion-cpp'):
            for part in ('config', 'scripts', 'upstream-patches', 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'):
                if (ROOT / runtime / part).is_dir():
                    shutil.copytree(ROOT / runtime / part, self.root / runtime / part, ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(ROOT / runtime / 'CMakeLists.txt', self.root / runtime / 'CMakeLists.txt')
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
                    for profile in json.loads((self.root / runtime / 'config/profiles.json').read_text())
                    for variant in ('browser', 'test')]
        before = all_plans()
        self.assertEqual(len(before), 16)
        for name in ('scripts/pipeline_metrics.py', 'scripts/validate_runtime_package.py',
                     'scripts/package_inputs.py', 'scripts/summarize_pipeline_metrics.py',
                     'scripts/cache_admission.py', 'stable-diffusion-cpp/scripts/prepare_upstream.py',
                     'llama-cpp/scripts/package_runtime.py', 'stable-diffusion-cpp/scripts/package_runtime.py'):
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
        with (self.root/'scripts/patch_emscripten.py').open('a') as out: out.write('\n# fixture change\n')
        second = self.plan()
        self.env['ImageVersion'] = 'fixture2'
        third = self.plan()
        self.assertEqual(len({p['identity'] for p in (first, second, third)}), 3)

    def test_image_cache_does_not_depend_on_llama_upstream_pin(self):
        image, llama = self.plan(), self.plan(runtime='llama-cpp')
        self.change_json('llama-cpp/config/toolchain.json', 'llamaCommit', 'd'*40)
        self.assertEqual(image, self.plan())
        self.assertNotEqual(llama['em-key'], self.plan(runtime='llama-cpp')['em-key'])
        self.assertEqual(llama['cc-prefix'], self.plan(runtime='llama-cpp')['cc-prefix'])

    def test_profile_flags_change_em_and_object_cache_partitions(self):
        before = self.plan()
        path = self.root/'stable-diffusion-cpp/config/profiles.json'
        data = json.loads(path.read_text()); data['webgpu-wasm32-jspi']['maximumMemory'] = 2147483648
        path.write_text(json.dumps(data))
        after = self.plan()
        self.assertNotEqual(before['cc-prefix'], after['cc-prefix'])
        self.assertNotEqual(before['em-key'], after['em-key'])

    def test_runtime_build_recipe_changes_only_em_snapshot_not_object_lookup_scope(self):
        before = self.plan()
        with (self.root/'stable-diffusion-cpp/CMakeLists.txt').open('a') as out: out.write('\n# fixture flag change\n')
        after = self.plan()
        self.assertNotEqual(before['em-key'], after['em-key'])
        self.assertEqual(before['cc-prefix'], after['cc-prefix'])

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
        values = runtime_toolchain(self.root/'llama-cpp')
        self.assertEqual(values['emsdkVersion'], shared['emsdkVersion'])
        self.assertNotIn('llamaCommit', shared)
        self.change_json('llama-cpp/config/toolchain.json', 'emsdkVersion', 'bad')
        with self.assertRaisesRegex(ValueError, 'override'): runtime_toolchain(self.root/'llama-cpp')


if __name__ == '__main__': unittest.main()
