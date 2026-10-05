"""Static workflow contract checks; these do not pretend to execute Actions."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class SourceWorkflows(unittest.TestCase):
    def test_nightly_updates_call_shared_pinned_updater(self):
        workflow=(ROOT/'.github/workflows/update-llama-cpp-nightly.yml').read_text()
        shared=(ROOT/'.github/workflows/update-llama-cpp.yml').read_text()
        for value in ('workflow_dispatch:', 'source: upstream-nightly','uses: ./.github/workflows/update-llama-cpp.yml'):
            self.assertIn(value,workflow)
        self.assertIn('workflow_call:',shared);self.assertIn('--source "$SOURCE_ID"',shared)
        self.assertNotIn('schedule:',workflow);self.assertNotIn('secrets: inherit',workflow)
        self.assertNotIn('pull_request_target',workflow)
    def test_updater_only_materializes_the_selected_source(self):
        text = (ROOT / '.github/workflows/update-llama-cpp.yml').read_text()
        self.assertIn('submodules: false', text)
        self.assertNotIn('submodules: recursive', text)
        self.assertIn('scripts/checkout_source.py --source "$SOURCE_ID"', text)
        self.assertLess(text.index('scripts/checkout_source.py'), text.index('scripts/update_llama_cpp.py'))
        self.assertNotIn('stage-missing', text)

    def test_candidate_matrix_is_from_registry_not_fixed_profiles(self):
        text=(ROOT/'.github/workflows/build-llama-source.yml').read_text()
        for value in ('matrix(root,[name])', 'check_gitlink(root, source)', 'patch_series(root, source)', 'fromJSON(needs.plan.outputs.matrix)', '--source "$SOURCE_ID"'):
            self.assertIn(value,text)
        self.assertNotIn('max-parallel:',text)
        self.assertNotIn('publish_artifacts.py',text)
        self.assertNotIn('contents: write',text)
        self.assertIn('validate_runtime_package.py',text)
        self.assertIn('candidate-llama-',text)
    def test_smoke_scope_is_available_profiles_not_all_default_profiles(self):
        text=(ROOT/'llama-cpp/tests/browser-smoke.mjs').read_text()
        self.assertIn('Object.keys(manifest.profiles)',text)
        self.assertIn('BIC_LLAMA_CHAT_TEMPLATE',text)
        py=(ROOT/'scripts/validate_runtime_package.py').read_text()
        self.assertIn("profiles = {name: profiles[name] for name in entry['profiles']}",py)
        self.assertIn("'webgpu-wasm32-asyncify' in profiles",py)
        self.assertIn("'nodeAsyncifyPassed': True if",py)

    def test_candidate_builds_share_the_checked_checkout_boundary(self):
        text = (ROOT / '.github/workflows/build-llama-source.yml').read_text()
        self.assertEqual(text.count('python3 llama-cpp/scripts/checkout_source.py --source "$SOURCE_ID"'), 2)
        self.assertNotIn('git submodule update', text)
        helper = (ROOT / 'llama-cpp/scripts/checkout_source.py').read_text()
        self.assertIn("'--checkout', '--recursive', '--depth=1'", helper)
        update = helper.index("subprocess.run(['git'")
        self.assertLess(helper.index('check_index_gitlink(root, entry)'), update)
        self.assertLess(helper.index('check_existing_checkout(directory)'), update)
