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
