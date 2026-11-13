"""Packaged profiles and workflow matrix remain aligned with build behavior."""
import json
from pathlib import Path
import re
import unittest

from build_tools.project import runtime_root

ROOT = runtime_root('llama-cpp')


class ProfileConfiguration(unittest.TestCase):
    def test_ci_builds_every_packaged_profile(self):
        profiles = json.loads((ROOT / 'config/profiles.json').read_text())
        workflow = (ROOT.parents[1] / '.github/workflows/build.yml').read_text()
        matrix = workflow.split('        profile:\n', 1)[1].split('        variant:', 1)[0]
        self.assertEqual(set(re.findall(r'^          - ([a-z0-9-]+)$', matrix, re.MULTILINE)), set(profiles))
        variants = json.loads((ROOT / 'config/variants.json').read_text())
        variant_matrix = workflow.split('        variant:\n', 1)[1].split('    steps:', 1)[0]
        self.assertEqual(set(re.findall(r'^          - ([a-z0-9-]+)$', variant_matrix, re.MULTILINE)), set(variants))

    def test_profile_suspension_and_memory_contract(self):
        profiles = json.loads((ROOT / 'config/profiles.json').read_text())
        expected = {
            'cpu-wasm32': (False, False, False, False, 4294967296),
            'webgpu-wasm64-jspi-gpu-profile': (True, True, True, False, 17179869184),
            'webgpu-wasm32-asyncify': (False, True, False, True, 4294967296),
            'webgpu-wasm32-jspi': (False, True, True, False, 4294967296),
            'webgpu-wasm64-jspi': (True, True, True, False, 17179869184),
        }
        self.assertEqual(set(profiles), set(expected))
        for profile, values in expected.items():
            self.assertEqual(tuple(profiles[profile][field] for field in ['memory64', 'webgpu', 'jspi', 'asyncify', 'maximumMemory']), values)
        self.assertEqual(json.loads((ROOT / 'config/variants.json').read_text()), {
            'browser': {'assertions': 0, 'environment': 'web,worker'},
            'test': {'assertions': 1, 'environment': 'web,worker,node'},
        })
