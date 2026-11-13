"""Import boundaries enforce the distinction between current and frozen tests."""
import subprocess
import sys
import unittest

from build_tools.project import REPOSITORY_ROOT


class DependencyTiers(unittest.TestCase):
    def test_current_source_contracts_import_without_historical_composer_or_stages(self):
        code = r'''
import importlib.abc
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
blocked = ('tests.support.legacy_webgpu_source', 'tests.support.llama_cpp.webgpu_stages',
           'tests.support.historical_webgpu')
class RejectHistory(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Current source contract imports a historical fixture: ' + fullname)
sys.meta_path.insert(0, RejectHistory())
from build_tools.runtimes.llama_cpp import source, webgpu_plan
from tests.runtimes.llama_cpp.source import test_plan, test_full_headers
from tests.runtimes.llama_cpp import test_webgpu_source_overlay
from tests.support.llama_cpp import current_webgpu, dispatch_ast
for runtime in ('webgpu-q8-short', 'webgpu-q8-subgroup'):
    path = Path(sys.argv[1]) / 'tests/runtimes/llama_cpp' / runtime / 'emit_factory.py'
    spec = importlib.util.spec_from_file_location('current_' + runtime.replace('-', '_'), path)
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
assert source.prepare is webgpu_plan.prepare
print('current dependency tier is independent')
'''
        result = subprocess.run([sys.executable, '-I', '-c', code, str(REPOSITORY_ROOT)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('current dependency tier is independent', result.stdout)
