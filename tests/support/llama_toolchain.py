"""Create the actual monorepo toolchain layout in isolated test directories."""
import json
from pathlib import Path
import shutil
from build_tools.project import REPOSITORY_ROOT as COMMON, runtime_root
ROOT = runtime_root('llama-cpp')


def merged_toolchain():
    return {**json.loads((COMMON / 'toolchain/config.json').read_text()),
            **json.loads((ROOT / 'config/toolchain.json').read_text())}


def seed_toolchain(runtime, values=None, *, scripts=False):
    values = merged_toolchain() if values is None else values
    local_keys = {'llamaCommit', 'artifactBranch'}
    (runtime / 'config').mkdir(parents=True, exist_ok=True)
    (runtime.parents[1] / 'toolchain').mkdir(parents=True, exist_ok=True)
    (runtime / 'config/toolchain.json').write_text(json.dumps({k: v for k, v in values.items() if k in local_keys}))
    (runtime.parents[1] / 'toolchain/config.json').write_text(json.dumps({k: v for k, v in values.items() if k not in local_keys}))
    if scripts:
        shutil.copytree(COMMON / 'build_tools', runtime.parents[1] / 'build_tools',
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
