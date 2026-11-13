"""Run production package CLIs with fixture inputs and an observable npm process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from build_tools.project import REPOSITORY_ROOT


def run_cli(directory: Path, module: str, arguments: list[str], *,
            source_root: Path | None = None, fail_pack: int | None = None):
    bin_directory = directory / 'cli-bin'
    bin_directory.mkdir(exist_ok=True)
    calls = directory / 'npm-calls.jsonl'
    calls.write_text('')
    npm = bin_directory / 'npm'
    npm.write_text('#!' + sys.executable + '\n' + '''
import json, os, subprocess, sys
from pathlib import Path
log = Path(os.environ['BIC_TEST_NPM_LOG'])
previous = log.read_text().splitlines()
with log.open('a') as stream:
    stream.write(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:]}) + '\\n')
if str(len(previous) + 1) == os.environ.get('BIC_TEST_FAIL_PACK'):
    sys.stderr.write('injected npm pack failure\\n')
    sys.exit(47)
raise SystemExit(subprocess.run([os.environ['BIC_TEST_REAL_NPM'], *sys.argv[1:]]).returncode)
''')
    npm.chmod(0o755)
    environment = dict(os.environ, PATH=str(bin_directory) + os.pathsep + os.environ['PATH'],
                       BIC_TEST_NPM_LOG=str(calls), BIC_TEST_REAL_NPM=shutil.which('npm') or 'npm',
                       BIC_TEST_FAIL_PACK=str(fail_pack or ''), BIC_METRICS_MODE='off')
    # Only the owned source root is redirected to the synthetic fixture. The
    # production parser, assembly, validator, npm call and activation all run.
    code = '''
import importlib, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
module = importlib.import_module(sys.argv[2])
if sys.argv[3] != '-': module.ROOT = Path(sys.argv[3])
sys.argv = [module.__name__, *sys.argv[4:]]
module.main()
'''
    command = [sys.executable, '-c', code, str(REPOSITORY_ROOT), module,
               str(source_root) if source_root is not None else '-', *map(str, arguments)]
    result = subprocess.run(command, cwd=directory, env=environment, capture_output=True, text=True)
    return result, [json.loads(line) for line in calls.read_text().splitlines()]
