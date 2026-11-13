"""Local Git redirection/race injection used only by updater workflow tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

args = sys.argv[1:]
real = os.environ['FIXTURE_REAL_GIT']
root = Path(os.environ['UPDATER_FIXTURE'])
with (root / 'git-calls.jsonl').open('a') as out:
    out.write(json.dumps(args) + '\n')
advance = os.environ.get('UPDATER_ADVANCE')
if args and args[0] == 'fetch' and advance and not (root / 'advanced').exists():
    (root / 'advanced').touch()
    subprocess.run([real, '-C', str(root / 'source'), 'push', str(root / 'remote.git'),
                    advance + ':' + args[-1]], check=True)
if args and args[0] == 'push' and os.environ.get('UPDATER_DENY_PUSH') == '1':
    print('Branch write denied', file=sys.stderr)
    raise SystemExit(1)
replacements = {'https://github.com/example/core.git': str(root / 'remote.git'),
                'https://github.com/ggml-org/llama.cpp.git': str(root / 'upstream')}
args = [replacements.get(arg, arg) for arg in args]
raise SystemExit(subprocess.run([real, '-c', 'protocol.file.allow=always', *args]).returncode)
