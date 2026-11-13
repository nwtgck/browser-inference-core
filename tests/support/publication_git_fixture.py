"""Local Git fault injection for the workflow shell; never used by production."""
import os
from pathlib import Path
import subprocess
import sys

args = sys.argv[1:]
real = os.environ['FIXTURE_REAL_GIT']
mode = os.environ['FIXTURE_GIT_MODE']
state = Path(os.environ['FIXTURE_STATE'])
is_push = len(args) > 2 and args[0] == '-C' and args[2] == 'push'
if is_push and (not state.exists() or mode == 'always-conflict'):
    state.write_text('injected')
    if mode == 'always-conflict':
        print('non-fast-forward', file=sys.stderr)
        raise SystemExit(1)
    work, candidate = args[1], args[-1].split(':', 1)[0]
    tree = subprocess.check_output([real, '-C', work, 'rev-parse', candidate + '^{tree}'], text=True).strip()
    if mode == 'conflict-once':
        rival = subprocess.check_output([real, '-C', work, 'commit-tree', tree, '-m', 'competing publication fixture'], text=True).strip()
        subprocess.run([real, '-C', work, 'push', 'origin', rival + ':refs/heads/artifacts'], check=True)
    elif mode.startswith('lost-response'):
        subprocess.run([real, *args], check=True)
        if mode == 'lost-response-with-child':
            rival = subprocess.check_output([real, '-C', work, 'commit-tree', tree, '-p', candidate, '-m', 'later fixture publication'], text=True).strip()
            subprocess.run([real, '-C', work, 'push', 'origin', rival + ':refs/heads/artifacts'], check=True)
        print('connection lost after accepted push', file=sys.stderr)
        raise SystemExit(1)
raise SystemExit(subprocess.run([real, *args]).returncode)
