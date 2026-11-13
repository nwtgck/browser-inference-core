"""Test-only Docker CLI stand-in: argument/env execution, NOT network isolation."""
import os
import subprocess
import sys

args = sys.argv[1:]
assert args.pop(0) == 'run'
assert '--network' in args and args[args.index('--network') + 1] == 'none'
assert '--cap-drop=ALL' in args and '--security-opt=no-new-privileges:true' in args
child_env = {'PATH': os.environ.get('BIC_TEST_CONTAINER_PATH', os.environ['PATH'])}
cwd = None
while args and args[0].startswith('-'):
    option = args.pop(0)
    if option in ('--rm', '-i', '--cap-drop=ALL', '--security-opt=no-new-privileges:true'):
        continue
    if option in ('--network', '-u', '-v', '-w', '-e'):
        value = args.pop(0)
        if option == '-w': cwd = value
        if option == '-e':
            key, separator, selected = value.partition('=')
            if separator: child_env[key] = selected
            elif key in os.environ: child_env[key] = os.environ[key]
        continue
    raise AssertionError('Unexpected Docker option: ' + option)
args.pop(0)  # Image identity has no effect in this local stand-in.
assert 'HOME' in child_env
raise SystemExit(subprocess.run(args, cwd=cwd, env=child_env).returncode)
