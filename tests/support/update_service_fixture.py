"""Local gh response fixture for literal updater YAML; no network transport."""
import json
import os
from pathlib import Path
import sys

root = Path(os.environ['UPDATER_FIXTURE'])
args = sys.argv[1:]
assert args[:3] == ['api', '--hostname', 'github.com']
method = args[args.index('--method') + 1]
endpoint = args[args.index('--method') + 2]
assert method == 'GET'
config = json.loads((root / 'service.json').read_text())
with (root / 'api-calls.jsonl').open('a') as out: out.write(json.dumps(args) + '\n')
if config.get('fail'):
    print('Service read denied', file=sys.stderr)
    raise SystemExit(1)
fields = dict(value.split('=', 1) for index, value in enumerate(args) if index and args[index - 1] == '-f')
if endpoint == 'repos/ggml-org/llama.cpp/releases/latest':
    result = config['latest']
elif endpoint == 'repos/ggml-org/llama.cpp/releases':
    page = int(fields['page'])
    result = config['releases'][(page - 1) * 100:page * 100]
elif endpoint.startswith('repos/ggml-org/llama.cpp/compare/'):
    result = {'status': config.get('relation', 'ahead')}
else:
    raise AssertionError('Unexpected updater service operation: ' + endpoint)
print(json.dumps(result))
