"""Test-only gh/curl stand-ins. Every operation is recorded locally; no sockets."""
import json
import os
from pathlib import Path
import sys

root = Path(os.environ['COMMENTS_FIXTURE'])
cfg = json.loads((root / 'scenario.json').read_text())
args = sys.argv[1:]
tool = Path(sys.argv[0]).name
with (root / 'service-calls.jsonl').open('a') as out:
    out.write(json.dumps({'tool': tool, 'args': args, 'hasToken': 'GH_TOKEN' in os.environ}) + '\n')
if tool == 'curl':
    output = Path(args[args.index('--output') + 1])
    assert args[args.index('--max-filesize') + 1] == '262144'
    assert args[args.index('--proto') + 1] == '=https'
    if args[-1].startswith('https://api.github.com/'):
        assert '--location' not in args
        assert 'Authorization: Bearer fixture-token' in args
        assert args[args.index('--max-redirs') + 1] == '0'
        headers = Path(args[args.index('--dump-header') + 1])
        if cfg.get('directArchive'):
            output.write_bytes((root / 'report.zip').read_bytes())
            headers.write_text('HTTP/2 200 OK\r\n')
            print('200', end='')
        else:
            output.write_bytes(b'')
            value = cfg.get('redirect', 'https://storage.invalid/report?signature=fixture')
            headers.write_text('HTTP/2 302 Found\r\nLocation: ' + value + '\r\n')
            print('302', end='')
    else:
        assert 'GH_TOKEN' not in os.environ
        assert '-H' not in args
        assert args[args.index('--proto-redir') + 1] == '=https'
        raw = (root / 'report.zip').read_bytes()
        if len(raw) > 262144 or cfg.get('failDownload'):
            raise SystemExit(63)
        output.write_bytes(raw)
    raise SystemExit(0)
assert tool == 'gh' and args[:3] == ['api', '--hostname', 'github.com']
method = args[args.index('--method') + 1]
endpoint = args[args.index('--method') + 2]
fields = dict(value.split('=', 1) for index, value in enumerate(args) if index and args[index - 1] == '-f')
page = int(fields.get('page', '1'))
if method in ('POST', 'PATCH'):
    payload = json.loads(Path(args[args.index('--input') + 1]).read_text())
    assert set(payload) == {'body'}
    assert '/issues/' in endpoint and '/comments' in endpoint
    with (root / 'writes.jsonl').open('a') as out:
        out.write(json.dumps({'method': method, 'endpoint': endpoint, **payload}) + '\n')
    if cfg.get('failWrite'): raise SystemExit(1)
    print('{"id":42}')
    raise SystemExit(0)
assert method == 'GET'
if endpoint.endswith('/pulls'):
    values = cfg['pulls'][(page - 1) * 100:page * 100]
    result = values
elif '/pulls/' in endpoint:
    number = int(endpoint.rsplit('/', 1)[1])
    result = cfg.get('currentPr') or next(pr for pr in cfg['pulls'] if pr['number'] == number)
elif endpoint.endswith('/actions/workflows/build.yml/runs'):
    assert 'branch' in fields and 'head_sha' in fields
    refreshed = bool(list((root / 'state').glob('*/current-pr.json')))
    values = cfg.get('currentRuns', cfg['runs']) if refreshed else cfg['runs']
    count_key = 'currentTotalCount' if refreshed and 'currentTotalCount' in cfg else 'totalCount'
    result = {'total_count': cfg.get(count_key, len(values)),
              'workflow_runs': values[(page - 1) * 100:page * 100]}
elif endpoint.endswith('/artifacts'):
    if cfg.get('failArtifactListing'): raise SystemExit(1)
    values = cfg.get('artifacts', [{'id': 91, 'name': fields['name'], 'expired': False}])
    result = {'artifacts': values[(page - 1) * 100:page * 100]}
elif endpoint.endswith('/comments'):
    values = cfg.get('comments', [])
    result = values[(page - 1) * 100:page * 100]
else:
    raise AssertionError('Unexpected fixture endpoint: ' + endpoint)
print(json.dumps(result))
