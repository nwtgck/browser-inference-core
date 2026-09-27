#!/usr/bin/env python3
"""Record browser checks only in the exact profile/variant that was exercised."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
results=json.loads((ROOT/'build/browser-results.json').read_text())
expected={(profile,variant) for profile in
          ('cpu-wasm32','cpu-wasm64','webgpu-wasm32-jspi','webgpu-wasm64-jspi','webgpu-wasm32-asyncify')
          for variant in ('browser','test')}
if (len(results)!=len(expected) or {(r['profile'],r['variant']) for r in results}!=expected
        or not all(r.get('passed') is True for r in results)):
    raise RuntimeError('Missing or failed profile/variant browser smoke tests')
for result in results:
    path=ROOT/'build'/result['profile']/result['variant']/'provenance.json'
    value=json.loads(path.read_text())
    if value['profile']!=result['profile'] or value['variant']!=result['variant']:
        raise RuntimeError('Browser result does not match build provenance')
    scope='syntheticUntrainedModel' if result.get('syntheticModel') else 'mockedAdapterSuspension'
    value['validation'].update({'browserSmoke':True,scope:result})
    path.write_text(json.dumps(value,indent=2)+'\n')
