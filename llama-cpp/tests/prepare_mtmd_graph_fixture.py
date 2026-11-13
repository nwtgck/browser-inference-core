"""Extract pinned graph helpers for a narrow model-metadata fixture."""
from pathlib import Path
import argparse
import hashlib
import json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
graph = a.source / 'tools/mtmd/mtmd-graph.cpp'
model = a.source / 'tools/mtmd/mtmd-model.cpp'
s = graph.read_text()
start = s.index('mtmd_graph::mtmd_graph(')
end = s.index('std::unique_ptr<mtmd_graph> mtmd_get_graph_builder', start)
model_source = model.read_text()
model_start = model_source.index('int mtmd_model::n_mmproj_embd() const {')
model_end = model_source.index('\n}\n', model_start) + 3
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'mtmd-graph-base.inc').write_text(s[start:end])
(a.output / 'mtmd-model-embedding.inc').write_text(model_source[model_start:model_end])
(a.output / 'source-identities.json').write_text(json.dumps({
    str(path.relative_to(a.source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (graph, model)
}, indent=2) + '\n')
