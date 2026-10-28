"""Extract the pinned graph-builder implementation for a narrow model fixture."""
from pathlib import Path
import argparse

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
s = (a.source / 'tools/mtmd/clip.cpp').read_text()
start = s.index('clip_graph::clip_graph(')
end = s.index('static std::unique_ptr<clip_graph> clip_get_graph_builder', start)
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'clip-graph-base.inc').write_text(s[start:end])
