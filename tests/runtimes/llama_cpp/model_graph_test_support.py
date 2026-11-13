"""Synthetic CMake outputs for build-command tests; never real artifacts."""
import json
from pathlib import Path

def seed_model_graph_outputs(build: Path) -> dict:
    provenance = {"fixture": "model-graph-build-command"}
    schema = {"schemaSha256": "d" * 64}
    (build / "model-graph-chat-overlay").mkdir(exist_ok=True)
    (build / "generated").mkdir(exist_ok=True)
    (build / "model-graph-chat-overlay/provenance.json").write_text(json.dumps(provenance))
    (build / "generated/model-graph-v1.schema.json").write_text(json.dumps(schema))
    return {**provenance, "schemaSha256": schema["schemaSha256"]}
