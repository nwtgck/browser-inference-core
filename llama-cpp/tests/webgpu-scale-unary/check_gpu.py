"""Optional real Node/Dawn parity check; no browser, runtime build or speed claim."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('scale_unary_tests', ROOT / 'tests/test_webgpu_scale_unary.py')
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--webgpu-module', type=Path, required=True)
    parser.add_argument('--node-command-json', default='["node"]')
    parser.add_argument('--pipeline-order', choices=('forward', 'reverse'), default='forward')
    args = parser.parse_args()
    try:
        command = json.loads(args.node_command_json)
    except json.JSONDecodeError as error:
        parser.error('--node-command-json is invalid JSON: ' + str(error))
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        parser.error('--node-command-json must be a nonempty string array')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='scale-unary-gpu-') as temporary:
        root = Path(temporary)
        prepared = tests.prepare(args.source.resolve(), root / 'prepared')
        shaders = tests.emit_factory(args.source.resolve(), prepared, root / 'factory')
        subprocess.run([*command, str(Path(__file__).with_name('check-gpu.mjs')), str(shaders),
                        str(args.output.resolve()), str(args.webgpu_module.resolve()), args.pipeline_order], check=True)
    report = json.loads(args.output.read_text())
    if not report.get('ok') or len(report.get('cases', [])) != 392:
        raise RuntimeError('Incomplete or failed scale/unary GPU check')


if __name__ == '__main__':
    main()
