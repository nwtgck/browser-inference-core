"""Record the complete current Q8 factory with the real Dawn declarations.

C WebGPU calls are mocked; no pipeline factory, selector, or cache is extracted
or reverse-patched. The small current provider dispatch fragment remains a
compiler-AST-selected check, not a whole-provider runtime execution.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from build_tools.runtimes.llama_cpp import source
from tests.support.llama_cpp.current_webgpu import compile_recorder
from tests.support.llama_cpp.dispatch_ast import current_dispatch


def prepare(upstream: Path, output: Path) -> None:
    source.prepare(upstream, output)


def build_host(upstream: Path, prepared: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    fragment = output / 'current-dispatch.inc'
    fragment.write_text(current_dispatch(upstream, prepared))
    return compile_recorder(upstream, prepared, output, Path(__file__).with_name('host-cases.cpp'),
                            fragment=fragment)
