"""Record the current short-Q8 factory without importing historical stages."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from build_tools.runtimes.llama_cpp import source
from tests.support.llama_cpp.current_webgpu import compile_recorder


def prepare(upstream: Path, output: Path) -> None:
    source.prepare(upstream, output)


def build_host(upstream: Path, prepared: Path, output: Path) -> Path:
    return compile_recorder(upstream, prepared, output, Path(__file__).with_name('host-cases.cpp'))
