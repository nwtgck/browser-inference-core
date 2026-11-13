"""Repository-owned paths; runtime selection is explicit and independent of cwd."""
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = ("llama-cpp", "stable-diffusion-cpp")


def runtime_root(runtime: str, repository: Path = REPOSITORY_ROOT) -> Path:
    if runtime not in RUNTIMES:
        raise ValueError(f"Unknown runtime: {runtime}")
    return repository / "runtimes" / runtime


def runtime_test_root(runtime: str, repository: Path = REPOSITORY_ROOT) -> Path:
    """Source-bound verification inputs; never part of a distributed runtime."""
    if runtime not in RUNTIMES:
        raise ValueError(f"Unknown runtime: {runtime}")
    return repository / "tests" / "runtimes" / runtime.replace("-", "_")
