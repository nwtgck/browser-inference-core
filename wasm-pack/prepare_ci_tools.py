#!/usr/bin/env python3
"""Prepare producer-only tools on Linux; no pip environment or C++ matcher build.

The existing Node verifier supplies Brotli. Python uses its standard library and
libzstd's public C interface. Rust dependencies and parser revision are locked.
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def run(arguments, **kwargs):
    subprocess.run([str(argument) for argument in arguments], check=True, **kwargs)


def main():
    pins = json.loads((ROOT / "toolchain.json").read_text())
    run(["sudo", "apt-get", "update"])
    run(["sudo", "apt-get", "install", "-y", "--no-install-recommends", "libzstd1"])
    run(["rustup", "toolchain", "install", pins["rustVersion"], "--profile", "minimal",
         "--target", "wasm32-unknown-unknown"])
    source = REPO / ".tools/wasm-tools"
    source.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        raise ValueError("Expected fresh pinned wasm-tools directory")
    run(["git", "init", "-q", source])
    run(["git", "remote", "add", "origin", pins["wasmToolsRepository"]], cwd=source)
    run(["git", "fetch", "--depth=1", "origin", pins["wasmToolsCommit"]], cwd=source)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != pins["wasmToolsCommit"]:
        raise ValueError("Parser pin mismatch")
    run(["rustup", "run", pins["rustVersion"], sys.executable, ROOT / "build_tools.py", "--test"])


if __name__ == "__main__":
    main()
