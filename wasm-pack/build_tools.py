#!/usr/bin/env python3
"""Build one native encoder and a separate, dependency-minimal browser decoder.

This runs in the producer only. Compiler sources never enter a consumer package.
Use --offline with Cargo's vendored-source configuration for network-free builds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def run(arguments, **kwargs):
    subprocess.run([str(argument) for argument in arguments], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--test", action="store_true", help="Run encoder and decoder host tests before release build")
    parser.add_argument("--output", type=Path, default=REPO / "build/wasm-pack-tools")
    args = parser.parse_args()
    pins = json.loads((ROOT / "toolchain.json").read_text())
    source = REPO / ".tools/wasm-tools"
    marker = source / "WASM_TOOLS_COMMIT"
    revision = (marker.read_text().strip() if marker.is_file() else subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
    ).strip())
    if revision != pins["wasmToolsCommit"]:
        raise ValueError("wasm-tools source differs from pinned revision")
    rust = subprocess.check_output(["rustc", "--version"], text=True).strip()
    if rust.split()[1] != pins["rustVersion"]:
        raise ValueError("Unexpected Rust toolchain: " + rust)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CARGO_TARGET_DIR": str(output / "cargo")}
    encoder_manifest = ROOT / "encoder/native/Cargo.toml"
    options = ["--locked"] + (["--offline"] if args.offline else [])
    if args.test:
        run(["cargo", "test", *options, "--manifest-path", encoder_manifest], env=env)
        run(["cargo", "test", *options, "--manifest-path", ROOT / "decoder/Cargo.toml"], env=env)
    run(["cargo", "build", "--release", *options, "--manifest-path", encoder_manifest], env=env)
    run(["cargo", "build", "--release", *options, "--target", "wasm32-unknown-unknown",
         "--manifest-path", ROOT / "decoder/Cargo.toml"], env=env)
    shutil.copy2(output / "cargo/release/bicore-wasm-encoder", output / "wasm-encoder")
    shutil.copy2(output / "cargo/wasm32-unknown-unknown/release/bicore_zstd_decoder.wasm",
                 output / "zstd-decoder.wasm")
    files = {}
    for name in ("wasm-encoder", "zstd-decoder.wasm"):
        data = (output / name).read_bytes()
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (output / "tools.json").write_text(json.dumps({"pins": pins, "rustc": rust, "files": files}, indent=2) + "\n")
    print(output / "tools.json")


if __name__ == "__main__":
    main()
