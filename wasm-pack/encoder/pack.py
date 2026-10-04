#!/usr/bin/env python3
"""Package prebuilt Wasm without modifying inputs or compiling inference engines.

Python owns identities, subprocess boundaries, representation metadata and atomic
publication. The native Rust encoder owns all parser/prediction/matching/delta
algorithms. Node verifies every result with the actual browser decoder.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 64 * 1024 * 1024
ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\Z")
DIGEST = re.compile(r"[a-f0-9]{64}\Z")
CODECS = ("gzip", "brotli", "zstd")


def identity(data: bytes) -> dict:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def safe_path(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or len(name) > 512
            or not re.fullmatch(r"[a-zA-Z0-9_./-]+", name)
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Unsafe input path")
    path = root
    for part in name.split("/"):
        path /= part
        if path.is_symlink():
            raise ValueError("Linked input")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Input escapes root")
    return path


def read_input(root: Path, raw: dict) -> bytes:
    if (not isinstance(raw, dict) or type(raw.get("bytes")) is not int
            or not 8 <= raw["bytes"] <= LIMIT
            or not isinstance(raw.get("sha256"), str)
            or not DIGEST.fullmatch(raw["sha256"])):
        raise ValueError("Invalid input identity")
    path = safe_path(root, raw["path"])
    if not path.is_file() or path.stat().st_size != raw["bytes"]:
        raise ValueError("Input length mismatch")
    data = path.read_bytes()
    if identity(data) != {key: raw[key] for key in ("bytes", "sha256")}:
        raise ValueError("Input hash mismatch")
    if data[:8] != b"\x00asm\x01\x00\x00\x00":
        raise ValueError("Not a core Wasm binary")
    return data


def compress(data: bytes, codec: str) -> bytes:
    if codec == "gzip":
        return gzip.compress(data, compresslevel=9, mtime=0)
    if codec == "brotli":
        # Node is already a required verifier. Reuse its Brotli implementation
        # instead of adding another installed Python or native dependency.
        return subprocess.run(
            ["node", str(ROOT / "encoder/brotli.mjs")], input=data,
            stdout=subprocess.PIPE, check=True, timeout=180,
        ).stdout
    if codec == "zstd":
        from zstd_codec import encode
        return encode(data, level=19, window=20)
    raise ValueError("Unknown codec")


def encode_pair(encoder: Path, base: Path, target: Path, output: Path) -> list[dict]:
    """Strictly bind native output filenames to the supported private protocol."""
    subprocess.run(
        [str(encoder), "pair", str(base), str(target), str(output)],
        check=True, timeout=300,
    )
    result = json.loads((output / "result.json").read_text())
    candidates = result.get("candidates")
    if result.get("formatVersion") != 1 or not isinstance(candidates, list):
        raise ValueError("Unsupported native encoder protocol")
    expected = [{"label": "plain", "recipe": "plain.wxp"}]
    predicted = {"label": "predicted", "recipe": "predicted.wxp",
                 "prediction": "prediction.prd", "dictionary": "dictionary.bin"}
    if candidates not in (expected, [*expected, predicted]):
        raise ValueError("Invalid native encoder candidates")
    parsed = []
    for candidate in candidates:
        entry = {"label": candidate["label"]}
        for key in ("recipe", "prediction", "dictionary"):
            if key not in candidate:
                continue
            path = safe_path(output, candidate[key])
            maximum = 4 * 1024 * 1024 if key == "prediction" else LIMIT
            if not path.is_file() or path.stat().st_size > maximum:
                raise ValueError("Invalid encoder output size")
            data = path.read_bytes()
            if len(data) > maximum:
                raise ValueError("Encoder output grew beyond limit")
            entry[key] = data
        parsed.append(entry)
    return parsed


def build(root, spec, destination, *, codecs=("gzip", "brotli"), encoder=None,
          pairs=None, zstd_decoder=None):
    root = Path(root).absolute()
    destination = Path(destination).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Input directory missing")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Output must not already exist")
    if not codecs or len(set(codecs)) != len(codecs) or any(c not in CODECS for c in codecs):
        raise ValueError("Invalid codecs")
    if "zstd" in codecs and zstd_decoder is None:
        raise ValueError("Zstandard requires the matching built decoder")
    targets = spec.get("targets")
    if not isinstance(targets, dict) or not 1 <= len(targets) <= 128:
        raise ValueError("Invalid targets")
    values = {}
    for name in sorted(targets):
        if not isinstance(name, str) or not ID.fullmatch(name):
            raise ValueError("Invalid target identifier")
        target = targets[name]
        coordinates = target.get("identity", {}) if isinstance(target, dict) else {}
        if (set(coordinates) != {"runtime", "source", "profile", "variant"}
                or any(not isinstance(v, str) or not ID.fullmatch(v) for v in coordinates.values())):
            raise ValueError("Invalid target coordinates")
        if coordinates["variant"] != "browser":
            raise ValueError("Only browser artifacts belong in the browser pack")
        values[name] = read_input(root, target["raw"])
    if pairs is None:
        pairs = [(a, b) for a in values for b in values if a != b
                 and targets[a]["identity"]["runtime"] == targets[b]["identity"]["runtime"]]
    if (not isinstance(pairs, list) or len(pairs) > 4096
            or any(not isinstance(p, (tuple, list)) or len(p) != 2
                   or any(not isinstance(v, str) for v in p) for p in pairs)
            or len(set(map(tuple, pairs))) != len(pairs)):
        raise ValueError("Invalid pair list")
    for a, b in pairs:
        if (a not in values or b not in values or a == b
                or targets[a]["identity"]["runtime"] != targets[b]["identity"]["runtime"]):
            raise ValueError("Invalid pair")
    if pairs and encoder is None:
        raise ValueError("Delta pairs require the native Rust encoder")

    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {
        "encoder": "bicore-wasm-pack-rust-v1",
        "codecSettings": {"gzip": 9, "brotli": 11, "zstd": {"level": 19, "windowLog": 20}},
        "structuralPrediction": encoder is not None, "pairs": [],
    }
    if "zstd" in codecs:
        from zstd_codec import version
        report["zstdVersion"] = version
    # A sibling temporary directory keeps the final publication on one filesystem.
    with tempfile.TemporaryDirectory(prefix=".wasm-pack-", dir=destination.parent) as tmp:
        work = Path(tmp)
        out = work / "output"
        (out / "data").mkdir(parents=True)
        (out / "runtime").mkdir()
        catalog = {
            "formatVersion": 1, "decoderApiVersion": 1, "targets": {}, "assets": {},
            "runtime": {
                "entry": "runtime/loader.mjs" if any(c != "zstd" for c in codecs) else "runtime/loader-zstd.mjs",
                "entries": {c: "runtime/loader-zstd.mjs" if c == "zstd" else "runtime/loader.mjs" for c in codecs},
                "files": [],
            },
        }

        def asset(data: bytes, codec: str) -> str:
            packed = compress(data, codec)
            meta = identity(packed)
            key = meta["sha256"]
            relative = f"data/{key}." + {"gzip": "gz", "brotli": "br", "zstd": "zst"}[codec]
            entry = {"path": relative, **meta, "codec": codec, "decoded": identity(data)}
            if key in catalog["assets"] and catalog["assets"][key] != entry:
                raise ValueError("Asset collision")
            catalog["assets"][key] = entry
            (out / relative).write_bytes(packed)
            return key

        for name, data in values.items():
            catalog["targets"][name] = {
                "identity": targets[name]["identity"], "raw": targets[name]["raw"],
                "representations": [
                    {"id": f"full-{codec}", "kind": "full", "payload": asset(data, codec)}
                    for codec in codecs
                ],
            }
            (work / (name + ".wasm")).write_bytes(data)
            if encoder is not None:
                subprocess.run([str(encoder), "validate", str(work / (name + ".wasm"))],
                               check=True, timeout=180)

        def runtime_file(path: str, data: bytes, selected_codecs, role="always", **extra):
            (out / path).parent.mkdir(parents=True, exist_ok=True)
            (out / path).write_bytes(data)
            catalog["runtime"]["files"].append({
                "path": path, **identity(data), "codecs": list(selected_codecs), "role": role, **extra,
            })

        for file in sorted((ROOT / "runtime").glob("*.mjs")):
            if file.name in ("loader-zstd.mjs", "zstd-loader.mjs"):
                selected_codecs = [c for c in codecs if c == "zstd"]
            elif file.name == "loader.mjs":
                selected_codecs = [c for c in codecs if c != "zstd"]
            else:
                selected_codecs = codecs
            if selected_codecs:
                runtime_file("runtime/" + file.name, file.read_bytes(), selected_codecs)
        if "zstd" in codecs:
            decoder = Path(zstd_decoder).read_bytes()
            if len(decoder) > 4 * 1024 * 1024 or decoder[:8] != b"\x00asm\x01\x00\x00\x00":
                raise ValueError("Invalid decoder binary")
            runtime_file("runtime/zstd-decoder.wasm.gz",
                         gzip.compress(decoder, compresslevel=9, mtime=0), ["zstd"],
                         role="zstd", decoded=identity(decoder))
        # One repository license source; no separately maintained duplicate.
        runtime_file("LICENSE", (ROOT.parent / "LICENSE").read_bytes(), codecs)
        if "zstd" in codecs:
            for notice in sorted((ROOT / "licenses").glob("*.txt")):
                runtime_file("licenses/" + notice.name, notice.read_bytes(), ["zstd"])

        for pair_number, (a, b) in enumerate(pairs):
            pair_started = time.monotonic()
            candidates = encode_pair(Path(encoder), work / (a + ".wasm"),
                                     work / (b + ".wasm"), work / f"pair-{pair_number}")
            sizes = {}
            for candidate in candidates:
                for codec in codecs:
                    payload = asset(candidate["recipe"], codec)
                    representation = {
                        "id": f"delta-{a}-{codec}-{candidate['label']}",
                        "kind": "delta", "base": a, "payload": payload,
                    }
                    if "prediction" in candidate:
                        representation["prediction"] = asset(candidate["prediction"], codec)
                        representation["predicted"] = identity(candidate["dictionary"])
                    catalog["targets"][b]["representations"].append(representation)
                    prediction_size = (catalog["assets"][representation["prediction"]]["bytes"]
                                       if "prediction" in representation else 0)
                    sizes[f"{codec}-{candidate['label']}"] = catalog["assets"][payload]["bytes"] + prediction_size
            report["pairs"].append({"base": a, "target": b,
                                    "seconds": time.monotonic() - pair_started, "sizes": sizes})
        (out / "catalog.json").write_text(json.dumps(catalog, sort_keys=True, indent=2) + "\n")
        subprocess.run(["node", str(ROOT / "verify-pack.mjs"), str(out), str(root)],
                       check=True, timeout=max(180, len(pairs) * 15))
        if destination.exists():
            raise ValueError("Output appeared concurrently")
        out.rename(destination)
    report["seconds"] = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codecs", nargs="+", choices=CODECS, default=["gzip", "brotli"])
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--zstd-decoder", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = build(args.root, json.loads(args.spec.read_text()), args.output,
                   codecs=args.codecs, encoder=args.encoder, zstd_decoder=args.zstd_decoder,
                   pairs=json.loads(args.pairs.read_text()) if args.pairs else None)
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
