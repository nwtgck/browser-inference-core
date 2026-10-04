"""Cross-language contracts: Rust generates, shipped JavaScript reconstructs."""
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

from test_pack import ROOT, leb, wasm_fixture


class NativeEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = Path(os.environ.get(
            "BIC_TEST_WASM_ENCODER", ROOT.parent / "build/wasm-pack-tools/wasm-encoder"))
        if not cls.encoder.is_file():
            raise RuntimeError("Build the native encoder before testing")

    def test_randomized_modules_independent_decoder_and_execution(self):
        rng = random.Random(59339)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = []
            for index in range(64):
                before = (63, 64, 127, 128, 256, 8191, 8192, 16384)[index % 8]
                after = before + 1 + index % 3
                a = wasm_fixture(before)
                b = wasm_fixture(after)
                noise = bytes(rng.randrange(256) for _ in range(index * 19))
                content = b"\x01x" + noise
                a += b"\x00" + leb(len(content)) + content
                content = b"\x01x" + noise[:index] + b"changed" + noise[index:] * 2
                b += b"\x00" + leb(len(content)) + content
                base = root / f"base-{index}.wasm"
                target = root / f"target-{index}.wasm"
                output = root / f"pair-{index}"
                base.write_bytes(a)
                target.write_bytes(b)
                subprocess.run([str(self.encoder), "pair", str(base), str(target), str(output)],
                               check=True, capture_output=True, timeout=30)
                cases.append({"base": str(base), "target": str(target),
                              "output": str(output), "expected": after})
            (root / "cases.json").write_text(json.dumps(cases))
            subprocess.run(["node", str(ROOT / "tests/native-roundtrip.mjs"), str(root / "cases.json")],
                           check=True, timeout=30)

    def test_rejects_invalid_inputs_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "good.wasm"
            bad = root / "bad.wasm"
            good.write_bytes(wasm_fixture(256))
            bad.write_bytes(b"\0asm\1\0\0\0\x0a\xff")
            result = subprocess.run([str(self.encoder), "validate", str(bad)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            output = root / "output"
            output.mkdir()
            (output / "sentinel").write_text("keep")
            result = subprocess.run([str(self.encoder), "pair", str(good), str(good), str(output)],
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((output / "sentinel").read_text(), "keep")
            linked = root / "linked.wasm"
            linked.symlink_to(good)
            result = subprocess.run([str(self.encoder), "validate", str(linked)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_pair_is_deterministic_and_inputs_are_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = root / "a.wasm", root / "b.wasm"
            a.write_bytes(wasm_fixture(256))
            b.write_bytes(wasm_fixture(257))
            before = (a.read_bytes(), b.read_bytes())
            for suffix in ("first", "second"):
                subprocess.run([str(self.encoder), "pair", str(a), str(b), str(root / suffix)], check=True)
            one = {p.name: p.read_bytes() for p in (root / "first").iterdir() if p.name != "result.json"}
            two = {p.name: p.read_bytes() for p in (root / "second").iterdir() if p.name != "result.json"}
            self.assertEqual(one, two)
            self.assertEqual(before, (a.read_bytes(), b.read_bytes()))

    def test_producer_dependencies_do_not_include_experiment_tooling(self):
        for name in ("analysis.py", "align.py", "delta.py", "prediction.py", "hash_matcher.cpp"):
            self.assertFalse((ROOT / "encoder" / name).exists())
        self.assertFalse((ROOT / "requirements.txt").exists())
        self.assertFalse((ROOT / "LICENSE").exists())
        setup = (ROOT / "prepare_ci_tools.py").read_text()
        self.assertNotIn('"pip"', setup)
        self.assertNotIn('"g++"', setup)
        self.assertNotIn("pack-python", setup)
