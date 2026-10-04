"""Exercise build-tree patching without modifying the pinned submodule."""
import concurrent.futures
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_mtmd import prepare


class MtmdOverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lcb-mtmd-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "upstream"
        (self.source / "tools/mtmd").mkdir(parents=True)
        self.file = self.source / "tools/mtmd/clip.cpp"
        self.original = "before\noriginal\nafter\n"
        self.file.write_text(self.original)
        self.patch = self.root / "change.patch"
        self.patch.write_text("--- a/clip.cpp\n+++ b/clip.cpp\n@@ -1,3 +1,3 @@\n before\n-original\n+patched\n after\n")

    def test_reconfigure_is_idempotent_and_preserves_the_upstream_file(self):
        target = prepare(self.source, self.root / "build", self.patch)
        timestamp = target.stat().st_mtime_ns
        self.assertEqual(target.read_text(), "before\npatched\nafter\n")
        prepare(self.source, self.root / "build", self.patch)
        self.assertEqual(target.stat().st_mtime_ns, timestamp)
        self.assertEqual(self.file.read_text(), self.original)

    def test_parallel_profiles_do_not_share_patched_source_files(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            targets = list(pool.map(lambda index: prepare(self.source, self.root / str(index), self.patch), range(3)))
        self.assertEqual(len(set(targets)), 3)
        self.assertTrue(all(target.read_text() == "before\npatched\nafter\n" for target in targets))
        self.assertEqual(self.file.read_text(), self.original)

    def test_changed_upstream_fails_without_overwriting_a_previous_overlay(self):
        target = prepare(self.source, self.root / "build", self.patch)
        self.file.write_text("before\nchanged upstream\nafter\n")
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            prepare(self.source, self.root / "build", self.patch, capture_output=True)
        self.assertIn("patch does not apply", caught.exception.stderr)
        self.assertEqual(target.read_text(), "before\npatched\nafter\n")
        self.assertEqual(self.file.read_text(), "before\nchanged upstream\nafter\n")

    def test_overlapping_source_and_output_are_rejected(self):
        for output in (self.source, self.source / "build", self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                prepare(self.source, output, self.patch)
        self.assertEqual(self.file.read_text(), self.original)

    def test_symlink_to_upstream_is_rejected(self):
        link = self.root / "linked"
        link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            prepare(self.source, link, self.patch)
        self.assertEqual(self.file.read_text(), self.original)

    def test_missing_source_does_not_create_a_patched_file(self):
        self.file.unlink()
        with self.assertRaises(FileNotFoundError):
            prepare(self.source, self.root / "build", self.patch)
        self.assertFalse((self.root / "build/clip.cpp").exists())

    def test_real_upstream_patch_and_compiler_syntax(self):
        upstream = Path(os.environ.get("LCB_TEST_LLAMA_SOURCE", ROOT / "vendor/llama.cpp"))
        if not (upstream / "tools/mtmd/clip.cpp").is_file():
            self.skipTest("Initialize the pinned submodule or set LCB_TEST_LLAMA_SOURCE")
        original = (upstream / "tools/mtmd/clip.cpp").read_bytes()
        patch = ROOT / "upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-webgpu-bf16.patch"
        destination = prepare(upstream, self.root / "real-build", patch)
        # Upstream owns this guard at the v0.5.0 pin. The overlay must preserve
        # it exactly once, before inspecting any scheduler assignments.
        guard = ('    if (!ggml_backend_sched_alloc_graph(ctx->sched.get(), gf)) {\n'
                 '        LOG_ERR("%s: failed to allocate compute graph\\n", __func__);\n'
                 '        return false;\n'
                 '    }\n')
        result = destination.read_text()
        self.assertEqual(original.decode().count(guard), 1)
        self.assertEqual(result.count(guard), 1)
        self.assertEqual(result.count("lcb_clip: matmul placement"), 1)
        self.assertEqual(result.count("lcb_clip: bf16-f32 tensors="), 1)
        self.assertLess(result.index(guard), result.index("lcb_clip: matmul placement"))
        timestamp = destination.stat().st_mtime_ns
        self.assertEqual(prepare(upstream, self.root / "real-build", patch), destination)
        self.assertEqual(destination.stat().st_mtime_ns, timestamp)
        self.assertEqual(destination.read_text(), result)
        self.assertEqual((upstream / "tools/mtmd/clip.cpp").read_bytes(), original)
        compiler = shutil.which("clang++")
        if compiler:
            subprocess.run([compiler, "-std=c++17", "-fsyntax-only", str(destination),
                "-I" + str(upstream / "include"), "-I" + str(upstream / "ggml/include"),
                "-I" + str(upstream / "tools/mtmd"), "-I" + str(ROOT / "bridge")], check=True)
