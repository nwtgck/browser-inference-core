"""Focused regression for the mtmd allocation hunk, not a full upstream build.

The short input below is independent of the patch under test. The separate
real-upstream tests still apply every hunk and compile the full translation unit.
"""
import concurrent.futures
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path.insert(0, str(ROOT.parents[1]))
from build_tools.runtimes.llama_cpp.sources.mtmd import PATCH_DIRECTORY
import os
import shutil

# Allocation boundary in tools/mtmd/mtmd-subcontext.cpp:
# 202c1a229f5a83d1c9d758ed0f2d43760fa4bab0. Do not derive this input from the
# patch's minus/context lines: doing so would also accept the obsolete backport.
ALLOCATION_GUARD = r'''    if (!ggml_backend_sched_alloc_graph(backends.sched.get(), gf)) {
        LOG_ERR("%s: failed to allocate compute graph\n", __func__);
        return false;
    }
'''
UPSTREAM_FRAGMENT = (
    "// Standalone regression fragment; unrelated loader code is omitted.\n"
    "    // build the inference graph\n"
    "    ggml_backend_sched_reset(backends.sched.get());\n"
    "    ggml_cgraph * gf = mtmd_get_graph_builder(this, imgs, params)->build();\n"
    "\n"
    "    // the last node is the embedding tensor, code2wav has no out_embd\n"
    "    // read it before scheduling: backends may reorder the nodes of graphs with several outputs (Metal)\n"
    "    ggml_tensor * embeddings = params->out_embd ? ggml_graph_node(gf, -1) : nullptr;\n"
    "\n"
    + ALLOCATION_GUARD +
    "\n"
    "    // set inputs\n"
    "    const auto & hparams = model.hparams;\n"
    "// End of the standalone regression fragment.\n"
)


class MtmdAllocationGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lcb-allocation-test-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.source = self.work / "upstream"
        self.input = self.source / "tools/mtmd/mtmd-subcontext.cpp"
        self.input.parent.mkdir(parents=True)
        self.input.write_text(UPSTREAM_FRAGMENT)
        patch = (ROOT / PATCH_DIRECTORY / "mtmd-webgpu-bf16.patch").read_text()
        parts = re.split(r"(?=^@@ )", patch, flags=re.MULTILINE)
        hunks = [part for part in parts[1:] if "lcb_clip: matmul placement" in part]
        self.assertEqual(len(hunks), 1, "Review the allocation-boundary regression after a patch layout change")
        self.hunk = hunks[0]
        # Run the actual production preparer and actual diagnostic hunk. This
        # fixture deliberately does not pretend to test unrelated loader hunks.
        self.patch = self.work / "placement.patch"
        self.patch.write_text('--- a/mtmd-subcontext.cpp\n+++ b/mtmd-subcontext.cpp\n' + self.hunk)

    def apply(self, output=None):
        # The full production preparer checks the complete subtree in its own
        # tests. This focused fixture applies the actual diagnostic hunk only.
        destination = output or self.work / 'overlay'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            tree = Path(temporary)
            shutil.copy2(self.input, tree / 'mtmd-subcontext.cpp')
            env = os.environ.copy()
            for key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE'):
                env.pop(key, None)
            env['GIT_CEILING_DIRECTORIES'] = str(tree.parent)
            for args in (['--check'], []):
                subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *args, str(self.patch)],
                               cwd=tree, env=env, check=True, capture_output=True)
            target = destination / 'mtmd-subcontext.cpp'
            raw = (tree / target.name).read_bytes()
            destination.mkdir(exist_ok=True)
            if not target.exists() or raw != target.read_bytes():
                target.write_bytes(raw)
        return target

    def test_upstream_guard_is_context_not_another_downstream_backport(self):
        target = self.apply()
        result = target.read_text()
        self.assertEqual(result.count(ALLOCATION_GUARD), 1)
        self.assertEqual(result.count("ggml_backend_sched_alloc_graph("), 1)
        self.assertEqual(result.count("lcb_clip: matmul placement"), 1)
        # The allocation, error log and return must remain upstream-owned.
        changed = "\n".join(line[1:] for line in self.hunk.splitlines() if line.startswith(("+", "-")))
        self.assertNotIn("ggml_backend_sched_alloc_graph", changed)
        self.assertNotIn("failed to allocate compute graph", changed)
        removed = "\n".join(line[1:] for line in self.hunk.splitlines() if line.startswith("-"))
        self.assertNotIn("return false;", removed)
        # New retained-mode failures also return false; they do not replace the
        # upstream-owned allocation failure boundary.
        self.assertIn("retained_plan.verify_allocated", result)
        guard_end = result.index(ALLOCATION_GUARD) + len(ALLOCATION_GUARD)
        diagnostic_start = result.index("    if (model.modality == MTMD_MODALITY_VISION")
        inputs = result.index("    // set inputs")
        self.assertLess(guard_end, diagnostic_start)
        self.assertLess(diagnostic_start, inputs)
        self.assertIn("lcb_mtmd_is_webgpu(backends.backend)", result)
        self.assertEqual(self.input.read_text(), UPSTREAM_FRAGMENT)

    def test_unrelated_line_movement_still_applies(self):
        prefix = "// An unrelated upstream line.\n" * 64
        self.input.write_text(prefix + UPSTREAM_FRAGMENT)
        result = self.apply().read_text()
        self.assertTrue(result.startswith(prefix))
        self.assertEqual(result.count(ALLOCATION_GUARD), 1)
        self.assertEqual(self.input.read_text(), prefix + UPSTREAM_FRAGMENT)

    def test_guard_drift_aborts_without_publishing_or_reusing_a_stale_copy(self):
        target = self.apply()
        before = target.read_bytes()
        timestamp = target.stat().st_mtime_ns
        for guard in (
            "    ggml_backend_sched_alloc_graph(backends.sched.get(), gf);\n",
            ALLOCATION_GUARD.replace("return false;", "return true;"),
            ALLOCATION_GUARD.replace("if (!ggml_backend_sched_alloc_graph", "if (ggml_backend_sched_alloc_graph"),
        ):
            with self.subTest(guard=guard):
                changed = UPSTREAM_FRAGMENT.replace(ALLOCATION_GUARD, guard)
                self.input.write_text(changed)
                with self.assertRaises(subprocess.CalledProcessError):
                    self.apply()
                self.assertEqual(target.read_bytes(), before)
                self.assertEqual(target.stat().st_mtime_ns, timestamp)
                self.assertEqual(self.input.read_text(), changed)

    def test_already_patched_input_is_rejected(self):
        target = self.apply()
        self.input.write_bytes(target.read_bytes())
        output = self.work / "second-overlay"
        with self.assertRaises(subprocess.CalledProcessError):
            self.apply(output)
        self.assertFalse((output / "mtmd-subcontext.cpp").exists())

    def test_all_webgpu_profile_variants_have_independent_idempotent_copies(self):
        profiles = json.loads((ROOT / "config/profiles.json").read_text())
        variants = json.loads((ROOT / "config/variants.json").read_text())
        outputs = [self.work / profile / variant
                   for profile, info in profiles.items() if info["webgpu"]
                   for variant in variants]
        self.assertTrue(outputs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            targets = list(pool.map(self.apply, outputs))
        self.assertEqual(len(set(targets)), len(outputs))
        expected = targets[0].read_bytes()
        for target, output in zip(targets, outputs):
            with self.subTest(output=output):
                timestamp = target.stat().st_mtime_ns
                self.assertEqual(target.read_bytes(), expected)
                self.assertEqual(self.apply(output), target)
                self.assertEqual(target.stat().st_mtime_ns, timestamp)
        self.assertEqual(self.input.read_text(), UPSTREAM_FRAGMENT)


if __name__ == "__main__":
    unittest.main()
