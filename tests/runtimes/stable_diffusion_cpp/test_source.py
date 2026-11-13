import hashlib
from pathlib import Path
import json
from unittest.mock import patch
from build_tools.source.checkout import Checkout
import subprocess
import sys

from build_tools.runtimes.stable_diffusion_cpp import source
from tests.build_tools.source.support import SourceFixture


class ImageSource(SourceFixture):
    def test_independent_sources_are_prepared_through_shared_mechanism(self):
        report = source.prepare(self.sources, self.output, self.patches)
        self.assertEqual(len(report["patches"]), 1)
        self.assertEqual((self.output / "stable-diffusion/a.txt").read_text(), "one\nafter\n")

    def add_new_file_patch(self):
        (self.patches / "new.patch").write_text(
            "diff --git a/src/core/fixture.h b/src/core/fixture.h\n"
            "new file mode 100644\n--- /dev/null\n+++ b/src/core/fixture.h\n"
            "@@ -0,0 +1 @@\n+fixture\n")
        self.inventory["patches"].append({
            "file": "new.patch", "target": "stable-diffusion", "reason": "new header fixture",
            "files": ["src/core/fixture.h"],
        })
        self.write_inventory()

    def test_new_file_records_absent_input_and_actual_output(self):
        self.add_new_file_patch()
        result = source.prepare(self.sources, self.output, self.patches)
        output = self.output / "stable-diffusion/src/core/fixture.h"
        self.assertEqual(output.read_bytes(), b"fixture\n")
        self.assertFalse((self.sources["stable-diffusion"] / "src").exists())
        receipt = result["patches"][1]["files"][0]
        self.assertIsNone(receipt["beforeSha256"])
        self.assertEqual(receipt["afterSha256"], hashlib.sha256(output.read_bytes()).hexdigest())

    def test_new_file_cannot_replace_an_existing_file_directory_or_dangling_link(self):
        self.add_new_file_patch()
        original = self.sources["stable-diffusion"] / "src/core/fixture.h"
        original.parent.mkdir(parents=True)
        for kind in ("file", "directory", "dangling-link"):
            with self.subTest(kind=kind):
                if kind == "file":
                    original.write_bytes(b"original\n")
                elif kind == "directory":
                    original.mkdir()
                else:
                    original.symlink_to("missing.h")
                with self.assertRaises(ValueError):
                    source.prepare(self.sources, self.output, self.patches)
                self.assertFalse(self.output.exists())
                if kind == "file":
                    self.assertEqual(original.read_bytes(), b"original\n")
                elif kind == "dangling-link":
                    self.assertTrue(original.is_symlink())
                original.rmdir() if kind == "directory" else original.unlink()

    def test_unknown_role_is_not_silently_mapped_to_another_upstream(self):
        with self.assertRaisesRegex(ValueError, "Unknown image-runtime source set"):
            source.prepare({"llama": self.sources["ggml"]}, self.output, self.patches)

    def test_module_entrypoint_is_available_without_search_path_patches(self):
        result = subprocess.run([sys.executable, "-m", source.__name__, "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)


class ImageSourceDeclarations(SourceFixture):
    def setUp(self):
        super().setUp()
        self.repository = self.root / "repository"
        self.runtime = self.repository / "runtimes/stable-diffusion-cpp"
        (self.runtime / "config").mkdir(parents=True)
        self.config = self.runtime / "config/upstreams.json"
        self.declarations = {
            "stable-diffusion": {"path": "vendor/sd", "repository": "example/sd"},
            "ggml": {"path": "vendor/ggml", "repository": "example/ggml", "sourceSubdirectory": "ggml"},
        }
        self.config.write_text(json.dumps(self.declarations))
        (self.sources["ggml"] / "ggml").mkdir()

    def test_roles_resolve_gitlinks_not_another_config_commit_copy(self):
        checked = {role: Checkout(path, "a" * 40) for role, path in self.sources.items()}
        with patch.object(source, "checked_sources", return_value=checked) as resolve:
            paths, provenance = source.resolve_sources(self.repository, self.runtime)
        self.assertEqual(paths, self.sources)
        self.assertEqual(provenance["ggml"]["commit"], "a" * 40)
        resolve.assert_called_once_with(self.repository, {
            "stable-diffusion": "runtimes/stable-diffusion-cpp/vendor/sd",
            "ggml": "runtimes/stable-diffusion-cpp/vendor/ggml",
        }, revision=None)
        self.assertNotIn("commit", self.declarations["ggml"])

    def test_committed_declarations_and_gitlinks_use_the_same_revision(self):
        saved = self.config.read_text()
        self.config.write_text('{"changed": "must not be read"}')
        checked = {role: Checkout(path, 'a' * 40) for role, path in self.sources.items()}
        with patch.object(source, 'commit_file', return_value=saved) as read, \
             patch.object(source, 'checked_sources', return_value=checked) as resolve:
            _, provenance = source.resolve_sources(self.repository, self.runtime, revision='b' * 40)
        read.assert_called_once_with(self.repository, 'b' * 40, 'runtimes/stable-diffusion-cpp/config/upstreams.json')
        self.assertEqual(resolve.call_args.kwargs, {'revision': 'b' * 40})
        self.assertEqual(provenance['ggml']['repository'], 'example/ggml')

    def test_duplicate_json_keys_are_rejected_before_checkout(self):
        self.config.write_text('{"ggml":{},"ggml":{},"stable-diffusion":{}}')
        with patch.object(source, "checked_sources") as resolve, self.assertRaisesRegex(ValueError, "Duplicate"):
            source.resolve_sources(self.repository, self.runtime)
        resolve.assert_not_called()

    def test_config_shape_and_subdirectory_are_explicit(self):
        for key, value in (("commit", "a" * 40), ("sourceSubdirectory", "../outside"), ("path", "/outside")):
            with self.subTest(key=key):
                broken = json.loads(json.dumps(self.declarations))
                broken["ggml"][key] = value
                self.config.write_text(json.dumps(broken))
                with patch.object(source, "checked_sources") as resolve, self.assertRaises(ValueError):
                    source.resolve_sources(self.repository, self.runtime)
                resolve.assert_not_called()
        del self.declarations["ggml"]["sourceSubdirectory"]
        self.config.write_text(json.dumps(self.declarations))
        with self.assertRaisesRegex(ValueError, "fields"):
            source.resolve_sources(self.repository, self.runtime)

    def test_missing_declared_subdirectory_cannot_be_substituted(self):
        self.declarations["ggml"]["sourceSubdirectory"] = "missing"
        self.config.write_text(json.dumps(self.declarations))
        checked = {role: Checkout(path, "a" * 40) for role, path in self.sources.items()}
        with patch.object(source, "checked_sources", return_value=checked), self.assertRaisesRegex(ValueError, "Missing"):
            source.resolve_sources(self.repository, self.runtime)
