import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from build_tools.source.files import relative_path
from build_tools.source.patches import apply_patch, read_series
from tests.build_tools.source.support import SourceFixture, PATCH


class PatchSeries(SourceFixture):
    def test_reads_ordered_declarative_inventory_and_records_observed_identity(self):
        series = read_series(self.patches / "series.json", targets=self.sources)
        self.assertEqual(series.patches, (self.declaration(),))
        self.assertEqual(series.sha256, hashlib.sha256((self.patches / "series.json").read_bytes()).hexdigest())

    def test_rejects_unknown_target(self):
        self.entry["target"] = "elsewhere"
        self.write_inventory()
        with self.assertRaisesRegex(ValueError, "Unknown patch target"):
            read_series(self.patches / "series.json", targets=self.sources)

    def test_rejects_duplicate_patch_entries(self):
        self.inventory["patches"].append(dict(self.entry))
        self.write_inventory()
        with self.assertRaisesRegex(ValueError, "Duplicate patch"):
            read_series(self.patches / "series.json", targets=self.sources)

    def test_rejects_unknown_fields_and_old_hash_pins(self):
        for name in ("command", "url", "beforeSha256"):
            with self.subTest(name=name):
                self.entry[name] = "not an instruction"
                self.write_inventory()
                with self.assertRaises(ValueError):
                    read_series(self.patches / "series.json", targets=self.sources)
                del self.entry[name]

    def test_rejects_duplicate_json_keys(self):
        (self.patches / "series.json").write_text('{"formatVersion":2,"formatVersion":2,"patches":[]}')
        with self.assertRaisesRegex(ValueError, "Duplicate inventory key"):
            read_series(self.patches / "series.json", targets=self.sources)

    def test_rejects_unsupported_versions_including_boolean(self):
        for version in (True, 1, 3, "2", None):
            with self.subTest(version=version):
                self.inventory["formatVersion"] = version
                self.write_inventory()
                with self.assertRaises(ValueError):
                    read_series(self.patches / "series.json", targets=self.sources)

    def test_rejects_empty_and_duplicate_file_inventory(self):
        for paths in ([], ["a.txt", "a.txt"]):
            with self.subTest(paths=paths):
                self.entry["files"] = paths
                self.write_inventory()
                with self.assertRaisesRegex(ValueError, "Empty or duplicate"):
                    read_series(self.patches / "series.json", targets=self.sources)

    def test_accepts_spaces_unicode_but_not_noncanonical_or_metadata_paths(self):
        self.assertEqual(str(relative_path("some dir/資料.cpp")), "some dir/資料.cpp")
        for name in ("", ".", "..", "../outside", "/absolute", "a/../b", "a//b", "a/./b", "./a", "a/", ".git/config", "a/.GIT/b", "a\\b", "a\nb"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                relative_path(name)


class PatchApplication(SourceFixture):
    def apply(self):
        return apply_patch(self.sources["stable-diffusion"], self.patches, self.declaration())

    def test_records_actual_before_and_after_without_authored_source_hashes(self):
        result = self.apply()
        self.assertEqual((self.sources["stable-diffusion"] / "a.txt").read_text(), "one\nafter\n")
        self.assertEqual(result["files"], [{"path": "a.txt", "beforeSha256": hashlib.sha256(b"one\nbefore\n").hexdigest(), "afterSha256": hashlib.sha256(b"one\nafter\n").hexdigest()}])
        self.assertEqual(result["sha256"], hashlib.sha256(PATCH.encode()).hexdigest())

    def test_unrelated_comment_does_not_require_a_hash_update(self):
        path = self.sources["stable-diffusion"] / "a.txt"
        # A patch is still context-sensitive. This comment is outside its hunk,
        # not an assertion that arbitrary changes inside the hunk must apply.
        path.write_text("header\none\nbefore\ntail\n// unrelated comment\n")
        (self.patches / "fix.patch").write_text(
            "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n"
            "@@ -2,3 +2,3 @@\n one\n-before\n+after\n tail\n")
        self.apply()
        self.assertEqual(path.read_text(), "header\none\nafter\ntail\n// unrelated comment\n")

    def test_real_context_change_is_diagnosed_as_patch_application_failure(self):
        path = self.sources["stable-diffusion"] / "a.txt"
        path.write_text("one\nchanged contract\n")
        with self.assertRaisesRegex(ValueError, "Patch application failed"):
            self.apply()
        self.assertEqual(path.read_text(), "one\nchanged contract\n")

    def test_rejects_undeclared_edits_before_any_apply(self):
        (self.sources["stable-diffusion"] / "b.txt").write_text("one\nbefore\n")
        (self.patches / "fix.patch").write_text(PATCH + PATCH.replace("a.txt", "b.txt"))
        with self.assertRaisesRegex(ValueError, "outside its inventory"):
            self.apply()
        self.assertEqual((self.sources["stable-diffusion"] / "a.txt").read_text(), "one\nbefore\n")

    def test_receipt_follows_declared_order_not_patch_hunk_order(self):
        (self.sources["stable-diffusion"] / "b.txt").write_text("one\nbefore\n")
        (self.patches / "fix.patch").write_text(PATCH + PATCH.replace("a.txt", "b.txt"))
        self.entry["files"] = ["b.txt", "a.txt"]
        self.assertEqual([record["path"] for record in self.apply()["files"]], ["b.txt", "a.txt"])

    def test_rejects_linked_patch_and_linked_target(self):
        path = self.patches / "fix.patch"
        path.rename(self.root / "outside.patch")
        path.symlink_to(self.root / "outside.patch")
        with self.assertRaisesRegex(ValueError, "Linked"):
            self.apply()
        path.unlink()
        path.write_text(PATCH)
        target = self.sources["stable-diffusion"] / "a.txt"
        target.rename(self.root / "outside.txt")
        target.symlink_to(self.root / "outside.txt")
        with self.assertRaisesRegex(ValueError, "Linked"):
            self.apply()
        self.assertEqual((self.root / "outside.txt").read_text(), "one\nbefore\n")

    def test_cannot_patch_through_a_symlinked_directory(self):
        source = self.sources["stable-diffusion"]
        (source / "dir").symlink_to(self.sources["ggml"], target_is_directory=True)
        (self.sources["ggml"] / "a.txt").write_text("one\nbefore\n")
        (self.patches / "fix.patch").write_text(PATCH.replace("a.txt", "dir/a.txt"))
        self.entry["files"] = ["dir/a.txt"]
        with self.assertRaisesRegex(ValueError, "Linked"):
            self.apply()

    def test_git_environment_and_parent_repository_cannot_redirect_the_patch(self):
        self.git(self.root, "init", "-q")
        (self.root / "a.txt").write_text("parent must not change\n")
        self.git(self.root, "add", "a.txt")
        parent_index = (self.root / ".git/index").read_bytes()
        malicious = {"GIT_DIR": str(self.root / ".git"), "GIT_WORK_TREE": str(self.root),
                     "GIT_INDEX_FILE": str(self.root / ".git/index"), "GIT_CONFIG_COUNT": "1",
                     "GIT_CONFIG_KEY_0": "apply.ignoreWhitespace", "GIT_CONFIG_VALUE_0": "change"}
        with patch.dict(os.environ, malicious):
            self.apply()
        self.assertEqual((self.root / "a.txt").read_text(), "parent must not change\n")
        self.assertEqual((self.root / ".git/index").read_bytes(), parent_index)

    def test_rejects_git_metadata_path(self):
        (self.patches / "fix.patch").write_text(PATCH.replace("a.txt", ".git/config"))
        with self.assertRaises(ValueError):
            self.apply()

    def test_added_and_deleted_files_are_explicit_in_the_receipt(self):
        target = self.sources["stable-diffusion"] / "new.h"
        (self.patches / "fix.patch").write_text("diff --git a/new.h b/new.h\nnew file mode 100644\n--- /dev/null\n+++ b/new.h\n@@ -0,0 +1 @@\n+int value;\n")
        self.entry["files"] = ["new.h"]
        added = self.apply()
        self.assertIsNone(added["files"][0]["beforeSha256"])
        (self.patches / "fix.patch").write_text("diff --git a/new.h b/new.h\ndeleted file mode 100644\n--- a/new.h\n+++ /dev/null\n@@ -1 +0,0 @@\n-int value;\n")
        removed = self.apply()
        self.assertIsNone(removed["files"][0]["afterSha256"])
        self.assertFalse(target.exists())

    def test_link_gitlink_and_rename_patches_are_not_regular_source_patches(self):
        for mode in ("120000", "160000"):
            with self.subTest(mode=mode):
                (self.patches / "fix.patch").write_text("diff --git a/link b/link\nnew file mode " + mode + "\n--- /dev/null\n+++ b/link\n@@ -0,0 +1 @@\n+outside\n")
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    self.apply()
        (self.patches / "fix.patch").write_text("diff --git a/a.txt b/b.txt\nsimilarity index 100%\nrename from a.txt\nrename to b.txt\n")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.apply()
