"""Small source fixtures; no upstream downloads or generated-code rewriting."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import unittest

from build_tools.source.patches import Patch

PATCH = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1,2 +1,2 @@
 one
-before
+after
"""


class SourceFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = {name: self.root / name for name in ("stable-diffusion", "ggml")}
        for source in self.sources.values():
            source.mkdir()
        (self.sources["stable-diffusion"] / "a.txt").write_text("one\nbefore\n")
        self.patches = self.root / "patches"
        self.patches.mkdir()
        (self.patches / "fix.patch").write_text(PATCH)
        self.entry = {"file": "fix.patch", "target": "stable-diffusion", "reason": "fixture correction", "files": ["a.txt"]}
        self.inventory = {"formatVersion": 2, "patches": [self.entry]}
        self.write_inventory()
        self.output = self.root / "prepared"

    def write_inventory(self):
        (self.patches / "series.json").write_text(json.dumps(self.inventory))

    def declaration(self):
        return Patch(self.entry["file"], self.entry["target"], self.entry["reason"], tuple(self.entry["files"]))

    def git(self, where, *args, **kwargs):
        return subprocess.check_output(["git", "-C", str(where), *args], **kwargs)
