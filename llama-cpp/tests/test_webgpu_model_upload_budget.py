"""Exercise the actual loader-overlay helper with CPU queues, never a GPU claim."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source


class WebgpuModelUploadBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.LOADER_PATH).is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def prepared(self, directory):
        source.prepare(self.upstream, directory)
        return (directory / 'llama-model-loader.cpp').read_text()

    def test_actual_overlay_scope_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = self.prepared(Path(tmp))
        function = text[text.index('bool llama_model_loader::load_all_data('):]
        self.assertEqual(function.count('llama_webgpu_load_uploads webgpu_uploads;'), 1)
        self.assertLess(function.index('llama_webgpu_load_uploads webgpu_uploads;'),
                        function.index('for (struct ggml_tensor * cur : tensors)'))
        self.assertIn('''if (chunked) {
                            webgpu_uploads.reserve(dev, read_size);
                        }
                        ggml_backend_tensor_set(cur, read_buf.data(), offset, read_size);''', function)
        self.assertIn('auto * dev = ggml_backend_buft_get_device(buft);', function)
        self.assertIn('buft == ggml_backend_dev_buffer_type(dev)', function)
        self.assertIn('strcmp(ggml_backend_reg_name(reg), "WebGPU") == 0', function)
        self.assertIn('(8 * MiB / chunk_alignment) * chunk_alignment', function)
        self.assertIn('webgpu_uploads.drain();\n\n    // free temporary resources', function)
        self.assertEqual(function.count('webgpu_uploads.reserve('), 1)
        self.assertEqual(text.count('ggml_backend_synchronize(backend);'), 1)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_helper_with_deterministic_cpu_queues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = self.prepared(root / 'overlay')
            start = text.index('struct llama_webgpu_load_uploads {')
            end = text.index('\n};', start) + 3
            helper = text[start:end]
            harness = r'''
#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
static constexpr size_t MiB = 1024 * 1024;
struct device { int id; size_t queued = 0; };
using ggml_backend_dev_t = device *;
struct backend { device * dev; };
using ggml_backend_t = backend *;
static int live = 0, waits = 0;
static bool fail_init = false;
static std::vector<int> trace;
ggml_backend_t ggml_backend_dev_init(device * dev, const char *) {
    trace.push_back(10 + dev->id);
    if (fail_init) return nullptr;
    ++live; return new backend{dev};
}
void ggml_backend_synchronize(backend * b) {
    assert(b); ++waits; trace.push_back(20 + b->dev->id); b->dev->queued = 0;
}
void ggml_backend_free(backend * b) {
    if (b) { assert(b->dev->queued == 0); --live; trace.push_back(30 + b->dev->id); delete b; }
}
'''
            harness += helper
            harness += r'''
void upload(llama_webgpu_load_uploads & loader, device & d, size_t size) {
    loader.reserve(&d, size);
    d.queued += size + loader.overhead;
    assert(d.queued <= loader.budget);
    assert(loader.pending == d.queued);
    trace.push_back(40 + d.id);
}
int main() {
    using loader = llama_webgpu_load_uploads;
    device a{1}, b{2};
    { loader l; } assert(live == 0 && waits == 0); // CPU-only/no uploads
    { loader l;
      upload(l, a, 8 * MiB); upload(l, a, 8 * MiB); upload(l, a, 8 * MiB);
      assert(waits == 0); // carries across tensors, not drained per call
      upload(l, a, 8 * MiB); assert(waits == 1);
      assert(trace[trace.size()-2] == 21 && trace.back() == 41); // before write
    } assert(live == 0 && a.queued == 0 && waits == 2); // final partial
    // Alignment-rounded maxima and variable tails use bytes, not a write count.
    for (size_t rounded : {size_t(8388288), size_t(8388480)}) {
      loader l; int before = waits;
      for (int i = 0; i < 4; ++i) upload(l, a, rounded);
      assert(waits == before);
      upload(l, a, 4096); assert(waits == before + 1);
    } assert(live == 0 && a.queued == 0);
    { loader l;
      int before = waits;
      upload(l, a, 8 * MiB); upload(l, a, 8 * MiB); upload(l, a, 8 * MiB);
      upload(l, a, 4 * MiB); assert(waits == before);
      upload(l, a, 4 * MiB); assert(waits == before + 1);
    } assert(live == 0 && a.queued == 0);
    { loader l;
      upload(l, a, loader::budget - loader::overhead); // equality allowed
      int before = waits; upload(l, a, 1); assert(waits == before + 1);
      l.drain(); before = waits; l.drain(); assert(waits == before);
    } assert(live == 0);
    { loader l;
      upload(l, a, 7); int before = waits;
      upload(l, b, 5); assert(waits == before + 1 && a.queued == 0);
      assert(trace[trace.size()-4] == 21); // drain -> free -> init -> write
      assert(trace[trace.size()-3] == 31 && trace[trace.size()-2] == 12);
      upload(l, a, 3); assert(b.queued == 0);
    } assert(live == 0 && a.queued == 0 && b.queued == 0);
    try { loader l; upload(l, a, 12); throw std::runtime_error("read/validation failure"); }
    catch (const std::runtime_error &) {} assert(live == 0 && a.queued == 0);
    auto cancel = [&]() { loader l; upload(l, a, 13); return false; };
    assert(!cancel() && live == 0 && a.queued == 0);
    { loader l;
      for (size_t invalid : {loader::budget, std::numeric_limits<size_t>::max()}) {
        bool threw = false; try { l.reserve(&a, invalid); } catch (const std::runtime_error &) { threw = true; }
        assert(threw && l.pending == 0 && l.backend == nullptr);
      }
      upload(l, a, 10); fail_init = true;
      bool threw = false; try { l.reserve(&b, 1); } catch (const std::runtime_error &) { threw = true; }
      assert(threw && l.pending == 0 && l.backend == nullptr && live == 0 && a.queued == 0);
      fail_init = false; upload(l, b, 1);
    } assert(live == 0);
    // Many tiny tensors cannot evade byte accounting through auxiliary writes.
    { loader l; for (int i = 0; i < 100000; ++i) upload(l, a, 400); }
    assert(live == 0 && a.queued == 0);
    // Simulate a tensor_set exception after some work was submitted.
    try { loader l; l.reserve(&a, 4096); a.queued = 4096; throw std::runtime_error("partial submission"); }
    catch (const std::runtime_error &) {} assert(live == 0 && a.queued == 0);
}
'''
            cpp = root / 'budget.cpp'; cpp.write_text(harness)
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(root / 'budget')], check=True)
            subprocess.run([str(root / 'budget')], check=True)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_complete_loader_native_syntax_with_pinned_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.prepared(root)
            command = ['c++', '-std=c++17', '-fsyntax-only']
            command += ['-I' + str(self.upstream / path) for path in ('src', 'include', 'ggml/include')]
            subprocess.run(command + [str(root / 'llama-model-loader.cpp')], check=True)


if __name__ == '__main__':
    unittest.main()
