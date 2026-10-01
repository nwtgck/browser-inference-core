"""Compile the prepared wait functions against deterministic API doubles.

This probes finite deadline selection and error propagation, not WebGPU execution.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def function(source, prefix):
    start = source.index(prefix)
    opening = source.index('{', start)
    depth = 1
    for end in range(opening + 1, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start:end + 1]
    raise ValueError('Unclosed source function: ' + prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cxx', required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    constants = '\n'.join(re.findall(r'^#define WEBGPU_(?:QUEUE|RUNTIME)_WAIT_TIMEOUT_(?:MS|NS)[^\n]*', source, re.M))
    assert len(constants.splitlines()) == 4
    checker = function(source, 'template <typename T>\nstatic void ggml_backend_webgpu_check_wait_status')
    queue = function(source, 'static void ggml_backend_webgpu_wait_queue(')
    mapping = function(source, 'static void ggml_backend_webgpu_map_buffer(')
    event = function(source, 'static void ggml_backend_webgpu_device_event_synchronize(')
    assert 'WEBGPU_RUNTIME_WAIT_TIMEOUT_NS' in event
    program = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#define GGML_ABORT(...) do { char msg[512]; std::snprintf(msg, sizeof(msg), __VA_ARGS__); throw std::runtime_error(msg); } while (0)
namespace wgpu {
enum class WaitStatus { Success, TimedOut, Error };
enum class QueueWorkDoneStatus { Success, Error };
enum class MapAsyncStatus { Success, Error };
enum class CallbackMode { AllowSpontaneous };
enum class MapMode { Read };
using StringView = std::string;
inline uint64_t elapsed = 0, observed = 0;
inline WaitStatus result = WaitStatus::Success;
inline bool callbackError = false;
struct Queue {
 template<class F> auto OnSubmittedWorkDone(CallbackMode, F callback) {
   return [callback] { callback(callbackError ? QueueWorkDoneStatus::Error : QueueWorkDoneStatus::Success, "fixture queue"); };
 }
};
struct Buffer {
 template<class F> auto MapAsync(MapMode, size_t, size_t, CallbackMode, F callback) {
   return [callback] { callback(callbackError ? MapAsyncStatus::Error : MapAsyncStatus::Success, "fixture map"); };
 }
};
struct Instance {
 template<class F> WaitStatus WaitAny(F callback, uint64_t timeout) {
   observed = timeout;
   if (result == WaitStatus::Error) return result;
   if (elapsed >= timeout) return WaitStatus::TimedOut;
   callback(); return WaitStatus::Success;
 }
};
}
struct Context { wgpu::Instance instance; wgpu::Queue queue; };
using webgpu_global_context = std::shared_ptr<Context>;
'''
    program += '\n' + constants + '\n' + checker + '\n' + queue + '\n' + mapping
    program += r'''
int main() {
 auto context = std::make_shared<Context>(); wgpu::Buffer buffer;
 int checked = 0;
 for (bool isMap : {false, true}) {
   auto call = [&] { if (isMap) ggml_backend_webgpu_map_buffer(context, buffer, wgpu::MapMode::Read, 0, 4);
                    else ggml_backend_webgpu_wait_queue(context); };
   const uint64_t budget = isMap ? 30000 : 180000;
   for (uint64_t milliseconds : {uint64_t(0), uint64_t(35000), uint64_t(179000), uint64_t(181000)}) {
     wgpu::elapsed = milliseconds * 1000000ull; wgpu::result = wgpu::WaitStatus::Success; wgpu::callbackError = false;
     bool rejected = false;
     try { call(); } catch (const std::runtime_error & error) {
       rejected = true; assert(std::string(error.what()).find("timed out after " + std::to_string(budget) + " ms") != std::string::npos);
     }
     assert(rejected == (milliseconds >= budget)); assert(wgpu::observed == budget * 1000000ull); checked++;
   }
   wgpu::elapsed = 0; wgpu::result = wgpu::WaitStatus::Error;
   bool rejected = false; try { call(); } catch (const std::runtime_error & error) {
     rejected = true; assert(std::string(error.what()).find("wait failed") != std::string::npos);
   } assert(rejected); checked++;
   wgpu::result = wgpu::WaitStatus::Success; wgpu::callbackError = true; rejected = false;
   try { call(); } catch (const std::runtime_error & error) {
     rejected = true; assert(std::string(error.what()).find("failed with status") != std::string::npos);
   } assert(rejected); checked++;
 }
 std::cout << checked << " prepared-source wait cases passed; no GPU was executed\n";
}
'''
    with tempfile.TemporaryDirectory(prefix='sdb-wait-') as directory:
        root = Path(directory); cpp = root / 'probe.cpp'; binary = root / 'probe'
        cpp.write_text(program)
        subprocess.run([args.cxx, '-std=c++17', '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
