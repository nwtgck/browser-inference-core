"""Compile actual prepared wait functions against deterministic API doubles.

Long completion, failure and callback lifetime cases; not GPU execution. Neither
large logical elapsed values nor the probe's wall time measure inference speed.
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
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    source = args.source.read_text()
    constants = '\n'.join(re.findall(r'^#define WEBGPU_COMPLETION_WAIT_TIMEOUT_NS[^\n]*', source, re.M))
    assert len(constants.splitlines()) == 1 and 'UINT64_MAX' in constants
    assert not re.search(r'WEBGPU_(?:QUEUE|RUNTIME)_WAIT_TIMEOUT_', source)
    assert 'completion-wait-policy-v1 queue=completion map=completion event=completion' in source
    checker = function(source, 'template <typename T>\nstatic void ggml_backend_webgpu_check_wait_status')
    queue = function(source, 'static void ggml_backend_webgpu_wait_queue(')
    mapping = function(source, 'static void ggml_backend_webgpu_map_buffer(')
    event = function(source, 'static void ggml_backend_webgpu_device_event_synchronize(')
    record = function(source, 'static void ggml_backend_webgpu_event_record(')
    event_result = function(source, 'struct ggml_backend_webgpu_event_result {') + ';'
    event_context = function(source, 'struct ggml_backend_webgpu_event_context {') + ';'
    for body in [queue, mapping, event]:
        assert body.count('WaitAny(') == 1
        assert 'WEBGPU_COMPLETION_WAIT_TIMEOUT_NS' in body
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
#define GGML_UNUSED(value) ((void)(value))
namespace wgpu {
enum class WaitStatus { Success, TimedOut, Error };
enum class QueueWorkDoneStatus { Success, Error, CallbackCancelled };
enum class MapAsyncStatus { Success, Error, CallbackCancelled };
enum class CallbackMode { AllowSpontaneous };
enum class MapMode { Read };
using StringView = std::string;
using Future = std::function<void()>;
inline uint64_t elapsed = 0, observed = 0;
inline int waits = 0, requests = 0;
inline WaitStatus waitResult = WaitStatus::Success;
inline int callbackResult = 0;
inline bool fireCallback = true;
struct Queue {
 template<class F> Future OnSubmittedWorkDone(CallbackMode, F callback) {
   requests++;
   return [callback] { callback(static_cast<QueueWorkDoneStatus>(callbackResult), "fixture queue"); };
 }
};
struct Buffer {
 template<class F> Future MapAsync(MapMode, size_t, size_t, CallbackMode, F callback) {
   requests++;
   return [callback] { callback(static_cast<MapAsyncStatus>(callbackResult), "fixture map"); };
 }
};
struct Instance {
 WaitStatus WaitAny(const Future & callback, uint64_t timeout) {
   observed = timeout; waits++;
   if (waitResult != WaitStatus::Success) return waitResult;
   if (timeout != UINT64_MAX && elapsed >= timeout) return WaitStatus::TimedOut;
   if (fireCallback) callback();
   return WaitStatus::Success;
 }
};
}
struct Context { wgpu::Instance instance; wgpu::Queue queue; };
using webgpu_global_context = std::shared_ptr<Context>;
struct WebgpuContext { webgpu_global_context global_ctx; };
struct ggml_backend_webgpu_context { std::shared_ptr<WebgpuContext> webgpu_ctx; };
struct Backend { void *context; };
struct Event { void *context; };
using ggml_backend_t = Backend*;
using ggml_backend_event_t = Event*;
using ggml_backend_dev_t = void*;
'''
    program += '\n' + constants + '\n' + checker + '\n' + queue + '\n' + mapping
    program += '\n' + event_result + '\n' + event_context + '\n' + event + '\n' + record
    program += r'''
int main() {
 auto context = std::make_shared<Context>(); wgpu::Buffer buffer;
 ggml_backend_webgpu_context backendContext{std::make_shared<WebgpuContext>(WebgpuContext{context})};
 Backend backend{&backendContext};
 int checked = 0;
 for (int kind : {0, 1, 2}) {
   ggml_backend_webgpu_event_context eventContext; eventContext.global_ctx = context; Event ev{&eventContext};
   auto reset = [&] {
     wgpu::waits = 0; wgpu::requests = 0; wgpu::observed = 0; wgpu::waitResult = wgpu::WaitStatus::Success;
     wgpu::callbackResult = 0; wgpu::fireCallback = true;
     if (kind == 2) ggml_backend_webgpu_event_record(&backend, &ev);
   };
   auto call = [&] {
     if (kind == 0) ggml_backend_webgpu_wait_queue(context);
     else if (kind == 1) ggml_backend_webgpu_map_buffer(context, buffer, wgpu::MapMode::Read, 0, 4);
     else ggml_backend_webgpu_device_event_synchronize(nullptr, &ev);
   };
   for (uint64_t milliseconds : {uint64_t(0), uint64_t(35000), uint64_t(181000), uint64_t(3600000), uint64_t(86400000)}) {
     reset(); wgpu::elapsed = milliseconds * 1000000ull;
     call(); assert(wgpu::observed == UINT64_MAX); assert(wgpu::waits == 1 && wgpu::requests == 1);
     if (kind == 2) { assert(!eventContext.recorded); assert(!eventContext.result); }
     checked++;
   }
   for (auto status : {wgpu::WaitStatus::Error, wgpu::WaitStatus::TimedOut, static_cast<wgpu::WaitStatus>(123)}) {
     reset(); wgpu::waitResult = status;
     bool rejected = false;
     try { call(); } catch (const std::runtime_error & error) {
       rejected = true; assert(std::string(error.what()).find("failed with wait status") != std::string::npos);
     }
     assert(rejected && wgpu::waits == 1 && wgpu::requests == 1);
     if (kind == 2) assert(eventContext.recorded);
     checked++;
   }
   for (int status : {1, 2, 123}) {
     reset(); wgpu::callbackResult = status; bool rejected = false;
     try { call(); } catch (const std::runtime_error & error) {
       rejected = true; assert(std::string(error.what()).find("failed with status") != std::string::npos);
       assert(std::string(error.what()).find("fixture") != std::string::npos);
     }
     assert(rejected && wgpu::waits == 1 && wgpu::requests == 1); checked++;
   }
   reset(); wgpu::fireCallback = false; bool rejected = false;
   try { call(); } catch (const std::runtime_error &) { rejected = true; }
   assert(rejected); checked++;
 }
 wgpu::fireCallback = true; wgpu::callbackResult = 0; wgpu::waitResult = wgpu::WaitStatus::Success;
 // Unrecorded and already-synchronized events never issue extra queue waits.
 ggml_backend_webgpu_event_context idle; idle.global_ctx = context; Event ev{&idle}; wgpu::waits = 0; wgpu::requests = 0;
 ggml_backend_webgpu_device_event_synchronize(nullptr, &ev); assert(wgpu::waits == 0 && wgpu::requests == 0); checked++;
 ggml_backend_webgpu_event_record(&backend, &ev); ggml_backend_webgpu_device_event_synchronize(nullptr, &ev);
 ggml_backend_webgpu_device_event_synchronize(nullptr, &ev); assert(wgpu::waits == 1 && wgpu::requests == 1); checked++;
 // Re-recording keeps late callbacks isolated from the new future's result.
 ggml_backend_webgpu_event_record(&backend, &ev);
 auto oldFuture = idle.future; std::weak_ptr<ggml_backend_webgpu_event_result> oldResult = idle.result;
 ggml_backend_webgpu_event_record(&backend, &ev); auto newResult = idle.result;
 wgpu::callbackResult = 2; oldFuture();
 assert(newResult->status == wgpu::QueueWorkDoneStatus::Error); assert(oldResult.lock()->status == wgpu::QueueWorkDoneStatus::CallbackCancelled);
 wgpu::callbackResult = 0; ggml_backend_webgpu_device_event_synchronize(nullptr, &ev);
 assert(newResult->status == wgpu::QueueWorkDoneStatus::Success); oldFuture = {}; assert(oldResult.expired()); checked++;
 // A callback after event destruction owns only its independent shared result.
 wgpu::Future late; std::weak_ptr<ggml_backend_webgpu_event_result> lateResult;
 {
   auto temporary = std::make_unique<ggml_backend_webgpu_event_context>(); temporary->global_ctx = context;
   Event temporaryEvent{temporary.get()}; ggml_backend_webgpu_event_record(&backend, &temporaryEvent);
   late = temporary->future; lateResult = temporary->result;
 }
 wgpu::callbackResult = 2; late(); assert(lateResult.lock()->status == wgpu::QueueWorkDoneStatus::CallbackCancelled);
 late = {}; assert(lateResult.expired()); checked++;
 std::cout << checked << " prepared-source completion/failure/lifetime cases passed; no GPU was executed\n";
}
'''
    with tempfile.TemporaryDirectory(prefix='sdb-wait-') as directory:
        root = Path(directory); cpp = root / 'probe.cpp'; binary = root / 'probe'
        cpp.write_text(program)
        flags = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-g'] if args.sanitize else []
        subprocess.run([args.cxx, '-std=c++17', '-Wall', '-Wextra', '-Werror', *flags, str(cpp), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
