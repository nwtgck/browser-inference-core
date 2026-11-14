// Actual pinned audio graph and preprocessing APIs on a CPU scheduler. The
// numerical reference is a direct double DFT, not the retired FFT's output.
// Synthetic parameters do not establish model/audio quality or GPU behavior.
#include "mtmd-audio.h"
#include "mtmd-helper.h"
#include "mtmd-subcontext.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static size_t checks = 0;
static double max_error = 0;
static void require(bool value, const char * message) {
    ++checks;
    if (!value)
        throw std::runtime_error(message);
}
struct scheduler {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_sched_t sched = nullptr;
    scheduler() {
        require(cpu != nullptr, "CPU backend initialization failed");
        ggml_backend_cpu_set_n_threads(cpu, 1);
        sched = ggml_backend_sched_new(&cpu, nullptr, 1, 2048, false, true);
        require(sched != nullptr, "CPU scheduler initialization failed");
    }
    ~scheduler() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
    }
};
static void numeric(scheduler & runtime, int kind, int frames, bool natural, bool magnitude, bool add_floor) {
    constexpr int fft_size = 16, bins = 9, mels = 4, hop = 3;
    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(fft_size);
    cache.fill_hann_window(fft_size, true);
    cache.filters.n_mel = mels;
    cache.filters.n_fft = fft_size;
    cache.filters.data.resize(mels * bins);
    for (int mel = 0; mel < mels; ++mel)
        for (int k = 0; k < bins; ++k)
            cache.filters.data[mel * bins + k] = ((mel + k) % 3 + 1) / 32.0f;
    std::vector<float> samples((frames - 1) * hop + fft_size);
    for (size_t i = 0; i < samples.size(); ++i) {
        if (kind == 1)
            samples[i] = 1e-14f * std::sin(float(i) * 0.23f);
        if (kind == 2)
            samples[i] = i == 8 ? 1.0f : 0.0f;
        if (kind == 3)
            samples[i] = 0.2f * std::sin(float(i) * 0.31f) + 0.1f * std::cos(float(i) * 0.13f);
    }
    mtmd_audio_mel_config config;
    config.n_mel = mels;
    config.n_fft_bins = bins;
    config.hop_length = hop;
    config.n_threads = 1;
    config.use_natural_log = natural;
    config.use_magnitude = magnitude;
    config.mel_floor = 1e-6f;
    config.mel_floor_add = add_floor;
    config.cache = &cache;
    config.sched = runtime.sched;
    std::vector<float> output(mels * frames, -12345);
    require(mtmd_audio_log_mel(config, samples.data(), cache.hann_window.data(), frames, output.data()),
            "actual audio graph failed");
    const double pi = std::acos(-1.0);
    for (int frame = 0; frame < frames; ++frame) {
        double spectrum[bins] = {};
        for (int k = 0; k < bins; ++k) {
            double re = 0, im = 0;
            for (int j = 0; j < fft_size; ++j) {
                double input = double(samples[frame * hop + j]) * cache.hann_window[j];
                re += input * std::cos(2 * pi * j * k / fft_size);
                im += input * std::sin(2 * pi * j * k / fft_size);
            }
            spectrum[k] = magnitude ? std::hypot(re, im) : re * re + im * im;
        }
        for (int mel = 0; mel < mels; ++mel) {
            double sum = 0;
            for (int k = 0; k < bins; ++k)
                sum += spectrum[k] * cache.filters.data[mel * bins + k];
            sum = add_floor ? sum + config.mel_floor : std::max(sum, double(config.mel_floor));
            double expected = natural ? std::log(sum) : std::log10(sum);
            double actual = output[mel * frames + frame];
            double error = std::abs(expected - actual);
            max_error = std::max(max_error, error);
            require(std::isfinite(actual) && error <= 2e-4 + std::abs(expected) * 1e-5,
                    "DFT graph exceeds the numerical reference tolerance");
        }
    }
    float sentinel = 31;
    require(mtmd_audio_log_mel(config, samples.data(), cache.hann_window.data(), 0, &sentinel),
            "zero-frame call failed");
    require(sentinel == 31, "zero-frame call modified output");
}

static void check_processor(mtmd_audio_preprocessor & processor, scheduler & runtime) {
    processor.initialize();
    processor.mel_config.sched = runtime.sched;
    processor.mel_config.n_threads = 1;
    auto empty = processor.preprocess(nullptr, 0);
    require(empty.entries.empty() && empty.samples.empty(), "empty audio was accepted");
    std::vector<float> samples(256);
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i] = 0.2f * std::sin(float(i) * 0.13f);
    auto saved = samples;
    auto first = processor.preprocess(samples.data(), samples.size());
    require(first.needs_compute() && first.entries.size() == 1, "audio was not deferred");
    require(first.samples == saved, "deferred input is not owned");
    auto copied = first;
    for (float & value : samples)
        value += 0.3f;
    auto second = processor.preprocess(samples.data(), samples.size());
    std::fill(samples.begin(), samples.end(), 99);
    samples.clear();
    samples.shrink_to_fit();
    require(first.samples == saved && copied.samples == saved, "caller mutation changed owned PCM");
    require(processor.compute(second), "reverse-order second compute failed");
    require(processor.compute(first), "reverse-order first compute failed");
    require(first.samples.empty() && !first.needs_compute(), "successful compute retained PCM");
    require(!first.entries[0].is_placeholder(), "computed output is a placeholder");
    const auto result = first.entries[0].get_ro_buf();
    require(!result.empty(), "computed output is empty");
    for (float value : result)
        require(std::isfinite(value), "computed audio contains a nonfinite value");
    require(processor.compute(copied), "copied deferred output failed");
    require(copied.entries[0].get_ro_buf() == result, "copied PCM computed differently");
    require(processor.compute(first), "cached compute failed");
    require(first.entries[0].get_ro_buf() == result && first.samples.empty(), "cached compute changed output");
    require(second.entries[0].get_ro_buf() != result, "separate deferred inputs were conflated");
}
struct failing_processor : mtmd_audio_preprocessor {
    mtmd_audio_cache cache;
    mutable bool fail = true;
    failing_processor(const mtmd_hparams & params) : mtmd_audio_preprocessor(params, nullptr) {}
    void initialize() override {
        mel_config.cache = &cache;
    }
    bool process(const float *, size_t, std::vector<mtmd_audio_mel> & output, bool shape_only) const override {
        if (!shape_only && fail)
            return false;
        mtmd_audio_mel mel{};
        mel.n_len = mel.n_len_org = 2;
        mel.n_mel = 2;
        if (!shape_only)
            mel.data = {1, 2, 3, 4};
        output.push_back(std::move(mel));
        return true;
    }
};
static void resamplers() {
    require(mtmd_helper_resampler_init(0, 16000) == nullptr, "zero sample rate accepted");
    require(mtmd_helper_resampler_init(16000, -1) == nullptr, "negative sample rate accepted");
    mtmd_helper_resampler_free(nullptr);
    float input[] = {1, 2, 3, 4};
    const float * output = nullptr;
    auto * passthrough = mtmd_helper_resampler_init(16000, 16000);
    require(passthrough != nullptr, "passthrough initialization failed");
    require(mtmd_helper_resampler_process(passthrough, input, 4, &output) == 4 && output == input,
            "passthrough does not alias caller input");
    input[1] = 9;
    require(output[1] == 9, "passthrough output lost caller ownership");
    mtmd_helper_resampler_free(passthrough);
    require(output == input && output[1] == 9, "passthrough free altered caller memory");
    auto * converted = mtmd_helper_resampler_init(48000, 16000);
    require(converted != nullptr, "converting resampler initialization failed");
    std::vector<float> samples(960);
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i] = 0.1f * std::sin(float(i) * 0.03f);
    const int64_t count = mtmd_helper_resampler_process(converted, samples.data(), samples.size(), &output);
    require(count > 0 && count <= 960 && output && output != samples.data(),
            "converted output ownership/count invalid");
    std::vector<float> snapshot(output, output + count);
    std::fill(samples.begin(), samples.end(), 99);
    require(std::equal(snapshot.begin(), snapshot.end(), output), "converted output aliases caller samples");
    for (float value : snapshot)
        require(std::isfinite(value), "converted output is nonfinite");
    // Do not dereference the previous context-owned pointer after the next call.
    require(mtmd_helper_resampler_process(converted, nullptr, 0, &output) == 0, "empty resampler call failed");
    mtmd_helper_resampler_free(converted);
}
int main() {
    scheduler runtime;
    resamplers();
    for (int kind = 0; kind < 4; ++kind)
        for (bool natural : {false, true})
            for (bool magnitude : {false, true})
                for (bool add_floor : {false, true})
                    numeric(runtime, kind, 7, natural, magnitude, add_floor);
    numeric(runtime, 3, 8193, true, false, false); // crosses the actual 8192-frame boundary

    mtmd_model qwen_model{};
    auto & q = qwen_model.hparams;
    q.audio_n_fft = q.audio_window_len = 16;
    q.audio_hop_len = 4;
    q.audio_sample_rate = 16000;
    q.n_mel_bins = 4;
    mtmd_audio_preprocessor_qwen3tts_spk qwen(&qwen_model);
    check_processor(qwen, runtime);
    float short_sample = 0;
    require(qwen.preprocess(&short_sample, 1).entries.empty(), "too-short reference was accepted");

    mtmd_model parakeet_model{};
    auto & p = parakeet_model.hparams;
    p.audio_n_fft = 16;
    p.audio_window_len = 12;
    p.audio_hop_len = 4;
    p.audio_sample_rate = 16000;
    p.n_mel_bins = 4;
    mtmd_audio_cache bank;
    bank.fill_mel_filterbank_matrix(4, 16, 16000);
    bank.fill_hann_window(12, true);
    p.mel_filters = bank.filters.data;
    p.window = bank.hann_window;
    mtmd_audio_preprocessor_parakeet parakeet(&parakeet_model);
    check_processor(parakeet, runtime);

    failing_processor failure(p);
    failure.initialize();
    float input[] = {5, 6};
    auto out = failure.preprocess(input, 2);
    require(!failure.compute(out), "failed preprocessing was reported as success");
    require(out.needs_compute() && out.samples == std::vector<float>({5, 6}), "failure discarded retry input");
    failure.fail = false;
    require(failure.compute(out) && !out.needs_compute() && out.samples.empty(), "retry did not finish");
    std::printf("{\"checks\":%zu,\"numericCases\":33,\"framesPerLongCase\":8193,\"maxAbsError\":%.9g}\n", checks,
                max_error);
    return 0;
}
