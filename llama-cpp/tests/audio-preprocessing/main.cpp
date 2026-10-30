// Compile against the actual upstream/overlaid translation unit. Only model
// metadata and fatal/log callbacks are fixtures; preprocessing code is upstream.
#include "mtmd-audio.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>

static clip_hparams hparams;
const clip_hparams * clip_get_hparams(const clip_ctx *) {
    return &hparams;
}
clip_logger_state g_logger_state = {clip_log_callback_default, nullptr};
extern "C" void ggml_abort(const char *, int, const char * format, ...) {
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::abort();
}

static void dump(mtmd_audio_preprocessor & processor, const std::vector<float> & samples) {
    processor.initialize();
    std::vector<mtmd_audio_mel> output;
    if (processor.preprocess(samples.data(), 0, output) || !output.empty())
        std::abort();
    if (!processor.preprocess(samples.data(), samples.size(), output) || output.size() != 1)
        std::abort();
    const auto & mel = output.front();
    if (mel.n_len < 2 || mel.n_mel <= 0 || mel.data.size() != size_t(mel.n_len * mel.n_mel))
        std::abort();
    for (float value : mel.data)
        if (!std::isfinite(value))
            std::abort();
    std::fwrite(&mel.n_len, sizeof(mel.n_len), 1, stdout);
    std::fwrite(&mel.n_mel, sizeof(mel.n_mel), 1, stdout);
    std::fwrite(mel.data.data(), sizeof(float), mel.data.size(), stdout);
}

int main() {
    std::vector<float> samples(8192);
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i] = 0.2f * std::sin(float(i) * 0.13f) + 0.1f * std::cos(float(i) * 0.017f);
    hparams.audio_n_fft = 1024;
    hparams.audio_window_len = 1024;
    hparams.audio_hop_len = 256;
    hparams.audio_sample_rate = 24000;
    hparams.n_mel_bins = 128;
    mtmd_audio_preprocessor_qwen3tts_spk qwen(nullptr);
    dump(qwen, samples);
    std::vector<mtmd_audio_mel> too_short;
    if (qwen.preprocess(samples.data(), 1, too_short) || !too_short.empty())
        std::abort();

    hparams.audio_n_fft = 512;
    hparams.audio_window_len = 400;
    hparams.audio_hop_len = 160;
    hparams.audio_sample_rate = 16000;
    hparams.n_mel_bins = 80;
    mtmd_audio_cache cache;
    cache.fill_mel_filterbank_matrix(80, 512, 16000);
    cache.fill_hann_window(400, true);
    hparams.mel_filters = cache.filters.data;
    hparams.window = cache.hann_window;
    mtmd_audio_preprocessor_parakeet parakeet(nullptr);
    dump(parakeet, samples);
    return std::ferror(stdout) ? 1 : 0;
}
