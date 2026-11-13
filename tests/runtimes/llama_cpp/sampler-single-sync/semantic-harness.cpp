// CPU-only differential test of actual upstream and prepared sampler functions.
// Actual public getters and context synchronization are extracted from pinned source.
// The deferred two-backend scheduler/output store is a model, not a GPU runtime.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using llama_token = int32_t;
static constexpr llama_token LLAMA_TOKEN_NULL = -1;
#define GGML_ASSERT(value)                                                                                             \
    do {                                                                                                               \
        if (!(value))                                                                                                  \
            throw std::logic_error(#value);                                                                            \
    } while (0)
#define LLAMA_LOG_DEBUG(...) ((void)0)

struct llama_token_data {
    llama_token id;
    float logit;
    float p;
};
struct llama_token_data_array {
    llama_token_data * data;
    size_t size;
    int64_t selected;
    bool sorted;
};
struct llama_sampler_chain {
    std::vector<llama_token_data> cur;
};
static const int llama_sampler_chain_i = 1, other_sampler_i = 2;
struct llama_sampler {
    const int * iface;
    void * ctx;
};
struct llama_vocab {
    int n_tokens() const {
        return 8;
    }
};
struct llama_model {
    llama_vocab vocab;
};
struct fake_scheduler {
    bool pending[2] = {true, true};
    int synchronize_calls = 0, backend_waits[2] = {0, 0};
};
static int64_t clock_us = 1000;
static int64_t ggml_time_us() {
    return ++clock_us;
}
static void ggml_backend_sched_synchronize(fake_scheduler * sched) {
    ++sched->synchronize_calls;
    for (int backend = 0; backend < 2; ++backend) {
        ++sched->backend_waits[backend];
        sched->pending[backend] = false;
        clock_us += 10;
    }
}
enum class output_path { raw, logits, probs, token };
struct row {
    int logical_batch_index;
    std::vector<float> raw, backend_logits, probs;
    std::vector<llama_token> ids;
    llama_token token = LLAMA_TOKEN_NULL;
};
struct llama_context {
    std::unique_ptr<fake_scheduler> sched = std::make_unique<fake_scheduler>();
    struct {
        bool no_perf = false;
    } cparams;
    int64_t n_queued_tokens = 1, t_compute_start_us = 100, t_eval_us = 0, t_p_eval_us = 0;
    int64_t n_eval = 0, n_p_eval = 0, t_load_us = 0, t_start_us = 10;
    bool has_evaluated_once = false;
    llama_model model;
    std::vector<row> rows;
    std::vector<llama_token> full_ids{0, 1, 2, 3, 4, 5, 6, 7};
    bool reordered = false;
    int reorder_operations = 0;
    explicit llama_context(output_path path) {
        for (int logical : {4, 0, 2}) {
            row item;
            item.logical_batch_index = logical;
            item.raw = {-4, -3, -2, -1, 0, 1, 2, 3};
            item.raw[(logical + 1) % 8] = 10;
            if (path != output_path::raw) {
                item.ids = {llama_token((logical + 3) % 8), llama_token((logical + 6) % 8), llama_token(logical)};
                item.backend_logits = {-1, 5, 1};
            }
            if (path == output_path::probs)
                item.probs = {0.8f, 0.15f, 0.05f};
            if (path == output_path::token)
                item.token = (logical + 3) % 8;
            rows.push_back(std::move(item));
        }
    }
    void synchronize();
    void output_reorder() {
        assert(!sched->pending[0] && !sched->pending[1]);
        if (!reordered) {
            std::sort(rows.begin(), rows.end(),
                      [](const row & a, const row & b) { return a.logical_batch_index < b.logical_batch_index; });
            reordered = true;
            ++reorder_operations;
        }
    }
    row & output(int32_t index) {
        output_reorder();
        if (index < 0)
            return rows.at(rows.size() + index);
        auto found = std::find_if(rows.begin(), rows.end(),
                                  [index](const row & item) { return item.logical_batch_index == index; });
        if (found == rows.end())
            throw std::out_of_range("output index");
        return *found;
    }
    llama_token get_sampled_token_ith(int32_t index) {
        return output(index).token;
    }
    float * get_sampled_probs_ith(int32_t index) {
        auto & v = output(index).probs;
        return v.empty() ? nullptr : v.data();
    }
    float * get_sampled_logits_ith(int32_t index) {
        auto & v = output(index).backend_logits;
        return v.empty() ? nullptr : v.data();
    }
    const llama_token * get_sampled_candidates_ith(int32_t index) {
        auto & v = output(index).ids;
        return v.empty() ? full_ids.data() : v.data();
    }
    size_t get_sampled_probs_count(int32_t index) {
        return output(index).probs.size();
    }
    size_t get_sampled_logits_count(int32_t index) {
        return output(index).backend_logits.size();
    }
    float * get_logits_ith(int32_t index) {
        return output(index).raw.data();
    }
};
// @ACTUAL_CONTEXT_SYNCHRONIZE@
// @ACTUAL_PUBLIC_GETTERS@
static const llama_model * llama_get_model(llama_context * context) {
    return &context->model;
}
static const llama_vocab * llama_model_get_vocab(const llama_model * model) {
    return &model->vocab;
}
static int llama_vocab_n_tokens(const llama_vocab * vocab) {
    return vocab->n_tokens();
}
struct observation {
    std::vector<llama_token_data> candidates;
    std::vector<llama_token> accepted;
    int applies = 0;
    bool probs = false, enqueue_in_apply = false, enqueue_in_accept = false;
    llama_context * context = nullptr;
};
static observation * observed = nullptr;
static void enqueue_extra_work() {
    observed->context->n_queued_tokens = 7;
    observed->context->t_compute_start_us = ggml_time_us();
    observed->context->sched->pending[0] = observed->context->sched->pending[1] = true;
}
static void llama_sampler_apply(llama_sampler *, llama_token_data_array * values) {
    assert(!observed->context->sched->pending[0] && !observed->context->sched->pending[1]);
    assert(values->selected == -1 && !values->sorted);
    ++observed->applies;
    observed->candidates = {values->data, values->data + values->size};
    values->selected = 0;
    for (size_t i = 1; i < values->size; ++i) {
        const auto & best = values->data[values->selected];
        if ((observed->probs ? values->data[i].p > best.p : values->data[i].logit > best.logit))
            values->selected = i;
    }
    if (observed->enqueue_in_apply)
        enqueue_extra_work();
}
static void llama_sampler_accept(llama_sampler *, llama_token token) {
    observed->accepted.push_back(token);
    if (observed->enqueue_in_accept)
        enqueue_extra_work();
}
// @ACTUAL_BASELINE_SAMPLE@
// @ACTUAL_PREPARED_SAMPLE@
struct result {
    llama_token token;
    observation observed;
    int synchronizations, waits0, waits1, reorders;
    int64_t n_eval, n_p_eval, queued, t_eval_us, t_load_us;
    bool pending0, pending1;
};
static result exercise(bool prepared, output_path path, int32_t index, bool chain, bool no_perf,
                       bool enqueue_apply = false, bool enqueue_accept = false, bool repeat = false) {
    clock_us = 1000;
    llama_context context(path);
    context.cparams.no_perf = no_perf;
    llama_sampler_chain state;
    state.cur.resize(20, {99, 99, 99});
    llama_sampler sampler{chain ? &llama_sampler_chain_i : &other_sampler_i, &state};
    observation view;
    view.probs = path == output_path::probs;
    view.context = &context;
    view.enqueue_in_apply = enqueue_apply;
    view.enqueue_in_accept = enqueue_accept;
    observed = &view;
    auto fn = prepared ? prepared_sample : baseline_sample;
    const auto token = fn(&sampler, &context, index);
    if (repeat)
        assert(fn(&sampler, &context, index) == token);
    auto * sched = context.sched.get();
    view.context = nullptr;
    return {token,
            view,
            sched->synchronize_calls,
            sched->backend_waits[0],
            sched->backend_waits[1],
            context.reorder_operations,
            context.n_eval,
            context.n_p_eval,
            context.n_queued_tokens,
            context.t_eval_us,
            context.t_load_us,
            sched->pending[0],
            sched->pending[1]};
}
static void equal(const result & a, const result & b) {
    assert(a.token == b.token && a.observed.accepted == b.observed.accepted);
    assert(a.observed.applies == b.observed.applies && a.observed.candidates.size() == b.observed.candidates.size());
    for (size_t i = 0; i < a.observed.candidates.size(); ++i) {
        const auto & x = a.observed.candidates[i];
        const auto & y = b.observed.candidates[i];
        assert(x.id == y.id && x.logit == y.logit && x.p == y.p);
    }
    assert(a.reorders == b.reorders && a.reorders == 1);
    assert(a.n_eval == b.n_eval && a.n_eval == 1 && a.n_p_eval == b.n_p_eval && a.n_p_eval == 0);
    assert(a.queued == b.queued && a.pending0 == b.pending0 && a.pending1 == b.pending1);
    assert(a.t_eval_us == b.t_eval_us && a.t_load_us == b.t_load_us);
}
int main() {
    size_t cases = 0;
    for (auto path : {output_path::raw, output_path::logits, output_path::probs, output_path::token}) {
        for (int32_t index : {0, 2, 4, -1, -2, -3})
            for (bool chain : {false, true})
                for (bool no_perf : {false, true}) {
                    auto a = exercise(false, path, index, chain, no_perf),
                         b = exercise(true, path, index, chain, no_perf);
                    equal(a, b);
                    assert(a.synchronizations == (path == output_path::token ? 4 : 5) && b.synchronizations == 1);
                    assert(b.waits0 == 1 && b.waits1 == 1 && !b.pending0 && !b.pending1 && b.queued == 0);
                    assert(b.observed.accepted.size() == 1 &&
                           b.observed.applies == (path == output_path::token ? 0 : 1));
                    if (path == output_path::raw)
                        assert(b.observed.candidates.size() == 8);
                    if (path == output_path::logits || path == output_path::probs)
                        assert(b.observed.candidates.size() == 3);
                    ++cases;
                }
        auto a = exercise(false, path, -1, true, false, false, false, true);
        auto b = exercise(true, path, -1, true, false, false, false, true);
        equal(a, b);
        assert(b.synchronizations == 2 && b.n_eval == 1 && b.observed.accepted.size() == 2);
        ++cases;
        // Callback side effects can enqueue subsequent work. The sample must not drain it afterward.
        for (bool from_apply : {false, true}) {
            if (path == output_path::token && from_apply)
                continue;
            auto x = exercise(false, path, 2, false, false, from_apply, !from_apply);
            auto y = exercise(true, path, 2, false, false, from_apply, !from_apply);
            equal(x, y);
            assert(y.queued == 7 && y.pending0 && y.pending1 && y.synchronizations == 1);
            ++cases;
        }
    }
    std::cout
        << "PASS: " << cases
        << " source-bound baseline/prepared sampling cases; two-backend completion, metrics, output order, apply/accept\n";
}
