// sim_core.cpp — C++ simulation core: Scheduler + Engine loop.
//
// Mirrors the Python simulator logic (scheduler.py + engine.py).
// Uses a pool-based design: Request objects are stored in a flat vector,
// and the scheduler tracks pool indices in its running/waiting queues.

#include "sim_core.h"

#include <algorithm>
#include <cassert>

namespace sim {

// -----------------------------------------------------------------------
// PerfPredictor
// -----------------------------------------------------------------------

double PerfPredictor::predict(int B, int C, int A) {
    // Quantize (matches Python predict_cached logic)
    int qB = std::max(1, B);
    int qC = std::max(1, ((C + 16) / 32) * 32);
    int qA = ((A + 512) / 1024) * 1024;

    uint64_t key = pack_key(qB, qC, qA);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }
    double val = py_predict_(qB, qC, qA);
    cache_[key] = val;
    return val;
}

// -----------------------------------------------------------------------
// Scheduler
// -----------------------------------------------------------------------

void Scheduler::add_request(int pool_idx) {
    waiting_.push_back(pool_idx);
}

bool Scheduler::has_work() const {
    return !running_.empty() || !waiting_.empty();
}

void Scheduler::abort_all() {
    running_.clear();
    waiting_.clear();
    kv_used_ = 0;
}

int Scheduler::tokens_for_running(const Request& req, int budget,
                                  bool /*chunked_prefill*/) {
    if (req.is_prefilling()) {
        return std::min(req.remaining_prefill(), budget);
    }
    return std::min(1, budget);
}

int Scheduler::tokens_for_new(const Request& req, int budget,
                              bool chunked_prefill) {
    if (req.is_prefilling()) {
        int remaining = req.remaining_prefill();
        if (!chunked_prefill && remaining > budget) {
            return 0;
        }
        return std::min(remaining, budget);
    }
    return std::min(1, budget);
}

bool Scheduler::schedule(StepPlan& plan, std::vector<Request>& pool) {
    plan.clear();

    if (running_.empty() && waiting_.empty()) {
        return false;
    }

    int token_budget = config_.max_num_batched_tokens;
    plan.is_decode_only = true;

    // --- Phase 1: plan tokens for every running request ---
    for (int ri = 0; ri < static_cast<int>(running_.size()); ++ri) {
        int pidx = running_[ri];
        Request& req = pool[pidx];
        if (req.num_remaining_tokens() <= 0) continue;

        int new_tokens = tokens_for_running(req, token_budget,
                                            config_.enable_chunked_prefill);
        if (new_tokens <= 0) break;

        token_budget -= new_tokens;

        ScheduledEntry se;
        se.pool_idx = pidx;
        se.num_new_tokens = new_tokens;
        se.access_tokens = req.num_computed_tokens;
        plan.scheduled.push_back(se);

        if (new_tokens != 1 || req.is_prefilling()) {
            plan.is_decode_only = false;
        }
    }

    // --- Phase 1b: preempt from the tail if KV overflows ---
    int kv_needed = 0;
    for (const auto& se : plan.scheduled) {
        kv_needed += se.num_new_tokens;
    }

    while (!plan.scheduled.empty() &&
           kv_used_ + kv_needed > config_.max_kv_tokens) {
        auto& victim_se = plan.scheduled.back();
        int victim_pidx = victim_se.pool_idx;
        token_budget += victim_se.num_new_tokens;
        kv_needed -= victim_se.num_new_tokens;
        plan.scheduled.pop_back();

        Request& victim = pool[victim_pidx];
        kv_used_ -= victim.kv_cache_tokens();
        victim.status = RequestStatus::PREEMPTED;
        victim.num_computed_tokens = 0;
        victim.preemption_count++;
        plan.preempted_pool_ids.push_back(victim_pidx);
        waiting_.push_front(victim_pidx);

        plan.is_decode_only = false;
    }

    // Commit KV for surviving planned requests.
    kv_used_ += kv_needed;

    // Remove preempted from running_.
    if (!plan.preempted_pool_ids.empty()) {
        // Build a set of preempted pool indices for O(1) lookup.
        // (Typically very small, but let's be correct.)
        std::vector<int> new_running;
        new_running.reserve(running_.size());
        for (int pidx : running_) {
            bool preempted = false;
            for (int pp : plan.preempted_pool_ids) {
                if (pidx == pp) { preempted = true; break; }
            }
            if (!preempted) {
                new_running.push_back(pidx);
            }
        }
        running_ = std::move(new_running);
    }

    // --- Phase 2: admit WAITING requests (only if no preemptions) ---
    if (plan.preempted_pool_ids.empty()) {
        while (!waiting_.empty() && token_budget > 0) {
            if (static_cast<int>(running_.size()) >= config_.max_num_seqs) {
                break;
            }

            int pidx = waiting_.front();
            Request& req = pool[pidx];
            int new_tokens = tokens_for_new(req, token_budget,
                                            config_.enable_chunked_prefill);
            if (new_tokens <= 0) break;
            if (kv_used_ + new_tokens > config_.max_kv_tokens) break;

            // Admit.
            waiting_.pop_front();
            req.status = RequestStatus::RUNNING;
            kv_used_ += new_tokens;
            token_budget -= new_tokens;

            running_.push_back(pidx);

            ScheduledEntry se;
            se.pool_idx = pidx;
            se.num_new_tokens = new_tokens;
            se.access_tokens = req.num_computed_tokens;
            plan.scheduled.push_back(se);

            if (new_tokens != 1 || req.is_prefilling()) {
                plan.is_decode_only = false;
            }
        }
    }

    if (plan.scheduled.empty()) {
        return false;
    }

    // Compute aggregate BCA.
    plan.batch_size = static_cast<int>(plan.scheduled.size());
    plan.compute_tokens = 0;
    plan.access_tokens = 0;
    for (const auto& se : plan.scheduled) {
        plan.compute_tokens += se.num_new_tokens;
        plan.access_tokens += se.access_tokens;
    }

    return true;
}

void Scheduler::advance_after_step(
        const StepPlan& plan, double step_end,
        std::vector<Request>& pool,
        std::vector<int>& finished_out) {
    finished_out.clear();

    for (const auto& se : plan.scheduled) {
        Request& req = pool[se.pool_idx];
        req.num_computed_tokens += se.num_new_tokens;
        if (req.num_computed_tokens >= req.num_total_tokens) {
            req.status = RequestStatus::FINISHED;
            req.finish_time = step_end;
            finished_out.push_back(se.pool_idx);
        }
    }

    // Remove finished from running_ and reclaim KV.
    if (!finished_out.empty()) {
        // For small finished_out, linear scan is fine.
        std::vector<int> new_running;
        new_running.reserve(running_.size() - finished_out.size());
        for (int pidx : running_) {
            bool is_finished = false;
            for (int fp : finished_out) {
                if (pidx == fp) { is_finished = true; break; }
            }
            if (is_finished) {
                kv_used_ -= pool[pidx].kv_cache_tokens();
            } else {
                new_running.push_back(pidx);
            }
        }
        running_ = std::move(new_running);
    }
}

// -----------------------------------------------------------------------
// Engine: main simulation loop
// -----------------------------------------------------------------------

SimResult run_simulation(
        const SimConfig& config,
        std::vector<Request> pool,
        PredictFn predict_fn) {

    int total = static_cast<int>(pool.size());

    // Sort by arrival time (stable sort to preserve order for ties).
    std::vector<int> arrival_order(total);
    for (int i = 0; i < total; ++i) arrival_order[i] = i;
    std::stable_sort(arrival_order.begin(), arrival_order.end(),
                     [&pool](int a, int b) {
                         return pool[a].arrival_time < pool[b].arrival_time;
                     });

    // Precompute num_total_tokens.
    for (auto& r : pool) {
        r.num_total_tokens = r.prompt_tokens + std::max(r.output_tokens - 1, 0);
        r.status = RequestStatus::WAITING;
        r.num_computed_tokens = 0;
        r.first_token_time = -1.0;
        r.finish_time = -1.0;
        r.preemption_count = 0;
    }

    Scheduler scheduler(config);
    PerfPredictor predictor(std::move(predict_fn));

    double clock = 0.0;
    int step_index = 0;
    int pending_pos = 0;  // position in arrival_order

    StepPlan plan;
    std::vector<int> finished_pool_ids;

    // Reserve to avoid repeated allocations.
    plan.scheduled.reserve(config.max_num_seqs + 16);
    finished_pool_ids.reserve(64);

    while (pending_pos < total || scheduler.has_work()) {
        // Phase 1: inject all requests that have arrived.
        while (pending_pos < total) {
            int pidx = arrival_order[pending_pos];
            if (pool[pidx].arrival_time > clock) break;
            scheduler.add_request(pidx);
            pending_pos++;
        }

        // Phase 2: schedule.
        bool ok = scheduler.schedule(plan, pool);

        if (!ok) {
            if (scheduler.has_work()) {
                scheduler.abort_all();
            }
            if (pending_pos < total) {
                int next_pidx = arrival_order[pending_pos];
                clock = pool[next_pidx].arrival_time;
                continue;
            } else {
                break;
            }
        }

        // Phase 3: predict step time.
        double step_time_ms = predictor.predict(
            plan.batch_size, plan.compute_tokens, plan.access_tokens);
        double step_time_s = step_time_ms / 1000.0;
        double step_end = clock + step_time_s;

        // Phase 4: detect first-token events (skip for decode-only batches).
        if (!plan.is_decode_only) {
            for (const auto& se : plan.scheduled) {
                Request& req = pool[se.pool_idx];
                if (req.first_token_time < 0.0 &&
                    req.num_computed_tokens + se.num_new_tokens
                        >= req.prompt_tokens) {
                    req.first_token_time = step_end;
                }
            }
        }

        // Phase 5: advance state, detect finished, set finish_time.
        scheduler.advance_after_step(plan, step_end, pool, finished_pool_ids);

        // Phase 6: advance clock.
        clock = step_end;
        step_index++;
    }

    // Collect results from the pool.
    SimResult result;
    result.total_steps = step_index;
    result.finished.reserve(total);
    for (const auto& req : pool) {
        if (req.status == RequestStatus::FINISHED && req.finish_time >= 0.0) {
            RequestResult rr;
            rr.request_id = req.request_id;
            rr.arrival_time = req.arrival_time;
            rr.first_token_time = req.first_token_time;
            rr.finish_time = req.finish_time;
            rr.prompt_tokens = req.prompt_tokens;
            rr.output_tokens = req.output_tokens;
            rr.preemption_count = req.preemption_count;
            result.finished.push_back(std::move(rr));
        }
    }

    return result;
}

}  // namespace sim
