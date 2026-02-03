// sim_core.cpp — C++ simulation core: Scheduler + Engine loop.
//
// Mirrors the Python simulator logic (scheduler.py + engine.py).
// Uses a pool-based design: Request objects are stored in a flat vector,
// and the scheduler tracks pool indices in its running/waiting queues.

#include "sim_core.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace sim {

// -----------------------------------------------------------------------
// TreeEnsemble
// -----------------------------------------------------------------------

double TreeEnsemble::predict(int B, int C, int A) const {
    double features[9];
    double bf = static_cast<double>(B);
    double cf = static_cast<double>(C);
    double af = static_cast<double>(A);

    if (d_.use_extended_features) {
        double safe_b = std::max(1.0, bf);
        features[0] = bf;                         // batch_size
        features[1] = cf;                         // compute_tokens
        features[2] = af;                         // access_tokens
        features[3] = cf / safe_b;                // per_req_compute
        features[4] = af / safe_b;                // per_req_access
        features[5] = (B == 1) ? 1.0 : 0.0;      // is_sync
        features[6] = std::log1p(bf);             // log1p_batch_size
        features[7] = std::log1p(cf);             // log1p_compute_tokens
        features[8] = std::log1p(af);             // log1p_access_tokens
    } else {
        features[0] = bf;
        features[1] = cf;
        features[2] = af;
    }

    double sum = d_.init_value;
    for (int t = 0; t < d_.n_trees; ++t) {
        sum += d_.learning_rate * traverse_tree(t, features);
    }

    if (d_.use_log_target) return std::exp(sum);
    return std::max(0.0, sum);
}

double TreeEnsemble::traverse_tree(int tree_idx,
                                   const double* features) const {
    int base = d_.tree_offsets[tree_idx];
    int node = 0;  // root is always node 0 within each tree
    // sklearn: feature == -2 (TREE_UNDEFINED) indicates a leaf node
    while (d_.feature[base + node] != -2) {
        if (features[d_.feature[base + node]] <= d_.threshold[base + node])
            node = d_.children_left[base + node];
        else
            node = d_.children_right[base + node];
    }
    return d_.value[base + node];
}

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
    double val = raw_predict(qB, qC, qA);
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
// Engine: main simulation loop (shared implementation)
// -----------------------------------------------------------------------

static SimResult run_simulation_impl(
        const SimConfig& config,
        std::vector<Request> pool,
        PerfPredictor& predictor) {

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

// -----------------------------------------------------------------------
// Public API: Python callback mode
// -----------------------------------------------------------------------

SimResult run_simulation(
        const SimConfig& config,
        std::vector<Request> pool,
        PredictFn predict_fn) {
    PerfPredictor predictor(std::move(predict_fn));
    return run_simulation_impl(config, std::move(pool), predictor);
}

// -----------------------------------------------------------------------
// Public API: Native tree ensemble mode
// -----------------------------------------------------------------------

SimResult run_simulation_native(
        const SimConfig& config,
        std::vector<Request> pool,
        TreeEnsembleData tree_data) {
    PerfPredictor predictor(std::move(tree_data));
    return run_simulation_impl(config, std::move(pool), predictor);
}

// -----------------------------------------------------------------------
// Cluster simulation
// -----------------------------------------------------------------------

using EventQueue = std::priority_queue<Event, std::vector<Event>,
                                       std::greater<Event>>;

// Dispatch: choose target instance for a new request.
static int cluster_dispatch(DispatchStrategy strategy,
                            const std::vector<std::unique_ptr<InstanceState>>& instances,
                            int& rr_counter) {
    int n = static_cast<int>(instances.size());
    switch (strategy) {
        case DispatchStrategy::ROUND_ROBIN:
            return (rr_counter++) % n;

        case DispatchStrategy::LEAST_LOADED: {
            int best = 0;
            int best_load = std::numeric_limits<int>::max();
            for (int i = 0; i < n; ++i) {
                auto& inst = *instances[i];
                int load = inst.scheduler.num_running()
                         + inst.scheduler.num_waiting()
                         + static_cast<int>(inst.pending.size());
                if (load < best_load) {
                    best_load = load;
                    best = i;
                }
            }
            return best;
        }
    }
    return 0;  // fallback
}

// Try to start a step on the given instance.
// May push a STEP_COMPLETE event into the event queue.
// Reusable plan and finished_out are passed to avoid allocation.
static void cluster_try_start_step(
        InstanceState& inst,
        std::vector<Request>& pool,
        std::vector<PerfPredictor>& predictors,
        EventQueue& events,
        StepPlan& plan,
        std::vector<int>& finished_out) {

    for (;;) {
        // Inject pending requests whose arrival_time <= inst.clock.
        while (!inst.pending.empty()) {
            int pidx = inst.pending.front();
            if (pool[pidx].arrival_time > inst.clock) break;
            inst.scheduler.add_request(pidx);
            inst.pending.pop_front();
        }

        // Schedule.
        bool ok = inst.scheduler.schedule(plan, pool);

        if (!ok) {
            if (inst.scheduler.has_work()) {
                inst.scheduler.abort_all();
            }
            if (!inst.pending.empty()) {
                // Fast-forward clock to next pending request arrival.
                inst.clock = pool[inst.pending.front()].arrival_time;
                continue;  // Retry (loop instead of recursion).
            }
            return;  // Instance goes idle.
        }

        // Predict step time.
        PerfPredictor& predictor = predictors[inst.group_idx];
        double step_time_ms = predictor.predict(
            plan.batch_size, plan.compute_tokens, plan.access_tokens);
        double step_end = inst.clock + step_time_ms / 1000.0;

        // Detect first-token events.
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

        // Advance state.
        inst.scheduler.advance_after_step(plan, step_end, pool, finished_out);

        // Advance clock.
        inst.clock = step_end;
        inst.total_steps++;

        // Push STEP_COMPLETE event.
        Event ev;
        ev.time = step_end;
        ev.type = Event::STEP_COMPLETE;
        ev.data = inst.instance_idx;
        events.push(ev);
        inst.step_scheduled = true;
        break;  // Step started; exit the loop.
    }
}

static ClusterSimResult run_cluster_impl(
        const ClusterConfig& config,
        std::vector<Request> pool,
        std::vector<PerfPredictor>& predictors) {

    int total = static_cast<int>(pool.size());

    // Precompute num_total_tokens and reset state.
    for (auto& r : pool) {
        r.num_total_tokens = r.prompt_tokens + std::max(r.output_tokens - 1, 0);
        r.status = RequestStatus::WAITING;
        r.num_computed_tokens = 0;
        r.first_token_time = -1.0;
        r.finish_time = -1.0;
        r.preemption_count = 0;
    }

    // Sort by arrival time.
    std::vector<int> arrival_order(total);
    for (int i = 0; i < total; ++i) arrival_order[i] = i;
    std::stable_sort(arrival_order.begin(), arrival_order.end(),
                     [&pool](int a, int b) {
                         return pool[a].arrival_time < pool[b].arrival_time;
                     });

    // Expand instance groups into flat instance list.
    std::vector<std::unique_ptr<InstanceState>> instances;
    for (int g = 0; g < static_cast<int>(config.groups.size()); ++g) {
        for (int i = 0; i < config.groups[g].count; ++i) {
            instances.push_back(std::make_unique<InstanceState>(
                static_cast<int>(instances.size()), g,
                config.groups[g].sim_config));
        }
    }
    int n_instances = static_cast<int>(instances.size());

    // Build event queue with all request arrivals.
    EventQueue events;
    for (int pidx : arrival_order) {
        Event ev;
        ev.time = pool[pidx].arrival_time;
        ev.type = Event::REQUEST_ARRIVAL;
        ev.data = pidx;
        events.push(ev);
    }

    // Dispatch state.
    int rr_counter = 0;

    // Reusable buffers (avoid per-step allocation).
    StepPlan plan;
    std::vector<int> finished_out;
    plan.scheduled.reserve(256);
    finished_out.reserve(64);

    // Track which instance each request is assigned to.
    std::vector<int> request_instance(total, -1);

    // Per-instance dispatch counts (number of requests dispatched).
    std::vector<int> dispatch_counts(n_instances, 0);

    // Temporary buffers for batching same-timestamp arrivals.
    std::vector<int> arrival_batch;
    std::vector<int> affected;
    arrival_batch.reserve(64);
    affected.reserve(n_instances);

    // Main event loop.
    while (!events.empty()) {
        Event ev = events.top();
        events.pop();

        if (ev.type == Event::REQUEST_ARRIVAL) {
            // Collect all ARRIVAL events at the same timestamp.
            arrival_batch.clear();
            arrival_batch.push_back(ev.data);
            while (!events.empty()
                   && events.top().time == ev.time
                   && events.top().type == Event::REQUEST_ARRIVAL) {
                arrival_batch.push_back(events.top().data);
                events.pop();
            }
            // Sort by pool_idx for deterministic dispatch order.
            std::sort(arrival_batch.begin(), arrival_batch.end());

            // Dispatch all same-timestamp arrivals before starting steps.
            affected.clear();
            for (int pool_idx : arrival_batch) {
                int target = cluster_dispatch(config.dispatch_strategy,
                                              instances, rr_counter);
                instances[target]->pending.push_back(pool_idx);
                request_instance[pool_idx] = target;
                dispatch_counts[target]++;

                if (!instances[target]->step_scheduled) {
                    // Add to affected set (check for duplicates).
                    if (std::find(affected.begin(), affected.end(), target)
                            == affected.end()) {
                        affected.push_back(target);
                    }
                }
            }

            // Trigger steps on affected idle instances (sorted for determinism).
            std::sort(affected.begin(), affected.end());
            for (int idx : affected) {
                cluster_try_start_step(*instances[idx], pool, predictors,
                                       events, plan, finished_out);
            }
        } else {
            // STEP_COMPLETE
            auto& inst = *instances[ev.data];
            inst.step_scheduled = false;
            cluster_try_start_step(inst, pool, predictors,
                                   events, plan, finished_out);
        }
    }

    // Collect results per instance.
    ClusterSimResult result;
    result.total_instances = n_instances;
    result.dispatch_counts = std::move(dispatch_counts);
    result.instance_results.resize(n_instances);
    for (int i = 0; i < n_instances; ++i) {
        result.instance_results[i].instance_idx = i;
        result.instance_results[i].group_idx = instances[i]->group_idx;
        result.instance_results[i].total_steps = instances[i]->total_steps;
    }

    result.all_finished.reserve(total);
    for (int pidx = 0; pidx < total; ++pidx) {
        const auto& req = pool[pidx];
        if (req.status == RequestStatus::FINISHED && req.finish_time >= 0.0) {
            RequestResult rr;
            rr.request_id = req.request_id;
            rr.arrival_time = req.arrival_time;
            rr.first_token_time = req.first_token_time;
            rr.finish_time = req.finish_time;
            rr.prompt_tokens = req.prompt_tokens;
            rr.output_tokens = req.output_tokens;
            rr.preemption_count = req.preemption_count;

            int inst_idx = request_instance[pidx];
            if (inst_idx >= 0) {
                result.instance_results[inst_idx].finished.push_back(rr);
            }
            result.all_finished.push_back(std::move(rr));
        }
    }

    return result;
}

// -----------------------------------------------------------------------
// Public API: Cluster simulation with native tree ensemble
// -----------------------------------------------------------------------

ClusterSimResult run_cluster_simulation_native(
        const ClusterConfig& config,
        std::vector<Request> pool,
        std::vector<TreeEnsembleData> tree_data_per_group) {

    // Build one PerfPredictor per group.
    std::vector<PerfPredictor> predictors;
    for (auto& td : tree_data_per_group) {
        predictors.emplace_back(std::move(td));
    }
    return run_cluster_impl(config, std::move(pool), predictors);
}

// -----------------------------------------------------------------------
// Public API: Cluster simulation with Python callbacks
// -----------------------------------------------------------------------

ClusterSimResult run_cluster_simulation(
        const ClusterConfig& config,
        std::vector<Request> pool,
        std::vector<PredictFn> predict_fns_per_group) {

    std::vector<PerfPredictor> predictors;
    for (auto& fn : predict_fns_per_group) {
        predictors.emplace_back(std::move(fn));
    }
    return run_cluster_impl(config, std::move(pool), predictors);
}

}  // namespace sim
