// sim_core.h — Data structures and declarations for the C++ simulation core.
//
// Design: All Request objects live in a flat "pool" vector (indexed 0..N-1).
// The Scheduler's running_ and waiting_ queues store pool indices (ints),
// not Request objects. This avoids invalidation when elements are removed.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace sim {

// -----------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------
struct SimConfig {
    int max_num_batched_tokens = 2048;
    int max_num_seqs = 256;
    int max_kv_tokens = 354704;
    bool enable_chunked_prefill = true;
};

// -----------------------------------------------------------------------
// Request (stored in pool, accessed by index)
// -----------------------------------------------------------------------
enum class RequestStatus : uint8_t {
    WAITING,
    RUNNING,
    PREEMPTED,
    FINISHED,
};

struct Request {
    std::string request_id;
    double arrival_time;
    int prompt_tokens;
    int output_tokens;
    int num_total_tokens;  // precomputed: prompt + max(output-1, 0)

    // Mutable state
    RequestStatus status = RequestStatus::WAITING;
    int num_computed_tokens = 0;

    // Timestamps
    double first_token_time = -1.0;  // <0 means unset
    double finish_time = -1.0;
    int preemption_count = 0;

    // Inline helpers
    bool is_prefilling() const { return num_computed_tokens < prompt_tokens; }
    int remaining_prefill() const {
        int r = prompt_tokens - num_computed_tokens;
        return r > 0 ? r : 0;
    }
    int num_remaining_tokens() const {
        return num_total_tokens - num_computed_tokens;
    }
    int kv_cache_tokens() const { return num_computed_tokens; }
};

// -----------------------------------------------------------------------
// Scheduled request (per-step assignment)
// -----------------------------------------------------------------------
struct ScheduledEntry {
    int pool_idx;       // index into the request pool
    int num_new_tokens;
    int access_tokens;  // num_computed_tokens before this step
};

// -----------------------------------------------------------------------
// Step plan (reusable, cleared each step)
// -----------------------------------------------------------------------
struct StepPlan {
    std::vector<ScheduledEntry> scheduled;
    std::vector<int> preempted_pool_ids;  // pool indices of preempted requests

    int batch_size = 0;
    int compute_tokens = 0;
    int access_tokens = 0;
    bool is_decode_only = false;

    void clear() {
        scheduled.clear();
        preempted_pool_ids.clear();
        batch_size = 0;
        compute_tokens = 0;
        access_tokens = 0;
        is_decode_only = false;
    }
};

// -----------------------------------------------------------------------
// Tree ensemble for native perf_model prediction (no Python callback)
// -----------------------------------------------------------------------
struct TreeEnsembleData {
    int n_trees = 0;
    std::vector<int> tree_offsets;      // [n_trees+1] node offsets
    std::vector<int> feature;           // split feature (-2 = leaf)
    std::vector<double> threshold;
    std::vector<int> children_left;     // left child (tree-local index)
    std::vector<int> children_right;
    std::vector<double> value;          // leaf/node value
    double learning_rate = 0.0;
    double init_value = 0.0;
    bool use_log_target = true;
    bool use_extended_features = true;
};

class TreeEnsemble {
public:
    explicit TreeEnsemble(TreeEnsembleData data) : d_(std::move(data)) {}

    // Predict step_time_ms from raw (B, C, A).
    double predict(int B, int C, int A) const;

private:
    TreeEnsembleData d_;
    double traverse_tree(int tree_idx, const double* features) const;
};

// -----------------------------------------------------------------------
// Performance predictor (two modes: Python callback OR native tree)
// -----------------------------------------------------------------------
using PredictFn = std::function<double(int, int, int)>;

class PerfPredictor {
public:
    // Mode 1: Python callback (legacy)
    explicit PerfPredictor(PredictFn fn) : py_predict_(std::move(fn)) {}

    // Mode 2: Native tree ensemble (zero Python calls)
    explicit PerfPredictor(TreeEnsembleData data)
        : ensemble_(std::make_unique<TreeEnsemble>(std::move(data))) {}

    double predict(int B, int C, int A);

private:
    PredictFn py_predict_;
    std::unique_ptr<TreeEnsemble> ensemble_;
    std::unordered_map<uint64_t, double> cache_;

    double raw_predict(int qB, int qC, int qA) {
        if (ensemble_) return ensemble_->predict(qB, qC, qA);
        return py_predict_(qB, qC, qA);
    }

    static uint64_t pack_key(int b, int c, int a) {
        return (static_cast<uint64_t>(static_cast<uint16_t>(b)) << 48) |
               (static_cast<uint64_t>(static_cast<uint16_t>(c)) << 32) |
               static_cast<uint64_t>(static_cast<uint32_t>(a));
    }
};

// -----------------------------------------------------------------------
// Scheduler (operates on pool indices)
// -----------------------------------------------------------------------
class Scheduler {
public:
    explicit Scheduler(const SimConfig& config) : config_(config) {}

    void add_request(int pool_idx);
    bool has_work() const;

    // Build the step plan. Returns false if nothing to schedule.
    bool schedule(StepPlan& plan, std::vector<Request>& pool);

    // Advance tokens, detect finished, set timestamps, remove from running.
    // Appends finished pool indices to finished_out.
    void advance_after_step(const StepPlan& plan, double step_end,
                            std::vector<Request>& pool,
                            std::vector<int>& finished_out);

    void abort_all();

    int kv_used() const { return kv_used_; }
    int num_running() const { return static_cast<int>(running_.size()); }
    int num_waiting() const { return static_cast<int>(waiting_.size()); }

private:
    SimConfig config_;
    std::deque<int> waiting_;    // pool indices
    std::vector<int> running_;   // pool indices
    int kv_used_ = 0;

    static int tokens_for_running(const Request& req, int budget,
                                  bool chunked_prefill);
    static int tokens_for_new(const Request& req, int budget,
                              bool chunked_prefill);
};

// -----------------------------------------------------------------------
// Per-request result (returned to Python)
// -----------------------------------------------------------------------
struct RequestResult {
    std::string request_id;
    double arrival_time;
    double first_token_time;
    double finish_time;
    int prompt_tokens;
    int output_tokens;
    int preemption_count;
};

// -----------------------------------------------------------------------
// Simulation result
// -----------------------------------------------------------------------
struct SimResult {
    std::vector<RequestResult> finished;
    int total_steps = 0;
};

// -----------------------------------------------------------------------
// Main entry points (single-instance)
// -----------------------------------------------------------------------

// Python callback mode (legacy)
SimResult run_simulation(
    const SimConfig& config,
    std::vector<Request> requests,
    PredictFn predict_fn);

// Native tree ensemble mode (zero Python calls)
SimResult run_simulation_native(
    const SimConfig& config,
    std::vector<Request> requests,
    TreeEnsembleData tree_data);

// -----------------------------------------------------------------------
// Cluster simulation types
// -----------------------------------------------------------------------

enum class DispatchStrategy : uint8_t {
    ROUND_ROBIN,
    LEAST_LOADED,
};

struct InstanceGroupConfig {
    SimConfig sim_config;
    int count = 1;
    int predictor_idx = 0;  // index into PerfPredictor vector
};

struct ClusterConfig {
    std::vector<InstanceGroupConfig> groups;
    DispatchStrategy dispatch_strategy = DispatchStrategy::ROUND_ROBIN;
};

// Per-instance runtime state during cluster simulation.
struct InstanceState {
    int instance_idx;
    int group_idx;
    Scheduler scheduler;
    double clock = 0.0;
    std::deque<int> pending;  // pool indices dispatched but not yet injected
    bool step_scheduled = false;
    int total_steps = 0;

    InstanceState(int idx, int gidx, const SimConfig& cfg)
        : instance_idx(idx), group_idx(gidx), scheduler(cfg) {}

    // Move constructor needed because Scheduler has non-trivial members.
    InstanceState(InstanceState&& o) noexcept = default;
    InstanceState& operator=(InstanceState&& o) noexcept = default;
    InstanceState(const InstanceState&) = delete;
    InstanceState& operator=(const InstanceState&) = delete;
};

// Event for the global event queue (min-heap).
// Tie-breaking: REQUEST_ARRIVAL (0) before STEP_COMPLETE (1) at same time.
struct Event {
    double time;
    enum Type : uint8_t { REQUEST_ARRIVAL = 0, STEP_COMPLETE = 1 } type;
    int data;  // REQUEST_ARRIVAL: pool index, STEP_COMPLETE: instance index

    bool operator>(const Event& o) const {
        if (time != o.time) return time > o.time;
        return type > o.type;
    }
};

// Per-instance result for cluster simulation.
struct InstanceSimResult {
    int instance_idx = 0;
    int group_idx = 0;
    std::vector<RequestResult> finished;
    int total_steps = 0;
};

// Cluster simulation result.
struct ClusterSimResult {
    std::vector<InstanceSimResult> instance_results;
    std::vector<RequestResult> all_finished;
    std::vector<int> dispatch_counts;  // per-instance dispatched request count
    int total_instances = 0;
};

// -----------------------------------------------------------------------
// Cluster entry points
// -----------------------------------------------------------------------

ClusterSimResult run_cluster_simulation_native(
    const ClusterConfig& config,
    std::vector<Request> requests,
    std::vector<TreeEnsembleData> tree_data_per_group);

ClusterSimResult run_cluster_simulation(
    const ClusterConfig& config,
    std::vector<Request> requests,
    std::vector<PredictFn> predict_fns_per_group);

}  // namespace sim
