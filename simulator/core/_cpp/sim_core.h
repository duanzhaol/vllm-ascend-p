// sim_core.h — Data structures and declarations for the C++ simulation core.
//
// Design: All Request objects live in a flat "pool" vector (indexed 0..N-1).
// The Scheduler's running_ and waiting_ queues store pool indices (ints),
// not Request objects. This avoids invalidation when elements are removed.
#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
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
// Performance predictor (Python callback + C++ cache)
// -----------------------------------------------------------------------
using PredictFn = std::function<double(int, int, int)>;

class PerfPredictor {
public:
    explicit PerfPredictor(PredictFn fn) : py_predict_(std::move(fn)) {}

    double predict(int B, int C, int A);

private:
    PredictFn py_predict_;
    std::unordered_map<uint64_t, double> cache_;

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
// Main entry point
// -----------------------------------------------------------------------
SimResult run_simulation(
    const SimConfig& config,
    std::vector<Request> requests,
    PredictFn predict_fn);

}  // namespace sim
