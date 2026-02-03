// bindings.cpp — pybind11 bindings for the C++ simulation core.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "sim_core.h"

namespace py = pybind11;

PYBIND11_MODULE(_sim_core, m) {
    m.doc() = "C++ simulation core for LLM inference scheduling";

    // SimConfig
    py::class_<sim::SimConfig>(m, "SimConfig")
        .def(py::init<>())
        .def_readwrite("max_num_batched_tokens",
                       &sim::SimConfig::max_num_batched_tokens)
        .def_readwrite("max_num_seqs", &sim::SimConfig::max_num_seqs)
        .def_readwrite("max_kv_tokens", &sim::SimConfig::max_kv_tokens)
        .def_readwrite("enable_chunked_prefill",
                       &sim::SimConfig::enable_chunked_prefill);

    // Request (input)
    py::class_<sim::Request>(m, "Request")
        .def(py::init<>())
        .def_readwrite("request_id", &sim::Request::request_id)
        .def_readwrite("arrival_time", &sim::Request::arrival_time)
        .def_readwrite("prompt_tokens", &sim::Request::prompt_tokens)
        .def_readwrite("output_tokens", &sim::Request::output_tokens);

    // RequestResult (output)
    py::class_<sim::RequestResult>(m, "RequestResult")
        .def_readonly("request_id", &sim::RequestResult::request_id)
        .def_readonly("arrival_time", &sim::RequestResult::arrival_time)
        .def_readonly("first_token_time", &sim::RequestResult::first_token_time)
        .def_readonly("finish_time", &sim::RequestResult::finish_time)
        .def_readonly("prompt_tokens", &sim::RequestResult::prompt_tokens)
        .def_readonly("output_tokens", &sim::RequestResult::output_tokens)
        .def_readonly("preemption_count", &sim::RequestResult::preemption_count);

    // SimResult
    py::class_<sim::SimResult>(m, "SimResult")
        .def_readonly("finished", &sim::SimResult::finished)
        .def_readonly("total_steps", &sim::SimResult::total_steps);

    // Main entry point
    m.def("run_simulation", &sim::run_simulation,
          py::arg("config"),
          py::arg("requests"),
          py::arg("predict_fn"),
          "Run the discrete-event simulation.\n\n"
          "Args:\n"
          "    config: Scheduling configuration.\n"
          "    requests: List of Request objects (arrival_time, prompt/output tokens).\n"
          "    predict_fn: Callable(B, C, A) -> step_time_ms.\n\n"
          "Returns:\n"
          "    SimResult with finished request data and total step count.");
}
