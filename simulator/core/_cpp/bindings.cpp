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

    // TreeEnsembleData (for native prediction)
    py::class_<sim::TreeEnsembleData>(m, "TreeEnsembleData")
        .def(py::init<>())
        .def_readwrite("n_trees", &sim::TreeEnsembleData::n_trees)
        .def_readwrite("tree_offsets", &sim::TreeEnsembleData::tree_offsets)
        .def_readwrite("feature", &sim::TreeEnsembleData::feature)
        .def_readwrite("threshold", &sim::TreeEnsembleData::threshold)
        .def_readwrite("children_left", &sim::TreeEnsembleData::children_left)
        .def_readwrite("children_right", &sim::TreeEnsembleData::children_right)
        .def_readwrite("value", &sim::TreeEnsembleData::value)
        .def_readwrite("learning_rate", &sim::TreeEnsembleData::learning_rate)
        .def_readwrite("init_value", &sim::TreeEnsembleData::init_value)
        .def_readwrite("use_log_target", &sim::TreeEnsembleData::use_log_target)
        .def_readwrite("use_extended_features",
                       &sim::TreeEnsembleData::use_extended_features);

    // Single-instance entry points
    m.def("run_simulation", &sim::run_simulation,
          py::arg("config"),
          py::arg("requests"),
          py::arg("predict_fn"),
          "Run simulation with Python callback predictor.");

    m.def("run_simulation_native", &sim::run_simulation_native,
          py::arg("config"),
          py::arg("requests"),
          py::arg("tree_data"),
          "Run simulation with native C++ tree ensemble predictor.\n\n"
          "Eliminates all Python callbacks during the simulation loop.");

    // ---------------------------------------------------------------
    // Cluster simulation types
    // ---------------------------------------------------------------

    // DispatchStrategy enum
    py::enum_<sim::DispatchStrategy>(m, "DispatchStrategy")
        .value("ROUND_ROBIN", sim::DispatchStrategy::ROUND_ROBIN)
        .value("LEAST_LOADED", sim::DispatchStrategy::LEAST_LOADED);

    // InstanceGroupConfig
    py::class_<sim::InstanceGroupConfig>(m, "InstanceGroupConfig")
        .def(py::init<>())
        .def_readwrite("sim_config",
                       &sim::InstanceGroupConfig::sim_config)
        .def_readwrite("count", &sim::InstanceGroupConfig::count)
        .def_readwrite("predictor_idx",
                       &sim::InstanceGroupConfig::predictor_idx);

    // ClusterConfig
    py::class_<sim::ClusterConfig>(m, "ClusterConfig")
        .def(py::init<>())
        .def_readwrite("groups", &sim::ClusterConfig::groups)
        .def_readwrite("dispatch_strategy",
                       &sim::ClusterConfig::dispatch_strategy);

    // InstanceSimResult
    py::class_<sim::InstanceSimResult>(m, "InstanceSimResult")
        .def_readonly("instance_idx",
                      &sim::InstanceSimResult::instance_idx)
        .def_readonly("group_idx", &sim::InstanceSimResult::group_idx)
        .def_readonly("finished", &sim::InstanceSimResult::finished)
        .def_readonly("total_steps",
                      &sim::InstanceSimResult::total_steps);

    // ClusterSimResult
    py::class_<sim::ClusterSimResult>(m, "ClusterSimResult")
        .def_readonly("instance_results",
                      &sim::ClusterSimResult::instance_results)
        .def_readonly("all_finished",
                      &sim::ClusterSimResult::all_finished)
        .def_readonly("total_instances",
                      &sim::ClusterSimResult::total_instances);

    // Cluster entry points
    m.def("run_cluster_simulation_native",
          &sim::run_cluster_simulation_native,
          py::arg("config"),
          py::arg("requests"),
          py::arg("tree_data_per_group"),
          "Run cluster simulation with native C++ tree ensemble.\n\n"
          "Each instance group uses its own tree ensemble predictor.");

    m.def("run_cluster_simulation",
          &sim::run_cluster_simulation,
          py::arg("config"),
          py::arg("requests"),
          py::arg("predict_fns_per_group"),
          "Run cluster simulation with Python callback predictors.\n\n"
          "Each instance group uses its own Python callback.");
}
