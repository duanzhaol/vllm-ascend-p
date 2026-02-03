cmake_minimum_required(VERSION 3.15)
project(_sim_core LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(_sim_core
    bindings.cpp
    sim_core.cpp
)

target_compile_features(_sim_core PRIVATE cxx_std_17)
target_compile_options(_sim_core PRIVATE -O2 -DNDEBUG)
