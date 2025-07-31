cmake_minimum_required(VERSION 3.12)
project(cpp_sampler)

set(CMAKE_CXX_STANDARD 14)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(cpp_sampler sampler.cpp)