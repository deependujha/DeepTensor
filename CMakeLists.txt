cmake_minimum_required(VERSION 3.21)

set(SKBUILD_PROJECT_NAME deeptensor)

set(CMAKE_CXX_STANDARD          17) # set c++ standard to 17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # without cpp 17, compiler will give error
set(CMAKE_CXX_EXTENSIONS        OFF) # it's fine to use c++ 17 without extensions (like, bits/stdc++.h)
option(CMAKE_EXPORT_COMPILE_COMMANDS "Generate compile_commands.json" ON) # for clang-tidy

project(${SKBUILD_PROJECT_NAME} VERSION 1.0.0 LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)

add_subdirectory(external/pybind11)

include_directories(external/pybind11)

# add_library(myLib STATIC csrc/add.cc)

pybind11_add_module(_core MODULE csrc/main.cc)

# target_link_libraries(_core PUBLIC myLib)

install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
