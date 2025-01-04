cmake_minimum_required(VERSION 3.21)

set(SKBUILD_PROJECT_NAME deeptensor)
set(DEEPTENSOR_LIBS deeptensor_libs)

set(CMAKE_CXX_STANDARD          17) # set c++ standard to 17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # without cpp 17, compiler will give error
set(CMAKE_CXX_EXTENSIONS        OFF) # it's fine to use c++ 17 without extensions (like, bits/stdc++.h)
option(CMAKE_EXPORT_COMPILE_COMMANDS "Generate compile_commands.json" ON) # for clang-tidy

option(BUILD_TESTS "Build tests" ON) # for googletest
option(BUILD_PYBIND "Build pybind" ON) # for pybind


project(${SKBUILD_PROJECT_NAME} VERSION 1.0.0 LANGUAGES CXX)


add_subdirectory(csrc)

# the -fPIC (Position Independent Code) flag, which is required when linking static library (deeptensor_libs) it into a shared object (pybind module)
set_target_properties(${DEEPTENSOR_LIBS} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_PYBIND)
    message(STATUS "----- Building PyBind -----")

    set(PYBIND11_FINDPYTHON ON)
    add_subdirectory(external/pybind11)
    include_directories(external/pybind11)

    pybind11_add_module(_core MODULE csrc/main.cc)

    target_link_libraries(_core PUBLIC ${DEEPTENSOR_LIBS})
    install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
endif()

if(BUILD_TESTS)
    message(STATUS "----- Building GTests -----")

    add_subdirectory(external/googletests)
    enable_testing()
    add_subdirectory(ctests)
endif()
