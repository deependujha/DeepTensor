include(GoogleTest)

enable_testing()

set(
    TEST_CODE
    value_test.cc
    value_fixture_test.cc
    neural_network_test.cc
    )

include_directories("${CMAKE_SOURCE_DIR}/csrc")

add_executable(TEST_CODE ${TEST_CODE})
target_link_libraries(TEST_CODE GTest::gtest_main ${DEEPTENSOR_LIBS})

gtest_discover_tests(TEST_CODE)
message(STATUS "Test executable path: ${CMAKE_BINARY_DIR}")
