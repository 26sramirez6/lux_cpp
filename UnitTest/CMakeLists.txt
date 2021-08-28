# Build lib tests
message(STATUS "Creating Unit Tests Target")
include(GoogleTest)

find_package(Torch REQUIRED)

#build google test target with custom module included from gtest
file(GLOB RL_TEST_SRCS "*.hpp" "*.cpp")

add_executable(rl_test ${RL_TEST_SRCS})
target_link_libraries(rl_test gtest ${TORCH_LIBRARIES})

target_include_directories(rl_test PUBLIC ${CMAKE_SOURCE_DIR})

gtest_discover_tests(rl_test)

