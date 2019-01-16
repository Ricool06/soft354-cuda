enable_testing()

find_package(GTest MODULE REQUIRED)

set(TEST_EXECUTABLE soft354_cuda_tests)
add_executable(${TEST_EXECUTABLE} ExampleClassTest.cpp)
target_link_libraries(${TEST_EXECUTABLE} PRIVATE GTest::GTest GTest::Main)
add_test(AllTests ${TEST_EXECUTABLE})
