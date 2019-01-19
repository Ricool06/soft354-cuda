enable_testing()

set(TEST_EXECUTABLE soft354_cuda_tests)
add_executable(${TEST_EXECUTABLE} GrayscaleFilterTest.cpp GaussianFilterTest.cpp)
target_link_libraries(${TEST_EXECUTABLE} ${LIBRARY})
target_link_libraries(${TEST_EXECUTABLE} GTest::GTest GTest::Main)
add_test(AllTests ${TEST_EXECUTABLE})
