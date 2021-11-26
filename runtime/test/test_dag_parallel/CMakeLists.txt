cmake_minimum_required(VERSION 3.1 FATAL_ERROR)


project(test_dag_parallel)

add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME} PRIVATE ../../driver/include)
target_include_directories(${PROJECT_NAME} PRIVATE ../test_common)

target_link_libraries(${PROJECT_NAME} PRIVATE graphcl_lib)
target_link_libraries(${PROJECT_NAME} PRIVATE test_common_utils)

add_test(Test_dag_parallel ${PROJECT_NAME})
