cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(test_common_utils)

add_library(${PROJECT_NAME} utils.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

if (UNIX)
    find_package(TBB REQUIRED) # expect this package to support parallel execution policies of c++ std_lib
    target_link_libraries(${PROJECT_NAME} PRIVATE tbb)
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
    #target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_LIBRARIES})
endif (UNIX)
