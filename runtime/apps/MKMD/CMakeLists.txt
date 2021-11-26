cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(mkmd_skmd_lib)


find_package(OpenCL REQUIRED)

# -----------------
# lib_mkmd_skmd
# -----------------

add_library(${PROJECT_NAME} mkmd.cpp skmd.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME} PUBLIC ../../driver/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE graphcl_lib)

# -----------------
# DemoAPP
# -----------------

add_executable(mkmd_app main.cpp)
set_target_properties(mkmd_app PROPERTIES CXX_STANDARD 17)
target_include_directories(mkmd_app PRIVATE .)
target_link_libraries(mkmd_app PRIVATE ${PROJECT_NAME})

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W4)#/WX)
else(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)#-Werror)
endif(MSVC)
