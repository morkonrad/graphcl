cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(GraphCL)

add_subdirectory(driver)

option(BUILD_TEST "Build UnitTests ?" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

