cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(GraphCL)

add_subdirectory(driver)

option(BUILD_APPS "Build applications ?" ON)
if(BUILD_APPS)  
  add_subdirectory(apps)
endif()

option(BUILD_TEST "Build UnitTests ?" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

