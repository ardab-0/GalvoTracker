# opencv install instructions: https://www.youtube.com/watch?v=MOOCpt4lDPw&list=LL&index=1&t=166s
# opencv .dll files should be placed next to module file in Windows

cmake_minimum_required(VERSION 3.4)
project(circle-detector)


set(whycon_srcs circle_detector.cpp)

# must set accordıng to system
set (PYTHON_EXECUTABLE "C:/Users/ardab/miniconda3/envs/pybind_demo/python.exe")
#set (PYTHON_EXECUTABLE "/home/arda/miniconda3/envs/pywhyconenv/bin/python")

add_subdirectory(pybind11)

## Check dependencies
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# create config file
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/config.h.cmake ${CMAKE_SOURCE_DIR}/config.h)



pybind11_add_module(circle_detector_module circle_detector_wrapper.cpp circle_detector.cpp)
target_link_libraries(circle_detector_module PRIVATE ${OpenCV_LIBS} )

# add_executable(test test_circle_detector.cpp ${whycon_srcs})
# target_link_libraries(test ${OpenCV_LIBS} )




