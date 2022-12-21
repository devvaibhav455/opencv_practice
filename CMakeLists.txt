cmake_minimum_required(VERSION 3.5) 
PROJECT (my_opencv_package)
find_package(OpenCV REQUIRED )
if (${OpenCV_FOUND})
  message("[MY_MESSAGE] OpenCV version: ${OpenCV_VERSION} found from the path: ${OpenCV_INCLUDE_DIRS}")
  else()
  message(FATAL_ERROR "OpenCV not found!")
endif()

set( SRC_FILES_PATH
    src/read_image.cpp    
)
message("[MY_MESSAGE] CMake reading source files from: ${SRC_FILES_PATH}")
# Ref: https://stackoverflow.com/questions/13970377/configuring-an-c-opencv-project-with-cmake
set( INCLUDE_FILES_PATH       
    #  include/header.h
    include/
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} )
message("[MY_MESSAGE] CMake reading include files from: ${CMAKE_CURRENT_SOURCE_DIR}/include ; ${OpenCV_INCLUDE_DIRS}. Check INCLUDE_DIRECTORIES in CMakeLists.txt to double confirm")

link_directories( ${CMAKE_BINARY_DIR}/bin)
message("[MY_MESSAGE] CMake link_directories set to: ${CMAKE_BINARY_DIR}/bin")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(MY_EXECUTABLE_NAME read_display_image)
add_executable( my_opencv_package ${SRC_FILES_PATH} )
message("[MY_MESSAGE] Executable is saved to: ${EXECUTABLE_OUTPUT_PATH}/${MY_EXECUTABLE_NAME}")


target_link_libraries( my_opencv_package ${OpenCV_LIBS} )
message("[MY_MESSAGE] Executable ${MY_EXECUTABLE_NAME} is linked against: ${OpenCV_LIBS}")
