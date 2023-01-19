cmake_minimum_required(VERSION 3.5) 
PROJECT (my_opencv_package)
find_package(OpenCV REQUIRED )
if (${OpenCV_FOUND})
  message("[MY_MESSAGE] OpenCV version: ${OpenCV_VERSION} found from the path: ${OpenCV_INCLUDE_DIRS}")
  else()
  message(FATAL_ERROR "OpenCV not found!")
endif()

set( SRC_FILES_PATH_IMG_DISP src/imgDisplay.cpp)
set( SRC_FILES_PATH_VID_DISP src/vidDisplay.cpp)

message("[MY_MESSAGE] CMake reading source files from: ${SRC_FILES_PATH_IMG_DISP}")
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

set(p01_t01_exec_name imgDisplay)
add_executable( ${p01_t01_exec_name} ${SRC_FILES_PATH_IMG_DISP} )
target_link_libraries( ${p01_t01_exec_name} ${OpenCV_LIBS} )

set(p01_t02_exec_name vidDisplay)
add_executable( ${p01_t02_exec_name} ${SRC_FILES_PATH_VID_DISP} )
target_link_libraries( ${p01_t02_exec_name} ${OpenCV_LIBS} )


# message("[MY_MESSAGE] Executable ${p01_t01_exec_name} is linked against: ${OpenCV_LIBS}")
message("[MY_MESSAGE] Executable is saved to: ${EXECUTABLE_OUTPUT_PATH}/${p01_t01_exec_name}")
