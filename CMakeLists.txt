cmake_minimum_required(VERSION 2.8)

project( DigitalImageProcess_Demos )

set( CMAKE_BUILD_TYPE release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

aux_source_directory(. DIR_SRCS)

add_subdirectory(intensity_transformations)

add_executable( DigitalImageProcess ${DIR_SRCS})
target_link_libraries( DigitalImageProcess IntensityTrans ${OpenCV_LIBS} )