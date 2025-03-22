cmake_minimum_required(VERSION 3.30)
project(Sweets_Tracker_and_Counter)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-enum-enum-conversion")

# OpenCV
set(OpenCV_DIR "$ENV{OpenCV_DIR}/share/opencv4")
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui imgproc video dnn)

find_package(Eigen3 REQUIRED CONFIG)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

link_directories($ENV{OpenCV_LIBS})


set(SOURCES
        src/main.cpp
        src/SORT.cpp
        src/Detect_and_Draw.cpp
        src/ImageMode.cpp
        src/CameraMode.cpp
)

set(HEADERS
        include/SORT.h
        include/Detect_and_Draw.h
        include/ImageMode.h
        include/CameraMode.h
)

add_executable(Sweets_Tracker_and_Counter ${SOURCES} ${HEADERS})

target_link_libraries(Sweets_Tracker_and_Counter ${OpenCV_LIBS} Eigen3::Eigen)
