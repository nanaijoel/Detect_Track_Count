cmake_minimum_required(VERSION 3.30)
project(Sweets_Tracker_and_Counter)


set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-enum-enum-conversion")


set(OpenCV_DIR "$ENV{OpenCV_DIR}/share/opencv4")
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui imgproc video dnn)


set(CMAKE_AUTOMOC ON)
find_package(Qt6 REQUIRED COMPONENTS Widgets)


find_package(Eigen3 REQUIRED CONFIG)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
include_directories(src)


set(SOURCES
        src/main.cpp
        src/Detect_and_Draw.cpp
        src/ImageMode.cpp
        src/CameraMode.cpp
        src/GUI.cpp
        src/TotalCounter.cpp
        src/BYTETracker/BYTETracker.cpp
        src/BYTETracker/KalmanFilter.cpp
        src/BYTETracker/lapjv.cpp
        src/BYTETracker/Object.cpp
        src/BYTETracker/Rect.cpp
        src/BYTETracker/STrack.cpp
)

set(HEADERS
        include/Detect_and_Draw.h
        include/ImageMode.h
        include/CameraMode.h
        include/GUI.h
        include/TotalCounter.h
        include/BYTETracker/BYTETracker.h
        include/BYTETracker/KalmanFilter.h
        include/BYTETracker/lapjv.h
        include/BYTETracker/Object.h
        include/BYTETracker/Rect.h
        include/BYTETracker/STrack.h
)


add_executable(Sweets_Tracker_and_Counter ${SOURCES} ${HEADERS})


target_link_libraries(Sweets_Tracker_and_Counter
        ${OpenCV_LIBS}
        Eigen3::Eigen
        Qt6::Widgets
)
