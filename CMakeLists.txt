cmake_minimum_required(VERSION 3.30)
project(Detect_Track_Count)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)


if (MSVC)
    add_compile_options(/W4 /wd4267 /wd4244 /wd4305 /wd4100)
else()
    add_compile_options(-Wall -Wextra -Wno-deprecated-enum-enum-conversion)
endif()

set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets Gui Core)
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

add_executable(Detect_Track_Count ${SOURCES} ${HEADERS})

target_link_libraries(Detect_Track_Count
        Qt5::Widgets
        Qt5::Gui
        Qt5::Core
        ${OpenCV_LIBS}
        Eigen3::Eigen
)
