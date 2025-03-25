cmake_minimum_required(VERSION 3.30)
project(Sweets_Tracker_and_Counter)

# Toolchain VCPKG-Pfad (wichtig für CLion + vcpkg integration)
set(CMAKE_TOOLCHAIN_FILE "C:/Users/j.nanai/.vcpkg-clion/vcpkg_heilige_scheisse/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-enum-enum-conversion")

# OpenCV
set(OpenCV_DIR "$ENV{OpenCV_DIR}/share/opencv4")
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui imgproc video dnn)

# Qt
set(CMAKE_AUTOMOC ON)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# Eigen
find_package(Eigen3 REQUIRED CONFIG)

# Includes
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)


set(SOURCES
        src/main.cpp
        src/SORT.cpp
        src/Detect_and_Draw.cpp
        src/ImageMode.cpp
        src/CameraMode.cpp
        src/TrackRecoveryHelper.cpp
        src/GUI.cpp
)

set(HEADERS
        include/SORT.h
        include/Detect_and_Draw.h
        include/ImageMode.h
        include/CameraMode.h
        include/TrackRecoveryHelper.h
        include/GUI.h
)


add_executable(Sweets_Tracker_and_Counter ${SOURCES} ${HEADERS})


target_link_libraries(Sweets_Tracker_and_Counter
        ${OpenCV_LIBS}
        Eigen3::Eigen
        Qt6::Widgets
)
