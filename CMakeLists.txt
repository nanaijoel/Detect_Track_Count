cmake_minimum_required(VERSION 3.30)
project(Sweets_Tracker_and_Counter)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-enum-enum-conversion")

set(OpenCV_DIR "C:/Users/j.nanai/.vcpkg-clion/vcpkg_heilige_scheisse/installed/x64-mingw-dynamic/share/opencv4")
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui imgproc video dnn)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories("C:/Users/j.nanai/.vcpkg-clion/vcpkg_heilige_scheisse/installed/x64-mingw-dynamic/lib")

add_executable(Sweets_Tracker_and_Counter main.cpp)

target_link_libraries(Sweets_Tracker_and_Counter ${OpenCV_LIBS})
