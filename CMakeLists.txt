cmake_minimum_required(VERSION 3.30)
project(Sweets_Tracker_and_Counter)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-enum-enum-conversion")

# OpenCV-Pfad setzen
set(OpenCV_DIR "C:/Users/j.nanai/.vcpkg-clion/vcpkg_heilige_scheisse/installed/x64-mingw-dynamic/share/opencv4")
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui imgproc video dnn)

# Include-Ordner hinzufügen
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Bibliotheksverzeichnis setzen
link_directories("C:/Users/j.nanai/.vcpkg-clion/vcpkg_heilige_scheisse/installed/x64-mingw-dynamic/lib")

# Quellcode-Dateien definieren
set(SOURCES
        src/main.cpp
        src/SORT.cpp
        src/Detect_and_Draw.cpp
        src/ImageMode.cpp
        src/CameraMode.cpp
)

# Header-Dateien als zusätzliche Includes angeben (für bessere IDE-Unterstützung)
set(HEADERS
        include/SORT.h
        include/Detect_and_Draw.h
        include/ImageMode.h
        include/CameraMode.h
)

# Executable mit allen Quellen erstellen
add_executable(Sweets_Tracker_and_Counter ${SOURCES} ${HEADERS})

# OpenCV-Bibliotheken linken
target_link_libraries(Sweets_Tracker_and_Counter ${OpenCV_LIBS})
