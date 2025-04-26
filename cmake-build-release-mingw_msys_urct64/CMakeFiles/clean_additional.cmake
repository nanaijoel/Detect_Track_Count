# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Release")
  file(REMOVE_RECURSE
  "CMakeFiles\\Detect_Track_Count_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\Detect_Track_Count_autogen.dir\\ParseCache.txt"
  "Detect_Track_Count_autogen"
  )
endif()
